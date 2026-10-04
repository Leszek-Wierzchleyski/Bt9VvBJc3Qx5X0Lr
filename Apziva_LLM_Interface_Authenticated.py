#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""LLM interface for the Bitcoin trading application.

The LLM is an interface layer.  It can read portfolio information and can
propose configuration changes, but mutating actions require explicit user
confirmation before the action tool is executed.
"""

import json
import math
import os

from Apziva_Security_Authenticated import audit_event

from llama_cpp import Llama

from Apziva_Binance_API import BinanceMarketData
from Apziva_LLM_Tools_Authenticated import (
    get_portfolio as _get_portfolio,
    get_portfolio_performance as _get_portfolio_performance,
    get_trading_status as _get_trading_status,
    get_dca_status as _get_dca_status,
    adjust_capital as _adjust_capital,
    set_dca_amount as _set_dca_amount,
    get_risk_level as _get_risk_level,
    set_risk_level as _set_risk_level,
    start_trading as _start_trading,
    stop_trading as _stop_trading,
    sell_btc as _sell_btc,
    buy_btc as _buy_btc,
)

MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
MODEL_PATH = os.getenv(
    "LLAMA_MODEL_PATH",
    "models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
)
MAX_USER_MESSAGE_CHARS = 4000
WELCOME_MESSAGE = """Welcome to the Bitcoin Trading Agent.

This interface provides a natural-language control layer for a simulated Bitcoin trading system. You can ask the agent about your portfolio, performance, strategy, risk level, and trading status, or make changes to your account configuration.

No real trades are placed. Market data is read-only, and all trading activity takes place within the simulated portfolio.

## Risk Levels

The system has three risk levels. Your selected risk level determines which trading strategy is active:

### Low Risk — Level 1
Regime-Adaptive DCA

Dollar-Cost Averaging (DCA) involves investing a predetermined amount at regular intervals rather than attempting to time the market.

The strategy adapts the frequency of scheduled DCA purchases according to the detected Bitcoin market regime:

- Bullish: DCA every 10 days
- Neutral: DCA every 14 days
- Bearish: DCA every 21 days

A portfolio drawdown safeguard can temporarily stop new purchases during significant declines.

### Medium Risk — Level 2
Volatility-Aware Trading

Includes everything in Level 1, with additional responses to significant daily Bitcoin price movements.

The strategy responds to significant daily Bitcoin price movements:

- A daily downside move of 4% or more doubles the next scheduled DCA purchase.
- A daily upside move of 5% or more triggers a sale equal to 15% of the gain attributable to that day's uptick.

If the relevant volatility conditions are not triggered, the underlying Level 1 strategy continues to operate.

### High Risk — Level 3
ML-Enhanced Trading

Includes everything in Level 2, with an additional machine-learning accumulation layer.

Two independent models — a tree-based model and an LSTM — must agree before an additional opportunistic purchase is made.

If the ML conditions are not met, the underlying Level 2 strategy continues to operate. If Level 2's volatility conditions are not triggered, it continues according to the underlying Level 1 strategy.

In this way, the higher risk levels add additional trading behaviour on top of the lower levels rather than replacing them.

## What You Can Configure

You can use natural language to:

- Add or withdraw simulated capital
- Set your DCA amount
- Change your risk level
- Start or stop trading
- View your portfolio and performance
- Check your current DCA regime and schedule
- Sell some or all of your Bitcoin

The DCA frequency is determined automatically by the trading strategy and cannot be configured manually.

## Important

The agent separates strategy decisions from account controls. Changing your risk level changes which trading strategy is active; it does not manually tell the system when to buy or sell.

Actions that change your simulated portfolio or account configuration require explicit confirmation before they are executed.

You can simply ask questions or give instructions in ordinary language — for example:

“What is my current portfolio?”

“Set my risk level to high.”

“Add £5,000 to the account.”

“Set my DCA amount to £250.”

“How is the current DCA schedule determined?”

“Stop trading.”

The system will explain what it intends to do before executing any action that changes your account.
"""


SYSTEM_PROMPT = """
You are a Bitcoin trading assistant.

Your job is to answer the user's questions clearly and directly.

When a read-only tool is used, use the information returned by the tool to answer the user's question. Do not describe the tool call. State relevant values explicitly.

Do not invent information. If the available tools do not provide enough information to answer a question, say so clearly.

You do not make trading decisions yourself.
You do not directly execute trading actions.

ACTION CONFIRMATION RULE:
- Action tools change user configuration or portfolio state and are potentially consequential.
- Never claim that an action has been completed merely because an action was requested.
- The application requires explicit user confirmation before an action tool can execute.
- The application, not the LLM, controls whether an action is executed.

TOOL SELECTION RULES:
- Use get_portfolio for current cash, Bitcoin holdings, current Bitcoin price, or current total portfolio value.
- Use get_portfolio_performance for historical portfolio performance, returns, profit, or loss over a specified period.
- Use get_trading_status for whether automated trading is enabled or disabled.
- Use get_dca_status for the current DCA amount and the strategy's current regime/effective DCA frequency, which are informational only.
- Use adjust_capital when the user explicitly asks to add/deposit capital or withdraw capital. Use a positive amount to add capital and a negative amount to withdraw capital. The application will request confirmation before execution.
- Use set_dca_amount when the user explicitly asks to change the base DCA amount. This changes future scheduled DCA purchases only.
- DCA frequency is controlled internally by the strategy's regime detector and is not user-configurable through the interface.
- Never create, remove, or alter completed DCA purchases retroactively.
- Use get_risk_level when the user asks which risk level/strategy is active.
- Use set_risk_level when the user explicitly asks to change to Level 1, 2, or 3. The application will request confirmation before execution.
- Use start_trading when the user explicitly asks to start or enable automated trading. The application will request confirmation before execution.
- Use stop_trading when the user explicitly asks to stop or disable automated trading. The application will request confirmation before execution.
- Use buy_btc when the user explicitly asks to buy/trade/purchase Bitcoin for a specified cash amount. A phrase such as “trade £500” means make a £500 simulated BTC purchase, not add £500 of capital. The application will request confirmation before execution.
- Use sell_btc when the user explicitly asks to sell some or all of their Bitcoin. The user may specify a BTC amount or a percentage of the current holding. The application will request confirmation before execution.
- Level 1 is regime-adaptive DCA plus a global drawdown safeguard. It has no normal profit-taking or ML overlay.
- Level 2 adds deterministic volatility-based downside/upside rules to Level 1.
- Level 3 adds an opportunistic Trees/LSTM machine-learning overlay to Level 2. ML can add an additional buy when Level 2 already produces an eligible BUY; it does not override Level 2 HOLD or SELL decisions.
- Changing risk level never liquidates or alters existing BTC, cash, completed transactions, or portfolio history; it only changes future strategy behaviour.
- Use the exact tool names provided. Do not invent, rename, or substitute tool names.
- If the user requests multiple independent actions in one message, return a JSON object with an `actions` array. Each item must contain `name` and `parameters`. Preserve the order requested by the user.
- If the user requests only one action, return the normal JSON object with `name` and `parameters`.
"""


class LlamaInterface:
    """Conversation interface around an already-selected trading strategy.

    The active strategy is supplied by the application.  This class therefore
    does not choose the user's risk level or instantiate a trading strategy.
    """

    def __init__(
        self,
        trading_strategy,
        market_data=None,
        device=None,
        risk_manager=None,
        tokenizer=None,
        model=None,
        session_id=None,
    ):
        # All portfolio/configuration state belongs to this application
        # session. The LLM model/tokenizer may be shared as immutable resources,
        # but they never contain user portfolio state.
        self.trading_strategy = trading_strategy
        self.market_data = market_data or BinanceMarketData()
        self.risk_manager = risk_manager
        self.session_id = session_id

        self.pending_action = None
        self.pending_actions = []

        # The production Docker runtime uses a CPU-friendly GGUF model through
        # llama.cpp.  ``tokenizer`` is retained as a compatibility argument for
        # older callers, but is no longer required.
        self.device = device or "cpu"

        if model is None:
            print(f"Loading quantised Llama from {MODEL_PATH}...")
            self.model = Llama(
                model_path=MODEL_PATH,
                n_ctx=4096,
                n_threads=max(1, min(8, os.cpu_count() or 1)),
                chat_format="llama-3",
                verbose=False,
            )
            print("Quantised Llama loaded successfully.")
        else:
            self.model = model

        # Kept for compatibility with callers that still pass a tokenizer.
        self.tokenizer = tokenizer

        audit_event("llm_interface_initialized", session_id=self.session_id)

    def get_welcome_message(self):
        """Return the user-facing welcome message."""
        return WELCOME_MESSAGE

    # ------------------------------------------------------------------
    # Tool wrappers
    # ------------------------------------------------------------------

    def get_portfolio(self):
        """
        Return the current portfolio state.

        Use this tool for current cash, Bitcoin holdings, current Bitcoin
        price, or current total portfolio value.
        """
        return _get_portfolio(self.trading_strategy, self.market_data)

    def get_portfolio_performance(self, period: str = "30d"):
        """
        Get historical Bitcoin portfolio performance over a specified period.

        Args:
            period: The performance lookback period. Valid values are
                "7d", "30d", "90d", "1y", or "all".
        """
        return _get_portfolio_performance(self.trading_strategy, period)

    def get_trading_status(self):
        """
        Return the current automated trading status.

        This is the authoritative tool for determining whether automated
        trading is currently enabled or disabled.
        """
        return _get_trading_status(self.trading_strategy)

    def get_dca_status(self):
        """
        Return the current Bitcoin DCA configuration and state.

        Use this tool for DCA amount, current regime/effective DCA frequency, or last/next DCA state. The frequency is informational only.
        """
        return _get_dca_status(self.trading_strategy)

    def adjust_capital(self, amount: float):
        """
        Add or withdraw simulated capital after explicit user confirmation.

        Args:
            amount: Capital adjustment. Positive adds capital; negative withdraws it.
        """
        return _adjust_capital(self.trading_strategy, amount)

    def set_dca_amount(self, amount: float):
        """
        Set the base DCA amount for future scheduled DCA purchases.

        Args:
            amount: The new base DCA amount in currency units.

        This does not alter completed purchases or portfolio history.
        """
        return _set_dca_amount(self.trading_strategy, self.market_data, amount)


    def get_risk_level(self):
        """
        Return the currently active risk level and its strategy description.
        """
        return _get_risk_level(self.risk_manager)

    def start_trading(self):
        """Enable automated trading after explicit user confirmation."""
        return _start_trading(self.trading_strategy)

    def stop_trading(self):
        """Disable automated trading after explicit user confirmation."""
        return _stop_trading(self.trading_strategy)

    def buy_btc(self, amount: float):
        """Make a manual simulated BTC purchase for a cash amount.

        Args:
            amount: The amount of existing cash to spend on Bitcoin.

        This converts existing available cash into Bitcoin at the current
        read-only market price. It does not add capital or alter DCA/risk
        configuration.
        """
        return _buy_btc(self.trading_strategy, self.market_data, amount)

    def sell_btc(self, btc_amount: float = None, percentage: float = None):
        """Sell a user-specified BTC amount or percentage after confirmation.

        Args:
            btc_amount: The amount of Bitcoin to sell, in BTC. Provide this
                when the user specifies an absolute BTC quantity.
            percentage: The percentage of the current Bitcoin holding to sell,
                from 0 to 100. Provide this when the user specifies a
                percentage rather than an absolute BTC quantity.
        """
        return _sell_btc(
            self.trading_strategy,
            self.market_data,
            btc_amount=btc_amount,
            percentage=percentage,
        )

    def set_risk_level(self, level: int):
        """
        Switch the active strategy for future trading decisions.

        Args:
            level: Target risk level. Valid values are 1, 2, or 3.

        Existing BTC, cash, cost basis, completed transactions, and portfolio
        history are preserved.
        """
        result = _set_risk_level(self.risk_manager, level)
        self.trading_strategy = self.risk_manager.current_strategy
        return result

    # ------------------------------------------------------------------
    # Tool registry / execution
    # ------------------------------------------------------------------

    def _tools(self):
        return {
            "get_portfolio": self.get_portfolio,
            "get_portfolio_performance": self.get_portfolio_performance,
            "get_trading_status": self.get_trading_status,
            "get_dca_status": self.get_dca_status,
            "adjust_capital": self.adjust_capital,
            "set_dca_amount": self.set_dca_amount,
            "get_risk_level": self.get_risk_level,
            "set_risk_level": self.set_risk_level,
            "start_trading": self.start_trading,
            "stop_trading": self.stop_trading,
            "buy_btc": self.buy_btc,
            "sell_btc": self.sell_btc,
        }

    def execute_read_only_tool(self, tool_name, parameters):
        tools = self._tools()

        if tool_name not in {
            "get_portfolio",
            "get_portfolio_performance",
            "get_trading_status",
            "get_dca_status",
            "get_risk_level",
        }:
            raise ValueError(f"'{tool_name}' is not a read-only tool.")

        if tool_name == "get_portfolio":
            if parameters:
                raise ValueError("get_portfolio does not accept parameters.")
            return tools[tool_name]()

        if tool_name == "get_portfolio_performance":
            period = parameters.get("period")
            valid_periods = {"7d", "30d", "90d", "1y", "all"}
            if period not in valid_periods:
                raise ValueError(
                    f"Invalid period '{period}'. Choose from: {sorted(valid_periods)}"
                )
            return tools[tool_name](period=period)

        if tool_name in {"get_trading_status", "get_dca_status", "get_risk_level"}:
            if parameters:
                raise ValueError(f"{tool_name} does not accept parameters.")
            return tools[tool_name]()

        raise ValueError(f"No execution rule defined for '{tool_name}'.")

    # ------------------------------------------------------------------
    # Confirmation handling
    # ------------------------------------------------------------------

    @staticmethod
    def _normalise_yes_no(message):
        return message.strip().lower()

    def _confirmation_prompt(self, action, parameters):
        if action == "adjust_capital":
            amount = float(parameters["amount"])
            if amount > 0:
                description = f"add £{amount:,.2f} to your simulated trading account"
            else:
                description = f"withdraw £{abs(amount):,.2f} from your simulated trading account"

            return (
                f"You asked to {description}.\n"
                "This changes the cash available to the strategy but does not rewrite "
                "historical portfolio values or alter the strategy rules.\n"
                "Please confirm that you want to make this change.\n"
                "Type `yes` or `no`."
            )

        if action == "set_dca_amount":
            amount = float(parameters["amount"])
            current_amount = float(self.trading_strategy.dca_amount)
            return (
                f"You asked to change the base DCA amount from "
                f"£{current_amount:,.2f} to £{amount:,.2f}.\n"
                "This changes future scheduled DCA purchases only. Completed DCA "
                "purchases and portfolio history will not be changed.\n"
                "Please confirm that you want to make this change.\n"
                "Type `yes` or `no`."
            )


        if action == "buy_btc":
            amount = float(parameters["amount"])
            current_cash = float(self.trading_strategy.cash)
            current_price = float(self.market_data.get_candles(interval="1d", limit=1)["Close"].iloc[-1])
            estimated_btc = amount / current_price
            return (
                f"You asked to make a £{amount:,.2f} simulated Bitcoin purchase.\n"
                f"Current BTC price: £{current_price:,.2f}. Estimated BTC purchased: {estimated_btc:.8f} BTC.\n"
                f"Available cash before purchase: £{current_cash:,.2f}.\n"
                "This uses existing cash; it does not add capital or change your DCA settings or risk level.\n"
                "Please confirm that you want to make this purchase.\n"
                "Type `yes` or `no`."
            )

        if action == "sell_btc":
            held = float(self.trading_strategy.btc_held)
            current_price = float(self.market_data.get_candles(interval="1d", limit=1)["Close"].iloc[-1])
            if "percentage" in parameters:
                percentage = float(parameters["percentage"])
                btc_amount = held * percentage / 100.0
                basis = float(getattr(self.trading_strategy, "btc_cost_basis", 0.0)) * (btc_amount / held) if held else 0.0
                proceeds = btc_amount * current_price
                return (
                    f"You asked to sell {percentage:g}% of your current Bitcoin holding "
                    f"({btc_amount:.8f} BTC).\n"
                    f"Current BTC price: £{current_price:,.2f}. Estimated proceeds: £{proceeds:,.2f}.\n"
                    f"Estimated realised P/L: £{(proceeds - basis):,.2f}.\n"
                    "This is a manual sale. It will not change your contributed capital, "
                    "risk level, or the strategy's historical records.\n"
                    "Please confirm that you want to make this sale.\n"
                    "Type `yes` or `no`."
                )
            btc_amount = float(parameters["btc_amount"])
            if btc_amount > held:
                raise ValueError(f"You only hold {held:.8f} BTC.")
            basis = float(getattr(self.trading_strategy, "btc_cost_basis", 0.0)) * (btc_amount / held) if held else 0.0
            proceeds = btc_amount * current_price
            return (
                f"You asked to sell {btc_amount:.8f} BTC ({(btc_amount / held) * 100:.2f}% of your current holding).\n"
                f"Current BTC price: £{current_price:,.2f}. Estimated proceeds: £{proceeds:,.2f}.\n"
                f"Estimated realised P/L: £{(proceeds - basis):,.2f}.\n"
                "This is a manual sale. It will not change your contributed capital, "
                "risk level, or the strategy's historical records.\n"
                "Please confirm that you want to make this sale.\n"
                "Type `yes` or `no`."
            )

        if action == "start_trading":
            return (
                "You asked to start automated trading.\n"
                "This enables the currently selected trading strategy to make future automated trading decisions.\n"
                "No existing Bitcoin holdings, cash, completed transactions, or portfolio history will be changed.\n"
                "Please confirm that you want to start trading.\n"
                "Type `yes` or `no`."
            )

        if action == "stop_trading":
            return (
                "You asked to stop automated trading.\n"
                "This prevents the strategy from making new automated trading decisions until trading is started again.\n"
                "Existing Bitcoin holdings, cash, completed transactions, and portfolio history will remain unchanged.\n"
                "Please confirm that you want to stop trading.\n"
                "Type `yes` or `no`."
            )

        if action == "set_risk_level":
            level = int(parameters["level"])
            current_level = int(self.risk_manager.current_risk_level)
            description = {
                1: "Regime-adaptive DCA + safeguard",
                2: "Level 1 + volatility-based rules",
                3: "Level 2 + opportunistic Trees/LSTM machine-learning overlay",
            }[level]
            return (
                f"You asked to change the risk level from Level {current_level} "
                f"to Level {level}.\n"
                f"Level {level}: {description}.\n"
                "Existing BTC, cash, cost basis, completed transactions, and "
                "portfolio history will remain unchanged. Only future strategy "
                "decisions will use the new risk level.\n"
                "Please confirm that you want to make this change.\n"
                "Type `yes` or `no`."
            )

        raise ValueError(f"No confirmation prompt defined for '{action}'.")

    def _validate_action_request(self, action, parameters):
        if action == "buy_btc":
            if set(parameters) != {"amount"}:
                raise ValueError("buy_btc requires exactly one parameter: amount.")
            try:
                amount = float(parameters["amount"])
            except (TypeError, ValueError):
                raise ValueError("Purchase amount must be numeric.")
            if not math.isfinite(amount) or amount <= 0:
                raise ValueError("Purchase amount must be finite and greater than 0.")
            # Do not validate available cash here. Compound requests are
            # validated as a queue before execution, so a purchase may depend
            # on an earlier queued capital contribution. The authoritative
            # cash check is performed by buy_btc() at execution time, after
            # earlier confirmed actions have updated the live account state.
            return {"amount": round(amount, 2)}

        if action == "adjust_capital":
            if set(parameters) != {"amount"}:
                raise ValueError("adjust_capital requires exactly one parameter: amount.")

            try:
                amount = float(parameters["amount"])
            except (TypeError, ValueError):
                raise ValueError("Capital adjustment must be a numeric value.")

            if not math.isfinite(amount) or amount == 0:
                raise ValueError("Capital adjustment must be finite and non-zero.")

            return {"amount": round(amount, 2)}

        if action == "set_dca_amount":
            if set(parameters) != {"amount"}:
                raise ValueError("set_dca_amount requires exactly one parameter: amount.")

            try:
                amount = float(parameters["amount"])
            except (TypeError, ValueError):
                raise ValueError("DCA amount must be numeric.")

            if not math.isfinite(amount) or amount <= 0:
                raise ValueError("DCA amount must be finite and greater than 0.")

            return {"amount": round(amount, 2)}


        if action == "sell_btc":
            keys = set(parameters)
            if keys not in ({"btc_amount"}, {"percentage"}):
                raise ValueError("sell_btc requires exactly one of btc_amount or percentage.")
            if "btc_amount" in parameters:
                try:
                    amount = float(parameters["btc_amount"])
                except (TypeError, ValueError):
                    raise ValueError("BTC amount must be numeric.")
                if not math.isfinite(amount) or amount <= 0:
                    raise ValueError("BTC amount must be finite and greater than 0.")
                held = float(self.trading_strategy.btc_held)
                if held <= 0:
                    raise ValueError("There is no Bitcoin position to sell.")
                if amount > held:
                    raise ValueError(f"You only hold {held:.8f} BTC.")
                return {"btc_amount": amount}

            try:
                percentage = float(parameters["percentage"])
            except (TypeError, ValueError):
                raise ValueError("Sale percentage must be numeric.")
            if not math.isfinite(percentage) or percentage <= 0 or percentage > 100:
                raise ValueError("Sale percentage must be greater than 0 and no more than 100.")
            if float(self.trading_strategy.btc_held) <= 0:
                raise ValueError("There is no Bitcoin position to sell.")
            return {"percentage": percentage}

        if action in {"start_trading", "stop_trading"}:
            if parameters:
                raise ValueError(f"{action} does not accept parameters.")
            if action == "start_trading" and getattr(self.trading_strategy, "trading_enabled", False):
                raise ValueError("Trading is already enabled; no change is required.")
            if action == "stop_trading" and not getattr(self.trading_strategy, "trading_enabled", False):
                raise ValueError("Trading is already stopped; no change is required.")
            return {}

        if action == "set_risk_level":
            if set(parameters) != {"level"}:
                raise ValueError("set_risk_level requires exactly one parameter: level.")

            try:
                level = int(parameters["level"])
            except (TypeError, ValueError):
                raise ValueError("Risk level must be an integer: 1, 2, or 3.")

            if level not in {1, 2, 3}:
                raise ValueError("Risk level must be 1, 2, or 3.")

            if self.risk_manager is None:
                raise ValueError("Risk manager is not configured.")

            return {"level": level}

        raise ValueError(f"Unsupported action '{action}'.")

    def _action_description(self, action, parameters):
        if action == "buy_btc":
            return f"Make a £{float(parameters['amount']):,.2f} simulated Bitcoin purchase."
        if action == "set_dca_amount":
            return f"Change the DCA amount to £{float(parameters['amount']):,.2f}."

        if action == "adjust_capital":
            amount = float(parameters["amount"])
            return f"Add £{amount:,.2f} of capital." if amount > 0 else f"Withdraw £{abs(amount):,.2f} of capital."
        if action == "sell_btc":
            if "percentage" in parameters:
                return f"Sell {float(parameters['percentage']):g}% of the current Bitcoin holding."
            return f"Sell {float(parameters['btc_amount']):.8f} BTC."
        if action == "set_risk_level":
            return f"Change the risk level to Level {int(parameters['level'])}."
        return f"Perform {action}."

    def _create_pending_actions(self, actions):
        validated = []
        for action, parameters in actions:
            validated.append({
                "action": action,
                "parameters": self._validate_action_request(action, parameters),
            })
        if not validated:
            raise ValueError("No actions were requested.")

        self.pending_actions = validated[1:]
        self.pending_action = validated[0]
        first_prompt = self._confirmation_prompt(
            self.pending_action["action"], self.pending_action["parameters"]
        )
        if len(validated) == 1:
            return first_prompt

        first_description = self._action_description(
            self.pending_action["action"], self.pending_action["parameters"]
        )
        lines = [
            f"We will deal with the first request ({first_description}) first.",
            "You have asked me to make the following changes:",
        ]
        lines.extend(f"- {self._action_description(item['action'], item['parameters'])}" for item in validated)
        lines.extend([
            "",
            "We will handle the remaining requests one at a time, with a separate confirmation for each.",
            "",
            first_prompt,
        ])
        return "\n".join(lines)

    def _create_pending_action(self, action, parameters):
        return self._create_pending_actions([(action, parameters)])

    def _action_is_noop(self, action, parameters):
        """Return True when a queued action would make no state change.

        This is especially important for compound LLM requests: after one
        action has been confirmed, a duplicate action may remain in the queue.
        We evaluate the queue against the *current* state immediately before
        presenting the next confirmation, so a duplicate such as
        ``£250 -> £500`` followed by ``£500 -> £500`` is silently skipped.
        """
        if action == "set_dca_amount":
            return round(float(parameters["amount"]), 2) == round(
                float(self.trading_strategy.dca_amount), 2
            )


        if action == "set_risk_level":
            return (
                self.risk_manager is not None
                and int(parameters["level"]) == int(self.risk_manager.current_risk_level)
            )

        if action == "start_trading":
            return bool(getattr(self.trading_strategy, "trading_enabled", False))

        if action == "stop_trading":
            return not bool(getattr(self.trading_strategy, "trading_enabled", False))

        # Capital adjustments are never treated as no-ops because even a
        # repeated capital contribution is an explicit accounting event.
        return False

    def _next_pending_action_prompt(self):
        while self.pending_actions:
            candidate = self.pending_actions.pop(0)

            # Re-check against live state.  Earlier confirmed actions may have
            # made a later queued action redundant.
            if self._action_is_noop(candidate["action"], candidate["parameters"]):
                continue

            self.pending_action = candidate
            return self._confirmation_prompt(
                self.pending_action["action"], self.pending_action["parameters"]
            )

        self.pending_action = None
        return None

    def _confirm_pending_action(self):
        if self.pending_action is None:
            return "There is no action awaiting confirmation."

        action = self.pending_action["action"]
        parameters = self.pending_action["parameters"]
        audit_event(
            "action_confirmation_requested",
            session_id=self.session_id,
            detail=f"action={action}",
        )
        try:
            if action == "buy_btc":
                result = self.buy_btc(**parameters)
            elif action == "adjust_capital":
                result = self.adjust_capital(**parameters)
            elif action == "set_dca_amount":
                result = self.set_dca_amount(**parameters)
            elif action == "set_risk_level":
                result = self.set_risk_level(**parameters)
            elif action == "start_trading":
                result = self.start_trading()
            elif action == "stop_trading":
                result = self.stop_trading()
            elif action == "sell_btc":
                result = self.sell_btc(**parameters)
            else:
                raise ValueError(f"Unsupported action '{action}'.")
        except Exception as exc:
            audit_event(
                "action_execution_failed",
                session_id=self.session_id,
                status="error",
                detail=f"action={action}",
            )
            self.pending_action = None
            next_prompt = self._next_pending_action_prompt()
            # Do not expose raw exception text to the user. Exceptions may
            # contain implementation details, filesystem paths, API responses,
            # or other information that should remain inside the application.
            message = "The requested action could not be completed. Please try again."
            if next_prompt:
                return message + "\n\nYou also asked for another change.\n\n" + next_prompt
            self.pending_actions = []
            return message

        audit_event(
            "action_executed",
            session_id=self.session_id,
            detail=f"action={action}",
        )
        # A risk-level switch creates a new strategy object. Rebind the
        # interface immediately so subsequent calls cannot operate on the
        # previous session strategy.
        if action == "set_risk_level" and self.risk_manager is not None:
            self.trading_strategy = self.risk_manager.current_strategy

        self.pending_action = None
        if action == "buy_btc":
            completed = (
                f"{result['message']}\nAmount invested: £{result['amount']:,.2f}\n"
                f"Purchase price: £{result['price']:,.2f}\nBTC purchased: {result['btc_bought']:.8f}\n"
                f"Remaining cash: £{result['new_cash']:,.2f}"
            )
        elif action == "adjust_capital":
            completed = f"{result['message']}\nPrevious cash: £{result['previous_cash']:,.2f}\nNew cash: £{result['new_cash']:,.2f}."
        elif action == "set_dca_amount":
            completed = f"{result['message']}\nPrevious DCA amount: £{result['previous_dca_amount']:,.2f}\nNew DCA amount: £{result['new_dca_amount']:,.2f}."

        elif action == "sell_btc":
            completed = (
                f"{result['message']}\nBTC sold: {result['btc_sold']:.8f}\n"
                f"Sale price: £{result['price']:,.2f}\n"
                f"Proceeds: £{result['proceeds']:,.2f}\n"
                f"Realised P/L: £{result['realised_profit']:,.2f}\n"
                f"Remaining BTC: {result['new_btc_held']:.8f}"
            )
        elif action == "set_risk_level":
            completed = f"{result['message']}\nPrevious risk level: Level {result['previous_risk_level']}\nNew risk level: Level {result['new_risk_level']} ({result['description']})."
        else:
            completed = result["message"]

        next_prompt = self._next_pending_action_prompt()
        if next_prompt:
            return completed + "\n\nYou also asked for another change.\n\n" + next_prompt
        return completed

    def _cancel_pending_action(self):
        audit_event(
            "action_cancelled",
            session_id=self.session_id,
            detail=f"action={self.pending_action['action'] if self.pending_action else 'none'}",
        )
        self.pending_action = None
        next_prompt = self._next_pending_action_prompt()
        if next_prompt:
            return ("The current action has been cancelled. No change was made.\n\n"
                    "You also asked for another change. We can deal with that request separately.\n\n"
                    + next_prompt)
        self.pending_actions = []
        return "The pending action has been cancelled. No changes were made."

    # ------------------------------------------------------------------
    # LLM interaction
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_tool_call(response):
        """Parse one or more tool-call JSON objects returned by Llama.

        Llama sometimes returns multiple valid JSON objects consecutively
        rather than wrapping them in the requested ``actions`` array.  Treat
        that output as a valid multi-action response rather than rejecting it.
        """
        response = response.strip()
        if not response:
            raise ValueError("Llama returned an empty tool-call response.")

        # First handle the preferred single-object or actions-array format.
        try:
            parsed = json.loads(response)
        except json.JSONDecodeError:
            parsed = None

        if parsed is not None:
            return LlamaInterface._normalise_parsed_actions(parsed)

        # Llama may emit several complete JSON objects separated by semicolons
        # or whitespace instead of wrapping them in the preferred actions array.
        # Accept those objects individually while preserving their order.
        decoder = json.JSONDecoder()
        objects = []
        position = 0

        while position < len(response):
            while position < len(response) and (response[position].isspace() or response[position] == ";"):
                position += 1
            if position >= len(response):
                break

            try:
                obj, end = decoder.raw_decode(response, position)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Llama returned invalid tool-call JSON: {response}"
                ) from exc

            objects.append(obj)
            position = end

        if not objects:
            raise ValueError(f"Llama returned invalid tool-call JSON: {response}")

        actions = []
        for obj in objects:
            actions.extend(LlamaInterface._normalise_parsed_actions(obj))
        return actions

    @staticmethod
    def _normalise_parsed_actions(parsed):
        """Convert supported parsed JSON shapes into an action list."""
        if isinstance(parsed, dict) and "actions" in parsed:
            actions = parsed["actions"]
            if not isinstance(actions, list) or not actions:
                raise ValueError("'actions' must be a non-empty list.")
        elif isinstance(parsed, dict):
            actions = [parsed]
        else:
            raise ValueError("Tool call must be a JSON object or actions array.")

        result = []
        for item in actions:
            if not isinstance(item, dict) or "name" not in item or "parameters" not in item:
                raise ValueError("Each action must contain 'name' and 'parameters'.")
            if not isinstance(item["parameters"], dict):
                raise ValueError("Action parameters must be a JSON object.")
            result.append((item["name"], item["parameters"]))
        return result

    def _tool_schemas(self):
        """Return the restricted tool registry as OpenAI-compatible schemas."""
        return [
            {"type": "function", "function": {"name": "get_portfolio", "description": "Get current cash, Bitcoin holdings, current Bitcoin price, and total portfolio value.", "parameters": {"type": "object", "properties": {}, "additionalProperties": False}}},
            {"type": "function", "function": {"name": "get_portfolio_performance", "description": "Get historical portfolio performance over a supported lookback period.", "parameters": {"type": "object", "properties": {"period": {"type": "string", "enum": ["7d", "30d", "90d", "1y", "all"]}}, "required": ["period"], "additionalProperties": False}}},
            {"type": "function", "function": {"name": "get_trading_status", "description": "Get whether automated trading is enabled.", "parameters": {"type": "object", "properties": {}, "additionalProperties": False}}},
            {"type": "function", "function": {"name": "get_dca_status", "description": "Get the current DCA amount, market regime, effective DCA frequency, and DCA state.", "parameters": {"type": "object", "properties": {}, "additionalProperties": False}}},
            {"type": "function", "function": {"name": "adjust_capital", "description": "Add or withdraw simulated capital. Positive adds capital; negative withdraws it.", "parameters": {"type": "object", "properties": {"amount": {"type": "number"}}, "required": ["amount"], "additionalProperties": False}}},
            {"type": "function", "function": {"name": "set_dca_amount", "description": "Change the base DCA amount for future scheduled purchases.", "parameters": {"type": "object", "properties": {"amount": {"type": "number"}}, "required": ["amount"], "additionalProperties": False}}},
            {"type": "function", "function": {"name": "get_risk_level", "description": "Get the currently active risk level and strategy description.", "parameters": {"type": "object", "properties": {}, "additionalProperties": False}}},
            {"type": "function", "function": {"name": "set_risk_level", "description": "Change the active risk level to 1, 2, or 3.", "parameters": {"type": "object", "properties": {"level": {"type": "integer", "enum": [1, 2, 3]}}, "required": ["level"], "additionalProperties": False}}},
            {"type": "function", "function": {"name": "start_trading", "description": "Enable automated trading.", "parameters": {"type": "object", "properties": {}, "additionalProperties": False}}},
            {"type": "function", "function": {"name": "stop_trading", "description": "Disable automated trading.", "parameters": {"type": "object", "properties": {}, "additionalProperties": False}}},
            {"type": "function", "function": {"name": "buy_btc", "description": "Make a simulated Bitcoin purchase using existing cash for a specified cash amount.", "parameters": {"type": "object", "properties": {"amount": {"type": "number"}}, "required": ["amount"], "additionalProperties": False}}},
            {"type": "function", "function": {"name": "sell_btc", "description": "Sell a specified Bitcoin amount or percentage of the current Bitcoin holding.", "parameters": {"type": "object", "properties": {"btc_amount": {"type": "number"}, "percentage": {"type": "number"}}, "additionalProperties": False}}},
        ]

    def _generate_tool_request(self, user_message):
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ]

        tool_descriptions = []
        for schema in self._tool_schemas():
            fn = schema["function"]
            tool_descriptions.append(
                json.dumps({
                    "name": fn["name"],
                    "description": fn["description"],
                    "parameters": fn["parameters"],
                }, separators=(",", ":"))
            )

        tool_prompt = (
            "\n\nAVAILABLE TOOLS:\n"
            + "\n".join(tool_descriptions)
            + "\n\nTOOL-CALL OUTPUT RULES:\n"
              "Return ONLY valid JSON. Do not use markdown or explanatory text. "
              "For one requested operation return {\"name\":\"tool_name\",\"parameters\":{...}}. "
              "For multiple independent operations return {\"actions\":[{\"name\":\"tool_name\",\"parameters\":{...}}, ...]}. "
              "For a read-only question, select exactly one appropriate read-only tool. "
        )
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT + tool_prompt},
            {"role": "user", "content": user_message},
        ]

        response = self.model.create_chat_completion(
            messages=messages,
            temperature=0.0,
            max_tokens=180,
            stop=["<|eot_id|>", "<|end_of_text|>"],
        )

        content = response["choices"][0]["message"].get("content", "").strip()
        if not content:
            raise ValueError("Llama returned an empty tool-call response.")

        # Some llama.cpp builds may still return a fenced JSON block. Remove
        # only the formatting wrapper; never alter the JSON payload itself.
        if content.startswith("```json") and content.endswith("```"):
            content = content[7:-3].strip()
        elif content.startswith("```") and content.endswith("```"):
            content = content[3:-3].strip()

        # Validate the JSON here so generation failures are logged as such
        # rather than surfacing later as an ambiguous tool execution error.
        json.loads(content)
        return content

    def _augment_explicit_multi_actions(self, user_message, parsed_actions):
        """Recover explicit configuration actions omitted by Llama.

        The application supplements Llama only when the user explicitly states
        a parameter. DCA frequency is intentionally excluded because it is
        selected internally by the strategy's regime detector.
        """
        import re

        text = user_message.lower().replace(",", "")
        recovered = list(parsed_actions)

        def has_action(name, key, value):
            return any(
                action == name and params.get(key) == value
                for action, params in recovered
            )

        dca_amount_patterns = [
            r"(?:dca|dollar[- ]cost averaging).*?(?:£|\$|€)\s*(\d+(?:\.\d{1,2})?)",
            r"(?:set|change)\s+(?:my\s+)?dca\s+(?:amount\s+)?to\s+(?:£|\$|€)\s*(\d+(?:\.\d{1,2})?)",
        ]
        for pattern in dca_amount_patterns:
            match = re.search(pattern, text)
            if match:
                requested_amount = float(match.group(1))
                if not has_action("set_dca_amount", "amount", requested_amount):
                    recovered.append(("set_dca_amount", {"amount": requested_amount}))
                break

        risk_patterns = [
            r"\b(?:set|change)\s+(?:my\s+)?risk(?:\s+level)?\s+(?:to\s+)?(low|medium|high|1|2|3)\b",
            r"\brisk\s+(?:level\s+)?(low|medium|high|1|2|3)\b",
        ]
        for pattern in risk_patterns:
            match = re.search(pattern, text)
            if match:
                risk_value = match.group(1).lower()
                risk_level = {"low": 1, "medium": 2, "high": 3}.get(risk_value)
                if risk_level is None:
                    risk_level = int(risk_value)
                if not any(a == "set_risk_level" for a, _ in recovered):
                    recovered.append(("set_risk_level", {"level": risk_level}))
                break

        buy_patterns = [
            r"\b(?:trade|buy|purchase|invest)\s+(?:£|\$|€)\s*(\d+(?:\.\d{1,2})?)\b",
            r"\b(?:buy|purchase|invest)\s+(?:bitcoin|btc)\s+(?:for|with)\s+(?:£|\$|€)\s*(\d+(?:\.\d{1,2})?)\b",
        ]
        for pattern in buy_patterns:
            for match in re.finditer(pattern, text):
                amount = float(match.group(1))
                if not has_action("buy_btc", "amount", amount):
                    recovered.append(("buy_btc", {"amount": amount}))

        capital_patterns = [
            r"\b(?:ad+d|deposit|top up|put)\s+(?:another\s+)?(?:£|\$|€)\s*(\d+(?:\.\d{1,2})?)\s*(?:to|into)\s+(?:the\s+)?(?:account|trading account|capital)",
            r"\b(?:ad+d|deposit|top up)\s+(?:another\s+)?(?:£|\$|€)\s*(\d+(?:\.\d{1,2})?)\b",
        ]
        for pattern in capital_patterns:
            for match in re.finditer(pattern, text):
                amount = float(match.group(1))
                if not has_action("adjust_capital", "amount", amount):
                    recovered.append(("adjust_capital", {"amount": amount}))

        sell_pct = re.search(r"\b(?:sell|liquidate)\s+(?:all|100%)\s+(?:of\s+)?(?:my\s+)?(?:bitcoin|btc)\b", text)
        if sell_pct and not any(a == "sell_btc" for a, _ in recovered):
            recovered.append(("sell_btc", {"percentage": 100.0}))
        else:
            pct_match = re.search(r"\b(?:sell|liquidate)\s+(\d+(?:\.\d+)?)\s*%\s+(?:of\s+)?(?:my\s+)?(?:bitcoin|btc)\b", text)
            btc_match = re.search(r"\b(?:sell|liquidate)\s+(\d+(?:\.\d+)?)\s*(?:btc|bitcoin)\b", text)
            if pct_match and not any(a == "sell_btc" for a, _ in recovered):
                recovered.append(("sell_btc", {"percentage": float(pct_match.group(1))}))
            elif btc_match and not any(a == "sell_btc" for a, _ in recovered):
                recovered.append(("sell_btc", {"btc_amount": float(btc_match.group(1))}))

        deduplicated = []
        seen = set()
        for action, params in recovered:
            canonical_values = []
            for key, value in params.items():
                if action == "set_dca_amount" and key == "amount":
                    value = round(float(value), 2)
                elif action == "set_risk_level" and key == "level":
                    value = int(float(value))
                elif action in {"buy_btc", "adjust_capital"} and key == "amount":
                    value = round(float(value), 2)
                elif action == "sell_btc" and key in {"btc_amount", "percentage"}:
                    value = round(float(value), 8)
                elif isinstance(value, (int, float)) and not isinstance(value, bool):
                    value = float(value)
                canonical_values.append((key, value))
            canonical = (action, tuple(sorted(canonical_values)))
            if canonical not in seen:
                seen.add(canonical)
                deduplicated.append((action, params))

        # Reconcile the final queue with the order in the user's message.
        # The recovery pass above is grouped by action type, so without this
        # step a request such as "add £9000, set risk high, trade £500" could
        # accidentally become BUY -> DEPOSIT -> RISK.  That is particularly
        # problematic when the purchase depends on the newly added cash.
        #
        # We only use the user's text to order actions that can be located
        # explicitly.  Any genuinely inferred LLM action that has no textual
        # match remains after the explicit actions.
        order_patterns = [
            ("adjust_capital", [
                r"\b(?:ad+d|deposit|top up|put)\s+(?:another\s+)?(?:£|\$|€)\s*\d+(?:\.\d{1,2})?\s*(?:to|into)\s+(?:the\s+)?(?:account|trading account|capital)",
                r"\b(?:ad+d|deposit|top up)\s+(?:another\s+)?(?:£|\$|€)\s*\d+(?:\.\d{1,2})?\b",
            ]),
            ("set_risk_level", [
                r"\b(?:set|change)\s+(?:my\s+)?risk(?:\s+level)?\s+(?:to\s+)?(?:low|medium|high|1|2|3)\b",
                r"\brisk\s+(?:level\s+)?(?:low|medium|high|1|2|3)\b",
            ]),
            ("set_dca_amount", [
                r"(?:dca|dollar[- ]cost averaging).*?(?:£|\$|€)\s*\d+(?:\.\d{1,2})?",
                r"(?:set|change)\s+(?:my\s+)?dca\s+(?:amount\s+)?to\s+(?:£|\$|€)\s*\d+(?:\.\d{1,2})?",
            ]),
            ("buy_btc", [
                r"\b(?:trade|buy|purchase|invest)\s+(?:£|\$|€)\s*\d+(?:\.\d{1,2})?\b",
                r"\b(?:buy|purchase|invest)\s+(?:bitcoin|btc)\s+(?:for|with)\s+(?:£|\$|€)\s*\d+(?:\.\d{1,2})?\b",
            ]),
            ("sell_btc", [
                r"\b(?:sell|liquidate)\s+(?:all|100%)\s+(?:of\s+)?(?:my\s+)?(?:bitcoin|btc)\b",
                r"\b(?:sell|liquidate)\s+\d+(?:\.\d+)?\s*%\s+(?:of\s+)?(?:my\s+)?(?:bitcoin|btc)\b",
                r"\b(?:sell|liquidate)\s+\d+(?:\.\d+)?\s*(?:btc|bitcoin)\b",
            ]),
            ("start_trading", [r"\b(?:start|enable)\s+(?:automated\s+)?trading\b"]),
            ("stop_trading", [r"\b(?:stop|disable)\s+(?:automated\s+)?trading\b"]),
        ]

        explicit_positions = []
        lower_text = user_message.lower()
        for action_name, patterns in order_patterns:
            for pattern in patterns:
                match = re.search(pattern, lower_text)
                if match is not None:
                    explicit_positions.append((match.start(), action_name))
                    break

        explicit_positions.sort(key=lambda item: item[0])
        order = {action: index for index, (_, action) in enumerate(explicit_positions)}

        # Preserve duplicate-action order where the same action appears more
        # than once, while ensuring explicitly stated actions precede inferred
        # actions.
        counters = {}
        indexed = []
        for original_index, item in enumerate(deduplicated):
            action = item[0]
            occurrence = counters.get(action, 0)
            counters[action] = occurrence + 1
            if action in order:
                indexed.append((0, order[action], occurrence, original_index, item))
            else:
                indexed.append((1, len(order), occurrence, original_index, item))

        indexed.sort(key=lambda row: row[:4])
        return [row[4] for row in indexed]

    def chat(self, user_message):
        """Process one user message.

        If the message is yes/no and an action is pending, confirmation is
        handled before asking the LLM to generate another tool request.
        """

        if not isinstance(user_message, str):
            raise ValueError("User message must be text.")
        if len(user_message) > MAX_USER_MESSAGE_CHARS:
            raise ValueError(
                f"User message exceeds the {MAX_USER_MESSAGE_CHARS} character limit."
            )

        if self.pending_action is not None:
            confirmation = self._normalise_yes_no(user_message)

            if confirmation == "yes":
                return self._confirm_pending_action()

            if confirmation == "no":
                return self._cancel_pending_action()

            return "Please confirm the pending action by typing `yes` or `no`."

        try:
            response = self._generate_tool_request(user_message)
        except Exception as exc:
            audit_event(
                "llm_request_failed",
                session_id=self.session_id,
                status="error",
                detail=f"phase=generation type={type(exc).__name__}",
            )
            raise

        try:
            parsed_actions = self._parse_tool_call(response)
            parsed_actions = self._augment_explicit_multi_actions(user_message, parsed_actions)
        except Exception as exc:
            audit_event(
                "llm_response_parse_failed",
                session_id=self.session_id,
                status="error",
                detail=f"phase=parsing type={type(exc).__name__}",
            )
            raise

        action_tools = {
            "buy_btc",
            "adjust_capital",
            "set_dca_amount",
            "set_risk_level",
            "start_trading",
            "stop_trading",
            "sell_btc",
        }

        if all(tool_name in action_tools for tool_name, _ in parsed_actions):
            return self._create_pending_actions(parsed_actions)

        if len(parsed_actions) != 1:
            raise ValueError("Read-only requests cannot be combined with multiple action requests in one tool response.")

        tool_name, parameters = parsed_actions[0]
        try:
            result = self.execute_read_only_tool(tool_name, parameters)
        except Exception as exc:
            audit_event(
                "read_only_tool_failed",
                session_id=self.session_id,
                status="error",
                detail=f"tool={tool_name} type={type(exc).__name__}",
            )
            raise

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
            {
                "role": "assistant",
                "content": json.dumps({
                    "name": tool_name,
                    "parameters": parameters,
                }),
            },
            {
                "role": "user",
                "content": (
                    "The read-only tool returned this JSON result. Answer the original "
                    "question directly using only this result. Do not mention tools, "
                    "tool calls, JSON, or internal instructions.\n\n"
                    + json.dumps(result)
                ),
            },
        ]

        try:
            response = self.model.create_chat_completion(
                messages=messages,
                temperature=0.0,
                max_tokens=100,
            )
            return response["choices"][0]["message"].get("content", "").strip()
        except Exception as exc:
            audit_event(
                "llm_response_generation_failed",
                session_id=self.session_id,
                status="error",
                detail=f"phase=response_generation type={type(exc).__name__}",
            )
            raise
