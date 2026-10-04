#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 19 19:19:12 2026

@author: leszekwierzchleyski
"""

from datetime import timedelta
from Apziva_Binance_API import BinanceMarketData


def get_portfolio_performance(trading_strategy, period="30d"):

    valid_periods = {
        "7d": timedelta(days=7),
        "30d": timedelta(days=30),
        "90d": timedelta(days=90),
        "1y": timedelta(days=365),
        "all": None
    }

    if period not in valid_periods:
        raise ValueError(
            f"Invalid period '{period}'. "
            f"Choose from: {list(valid_periods.keys())}"
        )

    history = trading_strategy.portfolio_history

    if not history:
        return {
            "period": period,
            "status": "no_data",
            "message": "No portfolio history is available yet.",
            "observations": 0,
            "return_available": False,
            "data": []
        }

    history = sorted(
        history,
        key=lambda x: x["Timestep"]
    )

    latest_date = history[-1]["Timestep"]
    lookback = valid_periods[period]

    if lookback is None:
        filtered_history = history
    else:
        start_date = latest_date - lookback

        filtered_history = [
            entry
            for entry in history
            if entry["Timestep"] >= start_date
        ]

    if not filtered_history:
        return {
            "period": period,
            "status": "insufficient_data",
            "message": "No portfolio history is available for this period.",
            "observations": 0,
            "return_available": False,
            "data": []
        }

    # Prepare time-series data FIRST
    data = [
        {
            "Timestep": entry["Timestep"].isoformat(),
            "Price": float(entry["Price"]),
            "Cash": float(entry["Cash"]),
            "BTC Held": float(entry["BTC Held"]),
            "Portfolio Value": float(entry["Portfolio Value"])
        }
        for entry in filtered_history
    ]

    # Not enough observations for a meaningful return
    if len(filtered_history) < 2:
        return {
            "period": period,
            "status": "insufficient_data",
            "message": "Not enough portfolio history to calculate performance.",
            "observations": len(filtered_history),
            "return_available": False,
            "data": data
        }

    # Performance calculations
    starting_value = float(
        filtered_history[0]["Portfolio Value"]
    )

    ending_value = float(
        filtered_history[-1]["Portfolio Value"]
    )

    profit_loss = ending_value - starting_value

    if starting_value != 0:
        percentage_return = (
            profit_loss / starting_value
        ) * 100
    else:
        percentage_return = 0.0

    return {
        "period": period,
        "status": "success",
        "starting_value": starting_value,
        "ending_value": ending_value,
        "profit_loss": profit_loss,
        "percentage_return": percentage_return,
        "observations": len(filtered_history),
        "return_available": True,
        "data": data
    }
    

def get_portfolio(trading_strategy, market_data):
    """
    Return the current portfolio state.
    """

    data = market_data.get_candles(interval="1d", limit=1)
    current_price = data["Close"].iloc[-1]

    portfolio_value = trading_strategy.portfolio_value(current_price)

    return {
        "cash": float(trading_strategy.cash),
        "btc_held": float(trading_strategy.btc_held),
        "btc_price": float(current_price),
        "portfolio_value": float(portfolio_value)
    }


def buy_btc(trading_strategy: object, market_data: object, amount: float):
    """Make a manual simulated BTC purchase for a specified cash amount.

    Args:
        trading_strategy: The active trading strategy whose cash and BTC state
            will be updated.
        market_data: The read-only market-data provider used for the purchase
            price.
        amount: The amount of available cash to spend on Bitcoin.

    This is distinct from adding capital: it converts available cash into BTC
    at the current read-only market price. It does not change contributed
    capital, DCA settings, risk level, or the automated strategy schedule.
    """
    try:
        amount = float(amount)
    except (TypeError, ValueError):
        raise ValueError("Purchase amount must be numeric.")

    if not amount == amount or amount in (float("inf"), float("-inf")):
        raise ValueError("Purchase amount must be finite.")
    amount = round(amount, 2)
    if amount <= 0:
        raise ValueError("Purchase amount must be greater than 0.")

    if trading_strategy is None:
        raise ValueError("Trading strategy is not configured.")

    cash = float(getattr(trading_strategy, "cash", 0.0))
    if cash <= 0:
        raise ValueError("There is no cash available for a purchase.")
    if amount > cash:
        raise ValueError(f"Only £{cash:,.2f} is available in cash.")

    data = market_data.get_candles(interval="1d", limit=1)
    price = float(data["Close"].iloc[-1])
    btc_bought = trading_strategy.buy(amount, price)

    try:
        trading_strategy.record_portfolio_value(data)
    except Exception:
        pass

    return {
        "status": "success",
        "action": "buy_btc",
        "amount": amount,
        "price": price,
        "btc_bought": btc_bought,
        "new_btc_held": float(trading_strategy.btc_held),
        "new_cash": float(trading_strategy.cash),
        "message": "Bitcoin purchased successfully.",
    }


def sell_btc(trading_strategy: object, market_data: object, btc_amount: float = None, percentage: float = None):
    """Sell a user-specified amount or percentage of the current BTC holding.

    Args:
        trading_strategy: The active trading strategy and portfolio state.
        market_data: The read-only market-data provider used to obtain the
            current BTC price.
        btc_amount: The amount of Bitcoin to sell, in BTC. Provide this for
            an absolute BTC quantity.
        percentage: The percentage of the current BTC holding to sell, from
            0 to 100. Provide this when selling by percentage.

    This is a manual, user-initiated sale. It does not change contributed
    capital, risk level, or the active strategy. The sale uses the current
    read-only market price and allocates cost basis proportionally.
    """
    if (btc_amount is None) == (percentage is None):
        raise ValueError("Provide exactly one of btc_amount or percentage.")

    if trading_strategy is None:
        raise ValueError("Trading strategy is not configured.")

    held = float(getattr(trading_strategy, "btc_held", 0.0))
    if held <= 0:
        raise ValueError("There is no Bitcoin position to sell.")

    if percentage is not None:
        try:
            percentage = float(percentage)
        except (TypeError, ValueError):
            raise ValueError("Sale percentage must be numeric.")
        if not percentage == percentage or percentage in (float("inf"), float("-inf")):
            raise ValueError("Sale percentage must be finite.")
        if percentage <= 0 or percentage > 100:
            raise ValueError("Sale percentage must be greater than 0 and no more than 100.")
        btc_amount = held * (percentage / 100.0)
    else:
        try:
            btc_amount = float(btc_amount)
        except (TypeError, ValueError):
            raise ValueError("BTC amount must be numeric.")
        if not btc_amount == btc_amount or btc_amount in (float("inf"), float("-inf")):
            raise ValueError("BTC amount must be finite.")
        if btc_amount <= 0:
            raise ValueError("BTC amount must be greater than 0.")
        if btc_amount > held:
            raise ValueError(f"You only hold {held:.8f} BTC.")

    data = market_data.get_candles(interval="1d", limit=1)
    price = float(data["Close"].iloc[-1])

    btc_amount = min(float(btc_amount), held)
    cost_basis_sold = float(trading_strategy.btc_cost_basis) * (btc_amount / held)
    estimated_proceeds = btc_amount * price
    estimated_realised_pnl = estimated_proceeds - cost_basis_sold

    realised_profit = trading_strategy.execute_sale(btc_amount, price)

    # Keep a current portfolio observation without rewriting prior history.
    try:
        trading_strategy.record_portfolio_value(data)
    except Exception:
        pass

    return {
        "status": "success",
        "action": "sell_btc",
        "btc_sold": btc_amount,
        "sale_percentage": (btc_amount / held) * 100.0,
        "price": price,
        "proceeds": estimated_proceeds,
        "realised_profit": realised_profit,
        "new_btc_held": float(trading_strategy.btc_held),
        "new_cash": float(trading_strategy.cash),
        "message": "Bitcoin sold successfully.",
    }


def get_trading_status(trading_strategy):
    """
    Return the current trading status.
    """

    return {
        "trading_enabled": getattr(
            trading_strategy,
            "trading_enabled",
            False
        )
    }




def start_trading(trading_strategy):
    """Enable automated trading after explicit user confirmation."""
    if trading_strategy is None:
        raise ValueError("Trading strategy is not configured.")

    if getattr(trading_strategy, "trading_enabled", False):
        raise ValueError("Trading is already enabled; no change is required.")

    trading_strategy.trading_enabled = True

    return {
        "status": "success",
        "action": "start_trading",
        "trading_enabled": True,
        "message": "Trading started successfully.",
    }


def stop_trading(trading_strategy):
    """Disable automated trading after explicit user confirmation."""
    if trading_strategy is None:
        raise ValueError("Trading strategy is not configured.")

    if not getattr(trading_strategy, "trading_enabled", False):
        raise ValueError("Trading is already stopped; no change is required.")

    trading_strategy.trading_enabled = False

    return {
        "status": "success",
        "action": "stop_trading",
        "trading_enabled": False,
        "message": "Trading stopped successfully.",
    }


def get_dca_status(trading_strategy):
    """
    Return the current DCA configuration and state.
    """

    return {
        "dca_configured": True,
        "dca_amount": float(trading_strategy.dca_amount),
        "regime": getattr(trading_strategy, "current_regime", None),
        "regime_return_pct": getattr(trading_strategy, "current_regime_return_pct", None),
        "effective_dca_frequency_days": getattr(trading_strategy, "current_dca_frequency", None),
        "frequency_user_configurable": False,
        "last_dca_day": (
            trading_strategy.last_dca_day.isoformat()
            if trading_strategy.last_dca_day is not None
            else None
        ),
        "status": (
            "configured_but_not_yet_executed"
            if trading_strategy.last_dca_day is None
            else "active"
        )
    }


def get_risk_level(risk_manager):
    """Return the currently active risk level and strategy description."""
    if risk_manager is None:
        raise ValueError("Risk manager is not configured.")
    return risk_manager.get_status()


def set_risk_level(risk_manager, level):
    """Switch the strategy used for future trading after explicit confirmation."""
    try:
        level = int(level)
    except (TypeError, ValueError):
        raise ValueError("Risk level must be an integer: 1, 2, or 3.")

    if level not in {1, 2, 3}:
        raise ValueError("Risk level must be 1, 2, or 3.")

    if risk_manager is None:
        raise ValueError("Risk manager is not configured.")

    previous_level = int(risk_manager.current_risk_level)
    result = risk_manager.switch_risk_level(level)

    return {
        "status": "success",
        "action": "set_risk_level",
        "previous_risk_level": previous_level,
        "new_risk_level": int(result["risk_level"]),
        "name": result["name"],
        "description": result["description"],
        "message": "Risk level updated successfully.",
    }


def adjust_capital(trading_strategy, amount):
    """Add or withdraw simulated capital after explicit confirmation.

    Positive amounts add fresh user capital; negative amounts withdraw
    available cash. The strategy owns all portfolio-state validation.
    """
    try:
        amount = float(amount)
    except (TypeError, ValueError):
        raise ValueError("Capital adjustment must be a numeric value.")

    if not amount == amount or amount in (float("inf"), float("-inf")):
        raise ValueError("Capital adjustment must be a finite number.")
    amount = round(amount, 2)
    if amount == 0:
        raise ValueError("Capital adjustment cannot be 0.")

    previous_budget = round(float(trading_strategy.budget), 2)
    previous_cash = round(float(trading_strategy.cash), 2)

    if not hasattr(trading_strategy, "adjust_capital"):
        raise AttributeError("Trading strategy does not implement adjust_capital().")

    trading_strategy.adjust_capital(amount)

    return {
        "status": "success",
        "action": "adjust_capital",
        "adjustment": amount,
        "previous_budget": previous_budget,
        "new_budget": round(float(trading_strategy.budget), 2),
        "previous_cash": previous_cash,
        "new_cash": round(float(trading_strategy.cash), 2),
        "message": ("Capital added successfully." if amount > 0
                    else "Capital withdrawn successfully."),
    }

def set_dca_amount(trading_strategy, market_data, amount):
    """Set the base DCA amount for future scheduled DCA purchases."""
    try:
        amount = float(amount)
    except (TypeError, ValueError):
        raise ValueError("DCA amount must be numeric.")

    if not amount == amount or amount in (float("inf"), float("-inf")):
        raise ValueError("DCA amount must be finite.")

    amount = round(amount, 2)

    if amount <= 0:
        raise ValueError("DCA amount must be greater than 0.")

    data = market_data.get_candles(interval="1d", limit=1)
    current_price = float(data["Close"].iloc[-1])
    account_total = float(trading_strategy.portfolio_value(current_price))

    if amount > account_total:
        raise ValueError(
            f"DCA amount must not exceed the current account value of £{account_total:,.2f}."
        )

    previous_amount = round(float(trading_strategy.dca_amount), 2)
    trading_strategy.dca_amount = amount

    return {
        "status": "success",
        "action": "set_dca_amount",
        "previous_dca_amount": previous_amount,
        "new_dca_amount": amount,
        "message": "DCA amount updated successfully.",
    }

