#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Production Level 1 trading strategy.

Level 1 = regime-adaptive DCA + global capital-deployment safeguard.

Frozen strategy parameters
---------------------------
Regime lookback: 30 days
    Bullish: 30-day return > +10%  -> DCA every 10 days
    Neutral: -10% <= return <= +10% -> DCA every 14 days
    Bearish: 30-day return < -10% -> DCA every 21 days

Global safeguard
    Rolling 30-day portfolio peak
    Stop NEW trading at -25% drawdown
    Resume at -15% drawdown
    BTC is not liquidated

User-configurable inputs
------------------------
    budget: initial/contributed capital
    dca_amount: base amount deployed at each scheduled DCA

DCA frequency is deliberately NOT user-configurable.  It is selected by the
frozen regime detector at decision time.
"""

import pandas as pd


class TradingStrategyLevel1:
    """Production Level 1 strategy."""

    risk_level = 1

    REGIME_LOOKBACK_DAYS = 30
    BULLISH_THRESHOLD_PCT = 10.0
    BEARISH_THRESHOLD_PCT = -10.0

    BULLISH_DCA_FREQUENCY_DAYS = 10
    NEUTRAL_DCA_FREQUENCY_DAYS = 14
    BEARISH_DCA_FREQUENCY_DAYS = 21

    def __init__(
        self,
        budget,
        dca_amount,
        global_stop_enabled=True,
        global_stop_drawdown_pct=25.0,
        global_stop_resume_drawdown_pct=15.0,
        global_stop_lookback_days=30,
    ):
        self.budget = float(budget)
        self.dca_amount = float(dca_amount)

        self.global_stop_enabled = bool(global_stop_enabled)
        self.global_stop_drawdown_pct = float(global_stop_drawdown_pct)
        self.global_stop_resume_drawdown_pct = float(global_stop_resume_drawdown_pct)
        self.global_stop_lookback_days = int(global_stop_lookback_days)

        self.cash = self.budget
        self.btc_held = 0.0
        self.btc_cost_basis = 0.0

        self.last_dca_day = None
        self.next_dca_multiplier = 1.0

        self.portfolio_history = []
        self.global_stop_active = False
        self.performance_baseline_value = None
        self.capital_adjustments = []

        # The application controls this flag.  The LLM does not choose
        # strategy parameters through this object.
        self.trading_enabled = False

        # Informational regime state.
        self.current_regime = "UNKNOWN"
        self.current_regime_return_pct = None
        self.current_dca_frequency = self.NEUTRAL_DCA_FREQUENCY_DAYS

    # ------------------------------------------------------------------
    # Capital
    # ------------------------------------------------------------------

    def adjust_capital(self, amount):
        """Add or withdraw simulated user capital."""
        try:
            amount = float(amount)
        except (TypeError, ValueError) as exc:
            raise ValueError("Capital adjustment must be a numeric value.") from exc

        if not pd.notna(amount) or amount in (float("inf"), float("-inf")):
            raise ValueError("Capital adjustment must be a finite number.")

        amount = round(amount, 2)
        if amount == 0:
            raise ValueError("Capital adjustment cannot be 0.")

        if amount < 0 and abs(amount) > self.cash:
            raise ValueError(
                "Withdrawal cannot exceed available cash. "
                "BTC will not be sold automatically to fund a withdrawal."
            )

        previous_budget = self.budget
        previous_cash = self.cash
        self.budget = round(self.budget + amount, 2)
        self.cash = round(self.cash + amount, 2)

        if self.budget < 0:
            self.budget = 0.0

        if self.performance_baseline_value is None and amount > 0:
            self.performance_baseline_value = amount

        self.capital_adjustments.append({
            "amount": amount,
            "previous_budget": previous_budget,
            "new_budget": self.budget,
            "previous_cash": previous_cash,
            "new_cash": self.cash,
        })
        return amount

    # ------------------------------------------------------------------
    # Portfolio / safeguard
    # ------------------------------------------------------------------

    def portfolio_value(self, price):
        return self.cash + self.btc_held * price

    def record_portfolio_value(self, data):
        day = data["Timestep"].iloc[-1]
        price = float(data["Close"].iloc[-1])
        value = self.portfolio_value(price)

        if self.portfolio_history and self.portfolio_history[-1]["Timestep"] == day:
            return value

        self.portfolio_history.append({
            "Timestep": day,
            "Price": price,
            "Cash": self.cash,
            "BTC Held": self.btc_held,
            "BTC Cost Basis": self.btc_cost_basis,
            "Portfolio Value": value,
        })
        return value

    def update_global_stop(self, day, portfolio_value):
        """Update the rolling 30-day capital-deployment safeguard."""
        if not self.global_stop_enabled:
            self.global_stop_active = False
            return {
                "active": False,
                "portfolio_value": portfolio_value,
                "rolling_peak": portfolio_value,
                "drawdown_pct": 0.0,
            }

        cutoff = day - pd.Timedelta(days=self.global_stop_lookback_days)
        recent_values = [
            entry["Portfolio Value"]
            for entry in self.portfolio_history
            if entry["Timestep"] >= cutoff
        ]
        recent_values.append(portfolio_value)
        rolling_peak = max(recent_values)

        drawdown_pct = 0.0 if rolling_peak <= 0 else (
            (portfolio_value - rolling_peak) / rolling_peak
        ) * 100.0

        if not self.global_stop_active:
            if drawdown_pct <= -self.global_stop_drawdown_pct:
                self.global_stop_active = True
        elif drawdown_pct >= -self.global_stop_resume_drawdown_pct:
            self.global_stop_active = False

        return {
            "active": bool(self.global_stop_active),
            "portfolio_value": portfolio_value,
            "rolling_peak": rolling_peak,
            "drawdown_pct": drawdown_pct,
        }

    # ------------------------------------------------------------------
    # Regime-adaptive DCA
    # ------------------------------------------------------------------

    def detect_regime(self, data):
        """Detect the frozen trailing-price regime using causal data only."""
        if len(data) <= self.REGIME_LOOKBACK_DAYS:
            return {
                "state": "UNKNOWN",
                "price_return_pct": None,
                "dca_frequency_days": self.NEUTRAL_DCA_FREQUENCY_DAYS,
            }

        current_price = float(data["Close"].iloc[-1])
        reference_price = float(
            data["Close"].iloc[-(self.REGIME_LOOKBACK_DAYS + 1)]
        )

        if reference_price <= 0:
            return {
                "state": "UNKNOWN",
                "price_return_pct": None,
                "dca_frequency_days": self.NEUTRAL_DCA_FREQUENCY_DAYS,
            }

        return_pct = ((current_price / reference_price) - 1.0) * 100.0

        if return_pct > self.BULLISH_THRESHOLD_PCT:
            state = "BULLISH"
            frequency = self.BULLISH_DCA_FREQUENCY_DAYS
        elif return_pct < self.BEARISH_THRESHOLD_PCT:
            state = "BEARISH"
            frequency = self.BEARISH_DCA_FREQUENCY_DAYS
        else:
            state = "NEUTRAL"
            frequency = self.NEUTRAL_DCA_FREQUENCY_DAYS

        return {
            "state": state,
            "price_return_pct": return_pct,
            "dca_frequency_days": frequency,
        }

    def update_regime(self, data):
        result = self.detect_regime(data)
        self.current_regime = result["state"]
        self.current_regime_return_pct = result["price_return_pct"]
        self.current_dca_frequency = int(result["dca_frequency_days"])
        return result

    def calculate_dca(self, day, data=None):
        """Return the base DCA amount if the internally selected schedule fires."""
        if data is not None:
            self.update_regime(data)

        if self.last_dca_day is None:
            return self.dca_amount

        if (day - self.last_dca_day).days >= self.current_dca_frequency:
            return self.dca_amount
        return 0.0

    # ------------------------------------------------------------------
    # Buying
    # ------------------------------------------------------------------

    def buy(self, amount, price):
        amount = min(float(amount), self.cash)
        if amount <= 0:
            return 0.0

        btc_bought = amount / price
        self.cash -= amount
        self.btc_held += btc_bought
        self.btc_cost_basis += amount
        return btc_bought

    # ------------------------------------------------------------------
    # Decision / execution
    # ------------------------------------------------------------------

    def make_decision(self, data, previous_price=None):
        del previous_price  # Level 1 does not use the prior close directly.

        day = data["Timestep"].iloc[-1]
        price = float(data["Close"].iloc[-1])

        self.update_regime(data)
        portfolio_value = self.portfolio_value(price)
        stop_status = self.update_global_stop(day, portfolio_value)
        self.record_portfolio_value(data)

        if not self.trading_enabled:
            return {
                "action": "HOLD",
                "amount": 0.0,
                "price": price,
                "Timestep": day,
                "reason": "Trading disabled",
            }

        if stop_status["active"]:
            return {
                "action": "HOLD",
                "amount": 0.0,
                "price": price,
                "Timestep": day,
                "reason": "Global drawdown safeguard active",
            }

        if self.last_dca_day is not None and day.date() == self.last_dca_day.date():
            return {
                "action": "HOLD",
                "amount": 0.0,
                "price": price,
                "Timestep": day,
                "reason": "DCA already executed today",
            }

        dca_amount = self.calculate_dca(day)
        if dca_amount <= 0:
            return {
                "action": "HOLD",
                "amount": 0.0,
                "price": price,
                "Timestep": day,
                "reason": "DCA not scheduled",
                "regime": self.current_regime,
                "dca_frequency_days": self.current_dca_frequency,
            }

        if self.cash <= 0:
            return {
                "action": "HOLD",
                "amount": 0.0,
                "price": price,
                "Timestep": day,
                "reason": "DCA scheduled but no cash available",
            }

        return {
            "action": "BUY",
            "amount": min(dca_amount, self.cash),
            "price": price,
            "Timestep": day,
            "reason": "Regime-adaptive DCA",
            "regime": self.current_regime,
            "regime_return_pct": self.current_regime_return_pct,
            "dca_frequency_days": self.current_dca_frequency,
        }

    def execute_decision(self, decision):
        if decision.get("action") != "BUY":
            return 0.0

        btc_bought = self.buy(decision["amount"], decision["price"])
        if btc_bought > 0:
            self.last_dca_day = decision["Timestep"]
        return btc_bought
