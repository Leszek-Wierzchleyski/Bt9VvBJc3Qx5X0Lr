#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Risk-level controller for the Bitcoin trading application.

Risk levels select the strategy used for FUTURE trading decisions. Changing
risk level does not liquidate BTC, rewrite portfolio history, or undo any
completed DCA/profit-taking activity.

The application supplies strategy factories for Levels 1-3. The concrete factories
accept the current budget and base DCA amount and return a fully initialised
strategy object. DCA frequency is selected internally by the active strategy.
"""


RISK_LEVEL_DESCRIPTIONS = {
    1: "DCA + profit taking",
    2: "DCA + profit taking + ATR/volatility-based rules",
    3: "Level 2 + Trees/LSTM ML overlay",
}


class RiskManager:
    """Own the active risk level and safely switch strategy implementations."""

    def __init__(self, current_strategy, strategy_factories):
        if not strategy_factories:
            raise ValueError("At least one strategy factory is required.")

        self.strategy_factories = dict(strategy_factories)
        self.current_strategy = current_strategy
        self.current_risk_level = self._infer_risk_level(current_strategy)

        self._validate_level(self.current_risk_level)

    @staticmethod
    def _infer_risk_level(strategy):
        level = getattr(strategy, "risk_level", None)
        if level is not None:
            return int(level)

        class_name = strategy.__class__.__name__.lower()
        if "level3" in class_name or "level_3" in class_name:
            return 3
        if "level2" in class_name or "level_2" in class_name:
            return 2
        return 1

    def _validate_level(self, level):
        if level not in self.strategy_factories:
            available = sorted(self.strategy_factories)
            raise ValueError(
                f"Unsupported risk level {level}. Available levels: {available}."
            )

        if level not in RISK_LEVEL_DESCRIPTIONS:
            raise ValueError(f"No description is defined for risk level {level}.")

    def get_status(self):
        return {
            "risk_level": int(self.current_risk_level),
            "name": f"Level {self.current_risk_level}",
            "description": RISK_LEVEL_DESCRIPTIONS[self.current_risk_level],
        }

    def _snapshot_state(self):
        strategy = self.current_strategy

        # Only state that represents the existing portfolio/account is carried
        # across. Strategy-specific parameters are deliberately NOT copied.
        state = {}
        for attribute in (
            "budget",
            "cash",
            "btc_held",
            "btc_cost_basis",
            "dca_amount",
            "last_dca_day",
            "next_dca_multiplier",
            "portfolio_history",
            "profit_taken_total",
            "profit_taking_events",
            "profit_taking_armed",
            "global_stop_active",
            "trading_enabled",
            "performance_baseline_value",
        ):
            if hasattr(strategy, attribute):
                value = getattr(strategy, attribute)
                if attribute == "portfolio_history":
                    value = [dict(row) for row in value]
                state[attribute] = value

        return state

    @staticmethod
    def _apply_common_state(strategy, state):
        for attribute, value in state.items():
            if attribute == "portfolio_history":
                value = [dict(row) for row in value]
            setattr(strategy, attribute, value)

        # The new strategy is now the active level.
        strategy.risk_level = getattr(strategy, "risk_level", None)

    def switch_risk_level(self, new_level):
        """Switch the strategy governing future trading decisions."""
        try:
            new_level = int(new_level)
        except (TypeError, ValueError):
            raise ValueError("Risk level must be an integer: 1, 2, or 3.")

        self._validate_level(new_level)

        if new_level == self.current_risk_level:
            raise ValueError(
                f"Risk level is already Level {new_level}; no change is required."
            )

        state = self._snapshot_state()
        factory = self.strategy_factories[new_level]

        new_strategy = factory(
            budget=float(state.get("budget", 0.0)),
            dca_amount=float(state.get("dca_amount", 0.0)),
        )

        self._apply_common_state(new_strategy, state)
        new_strategy.risk_level = new_level

        self.current_strategy = new_strategy
        self.current_risk_level = new_level

        return self.get_status()
