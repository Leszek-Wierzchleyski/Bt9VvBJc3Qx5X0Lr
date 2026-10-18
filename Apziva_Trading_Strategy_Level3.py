#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Production Level 3 trading strategy.

Level 3 = Level 2 + opportunistic ML accumulation.

Level 2 remains the complete baseline strategy. Level 3 never replaces a
Level 2 decision. The causal Tree and LSTM models are scored on each day using
market history available at that decision. If BOTH probabilities meet the
frozen 45% consensus threshold, Level 3 adds one additional £250 purchase,
independently of whether Level 2's action that day is BUY, HOLD, or SELL.
The Level 2 action is executed unchanged; the ML purchase is an additional
transaction, subject to available cash.

Frozen Level 3 parameters
-------------------------
    Tree/LSTM consensus threshold: 45%
    Additional buy amount: £250

The additional buy is opportunistic.  There is no requirement for the ML
layer to generate a signal on every day or every market regime.
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd

from Apziva_Trading_Strategy_Level2_Final import TradingStrategyLevel2


CONSENSUS_THRESHOLD = 0.45
ADDITIONAL_BUY_AMOUNT = 250.0


_PROJECT_DIR = Path(__file__).resolve().parent
_LEVEL3_CANDIDATES = (
    _PROJECT_DIR / "Level3_Trading_Package",
    _PROJECT_DIR,
)
_LEVEL3_DIR = next(
    (path for path in _LEVEL3_CANDIDATES if (path / "Level3_Tree.py").exists()),
    None,
)
if _LEVEL3_DIR is None:
    raise FileNotFoundError(
        "Could not find the Level 3 model package. Expected Level3_Tree.py "
        "inside Level3_Trading_Package or beside this file."
    )

if str(_LEVEL3_DIR) not in sys.path:
    sys.path.insert(0, str(_LEVEL3_DIR))

from Level3_Tree import predict_probability as tree_probability
from Level3_LSTM import predict_probability as lstm_probability


class TradingStrategyLevel3(TradingStrategyLevel2):
    """Level 2 with a high-consensus ML additional-buy overlay."""

    risk_level = 3

    def __init__(
        self,
        budget,
        dca_amount,
        market_history=None,
        consensus_threshold=CONSENSUS_THRESHOLD,
        additional_buy_amount=ADDITIONAL_BUY_AMOUNT,
        **level2_kwargs,
    ):
        super().__init__(budget=budget, dca_amount=dca_amount, **level2_kwargs)

        self.consensus_threshold = float(consensus_threshold)
        self.additional_buy_amount = float(additional_buy_amount)

        self.market_history = None
        self.decision_index = 0

        self.last_tree_probability = None
        self.last_lstm_probability = None
        self.last_ml_buy_signal = False
        self.ml_additional_buy_count = 0
        self.ml_additional_capital_deployed = 0.0

        if market_history is not None:
            self.set_market_history(market_history)

    # ------------------------------------------------------------------
    # Market history
    # ------------------------------------------------------------------

    @staticmethod
    def _normalise_history(data):
        history = data.copy()

        if "Date" in history.columns and "Timestep" not in history.columns:
            history = history.rename(columns={"Date": "Timestep"})

        if "Price" in history.columns and "Close" not in history.columns:
            history["Close"] = pd.to_numeric(
                history["Price"].astype(str).str.replace(",", "", regex=False),
                errors="coerce",
            )

        if "Vol." in history.columns and "VolNum" not in history.columns:
            volume = (
                history["Vol."].astype(str)
                .str.replace(",", "", regex=False)
                .str.replace("K", "e3", regex=False)
                .str.replace("M", "e6", regex=False)
            )
            history["VolNum"] = pd.to_numeric(volume, errors="coerce")

        required = {"Timestep", "Close", "High", "VolNum"}
        missing = required.difference(history.columns)
        if missing:
            raise ValueError(
                "Level 3 model history is missing required columns: "
                + ", ".join(sorted(missing))
            )

        history = history.reset_index(drop=True)
        history["Timestep"] = pd.to_datetime(history["Timestep"], utc=True)

        for column in ("Close", "High", "VolNum"):
            history[column] = pd.to_numeric(history[column], errors="coerce")

        if history[["Close", "High", "VolNum"]].isna().any().any():
            raise ValueError("Level 3 model history contains invalid market values.")

        return history

    def set_market_history(self, data):
        self.market_history = self._normalise_history(data)
        self.decision_index = len(self.market_history) - 1

    def append_market_data(self, data):
        new_data = self._normalise_history(data)

        if self.market_history is None:
            self.market_history = new_data
        else:
            self.market_history = pd.concat(
                [self.market_history, new_data],
                ignore_index=True,
            )
            self.market_history = (
                self.market_history
                .drop_duplicates(subset=["Timestep"], keep="last")
                .sort_values("Timestep")
                .reset_index(drop=True)
            )

        self.decision_index = len(self.market_history) - 1

    # ------------------------------------------------------------------
    # ML scoring
    # ------------------------------------------------------------------

    def _score_buy_opportunity(self):
        if self.market_history is None:
            return float("nan"), float("nan")

        index = self.decision_index
        history = self.market_history.iloc[: index + 1].copy()

        tree = tree_probability(history, decision_index=index)
        lstm = lstm_probability(history, decision_index=index)
        return tree, lstm

    def _ml_additional_buy_signal(self):
        tree, lstm = self._score_buy_opportunity()
        self.last_tree_probability = tree
        self.last_lstm_probability = lstm

        signal = (
            np.isfinite(tree)
            and np.isfinite(lstm)
            and tree >= self.consensus_threshold
            and lstm >= self.consensus_threshold
        )
        self.last_ml_buy_signal = bool(signal)
        return bool(signal)

    # ------------------------------------------------------------------
    # Decision / execution
    # ------------------------------------------------------------------

    def make_decision(self, data, previous_price=None):
        # Level 2 remains responsible for its own scheduled DCA, volatility
        # rules, safeguards, and BUY/HOLD/SELL decision.
        base_decision = super().make_decision(data, previous_price=previous_price)

        self.last_tree_probability = None
        self.last_lstm_probability = None
        self.last_ml_buy_signal = False

        if self.market_history is None:
            return base_decision

        # Score the ML overlay independently of the Level 2 action. A qualifying
        # ML signal can add a purchase on BUY, HOLD, or SELL days; it never
        # changes the underlying Level 2 action itself.
        if not self._ml_additional_buy_signal():
            return base_decision

        normal_amount = (
            float(base_decision.get("amount", 0.0))
            if base_decision.get("action") == "BUY"
            else 0.0
        )
        available_after_normal = max(0.0, self.cash - normal_amount)
        extra = min(self.additional_buy_amount, available_after_normal)
        if extra <= 0:
            return base_decision

        decision = dict(base_decision)
        decision["ml_additional_amount"] = extra
        if base_decision.get("action") == "BUY":
            decision["amount"] = normal_amount + extra
        decision["level3_tree_probability"] = float(self.last_tree_probability)
        decision["level3_lstm_probability"] = float(self.last_lstm_probability)
        decision["level3_ml_additional_buy"] = True
        existing_reason = str(base_decision.get("reason", "Level 2 decision"))
        decision["reason"] = f"{existing_reason}; Level 3 ML additional buy"
        return decision

    def execute_decision(self, decision):
        extra = float(decision.get("ml_additional_amount", 0.0))

        if decision.get("action") == "BUY":
            normal_amount = decision["amount"] - extra
            btc_bought = self.buy(normal_amount, decision["price"])
            if btc_bought > 0:
                self.last_dca_day = decision["Timestep"]
                self.next_dca_multiplier = 1.0
        else:
            # Preserve the Level 2 HOLD/SELL action before the independent ML
            # purchase. This allows sale proceeds, if any, to become available.
            btc_bought = super().execute_decision(decision)

        if extra > 0:
            # Re-check available cash at execution time; never overspend.
            actual_extra = min(extra, self.cash)
            if actual_extra > 0:
                extra_btc = self.buy(actual_extra, decision["price"])
                btc_bought += extra_btc
                self.ml_additional_buy_count += 1
                self.ml_additional_capital_deployed += actual_extra

        return btc_bought
