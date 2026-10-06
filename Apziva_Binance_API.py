#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 15 23:29:52 2026

@author: leszekwierzchleyski
"""

import time

import requests
import pandas as pd


class BinanceMarketData:
    """
    Read-only Binance market-data provider.

    Binance BTCUSDT prices are quoted in USD/USDT. The simulated trading
    account is GBP-denominated, so OHLC quote values are converted from USD
    to GBP before being returned to the trading strategies.

    The FX rate is cached briefly to avoid making a second FX request on
    every market-data call. A cached rate is used as a fallback if the FX
    provider is temporarily unavailable.
    """

    FX_URL = "https://api.frankfurter.app/latest"
    FX_CACHE_SECONDS = 300

    def __init__(
        self,
        symbol="BTCUSDT",
        base_url="https://api.binance.com",
        default_interval="1d",
        default_limit=100,
    ):
        self.symbol = str(symbol).strip().upper()
        self.base_url = str(base_url).rstrip("/")
        self.default_interval = str(default_interval)
        self.default_limit = int(default_limit)

        self.price_currency = "GBP"
        self.source_quote_currency = "USD"
        self._usd_gbp_rate = None
        self._fx_timestamp = 0.0

    def _get_usd_gbp_rate(self):
        """Return the current USD→GBP reference rate, with short caching."""

        now = time.monotonic()

        if (
            self._usd_gbp_rate is not None
            and (now - self._fx_timestamp) < self.FX_CACHE_SECONDS
        ):
            return self._usd_gbp_rate

        try:
            response = requests.get(
                self.FX_URL,
                params={"from": "USD", "to": "GBP"},
                timeout=10,
            )
            response.raise_for_status()

            payload = response.json()
            rate = float(payload["rates"]["GBP"])

            if not rate > 0:
                raise ValueError("USD/GBP exchange rate must be positive.")

            self._usd_gbp_rate = rate
            self._fx_timestamp = now
            return rate

        except Exception:
            # If the FX service is temporarily unavailable, retain the most
            # recent known rate rather than silently treating USD as GBP.
            if self._usd_gbp_rate is not None:
                return self._usd_gbp_rate
            raise RuntimeError(
                "Unable to obtain the USD-to-GBP exchange rate."
            )

    def get_candles(self, interval=None, limit=None):
        interval = self.default_interval if interval is None else interval
        limit = self.default_limit if limit is None else int(limit)

        endpoint = f"{self.base_url}/api/v3/klines"

        params = {
            "symbol": self.symbol,
            "interval": interval,
            "limit": limit,
        }

        response = requests.get(endpoint, params=params, timeout=10)
        response.raise_for_status()

        data = response.json()

        columns = [
            "Open Time",
            "Open",
            "High",
            "Low",
            "Close",
            "Volume",
            "Close Time",
            "Quote Asset Volume",
            "Number of Trades",
            "Taker Buy Base Volume",
            "Taker Buy Quote Volume",
            "Ignore",
        ]

        df = pd.DataFrame(data, columns=columns)

        df["Timestep"] = pd.to_datetime(
            df["Open Time"],
            unit="ms",
        )

        numerical_columns = [
            "Open",
            "High",
            "Low",
            "Close",
            "Volume",
        ]

        for column in numerical_columns:
            df[column] = pd.to_numeric(df[column])

        # BTCUSDT is quoted in USD/USDT. Convert all quote-denominated
        # price/value fields to GBP so the simulator has one consistent
        # account currency.
        usd_gbp = self._get_usd_gbp_rate()

        quote_value_columns = [
            "Open",
            "High",
            "Low",
            "Close",
            "Quote Asset Volume",
            "Taker Buy Quote Volume",
        ]

        for column in quote_value_columns:
            df[column] = pd.to_numeric(df[column]) * usd_gbp

        return df

