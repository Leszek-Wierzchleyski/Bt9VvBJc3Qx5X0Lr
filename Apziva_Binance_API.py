#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 15 23:29:52 2026

@author: leszekwierzchleyski
"""

import requests
import pandas as pd


class BinanceMarketData:

    def __init__(self, symbol="BTCUSDT", base_url="https://api.binance.com",
                 default_interval="1d", default_limit=100):
        self.symbol = str(symbol).strip().upper()
        self.base_url = str(base_url).rstrip("/")
        self.default_interval = str(default_interval)
        self.default_limit = int(default_limit)

    def get_candles(self, interval=None, limit=None):

        interval = self.default_interval if interval is None else interval
        limit = self.default_limit if limit is None else int(limit)

        endpoint = f"{self.base_url}/api/v3/klines"

        params = {
            "symbol": self.symbol,
            "interval": interval,
            "limit": limit
        }

        response = requests.get(endpoint, params=params)

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
            "Ignore"
        ]

        df = pd.DataFrame(data, columns=columns)

        # Convert timestamp
        df["Timestep"] = pd.to_datetime(
            df["Open Time"],
            unit="ms"
        )

        # Convert numerical columns
        numerical_columns = [
            "Open",
            "High",
            "Low",
            "Close",
            "Volume"
        ]

        for column in numerical_columns:
            df[column] = pd.to_numeric(df[column])

        return df