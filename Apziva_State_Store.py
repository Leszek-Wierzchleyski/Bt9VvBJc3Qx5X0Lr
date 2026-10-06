#!/usr/bin/env python3
"""Persistent per-user state for the simulated trading application.

Streamlit session_state is intentionally ephemeral. This module stores the
simulated account/strategy state in a small SQLite database so browser refreshes
and Streamlit session recreation do not reset the user's account.

Only application state is stored. Chat messages are not persisted here.
"""

from __future__ import annotations

import json
import os
import sqlite3
from datetime import date, datetime
from pathlib import Path
from typing import Any, Optional


DEFAULT_DB_PATH = Path(__file__).resolve().parent / "data" / "trading_state.db"
SCHEMA_VERSION = 1


def _db_path() -> Path:
    return Path(os.getenv("APZIVA_STATE_DB_PATH", str(DEFAULT_DB_PATH)))


def _connect() -> sqlite3.Connection:
    path = _db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path, timeout=10)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA foreign_keys=ON")
    return connection


def _json_safe(value: Any) -> Any:
    """Convert common Python/pandas/numpy values to JSON-safe values."""
    if value is None or isinstance(value, (str, bool, int, float)):
        return value

    if isinstance(value, datetime):
        return {"__type__": "datetime", "value": value.isoformat()}

    if isinstance(value, date):
        return {"__type__": "date", "value": value.isoformat()}

    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}

    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]

    # numpy scalar compatibility without making numpy a runtime dependency.
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return _json_safe(item())
        except Exception:
            pass

    raise TypeError(f"Unsupported state value type: {type(value).__name__}")


def _from_json_safe(value: Any) -> Any:
    if isinstance(value, list):
        return [_from_json_safe(item) for item in value]

    if isinstance(value, dict):
        marker = value.get("__type__")
        if marker == "datetime":
            return datetime.fromisoformat(value["value"])
        if marker == "date":
            return date.fromisoformat(value["value"])
        return {key: _from_json_safe(item) for key, item in value.items()}

    return value


def _initialise_schema() -> None:
    with _connect() as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS account_state (
                user_key TEXT PRIMARY KEY,
                schema_version INTEGER NOT NULL,
                risk_level INTEGER NOT NULL,
                strategy_state TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        connection.commit()


def load_account_state(user_key: str) -> Optional[dict[str, Any]]:
    """Load one user's persisted strategy state, or return None if absent."""
    if not user_key:
        raise ValueError("user_key must not be empty.")

    _initialise_schema()

    with _connect() as connection:
        row = connection.execute(
            "SELECT schema_version, risk_level, strategy_state FROM account_state WHERE user_key = ?",
            (user_key,),
        ).fetchone()

    if row is None:
        return None

    schema_version, risk_level, state_json = row
    if int(schema_version) != SCHEMA_VERSION:
        raise ValueError(f"Unsupported persisted state schema version: {schema_version}")

    state = _from_json_safe(json.loads(state_json))
    if not isinstance(state, dict):
        raise ValueError("Persisted strategy state is invalid.")

    return {
        "risk_level": int(risk_level),
        "state": state,
    }


def save_account_state(user_key: str, strategy, risk_level: int) -> None:
    """Persist the current strategy instance for one authenticated user."""
    if not user_key:
        raise ValueError("user_key must not be empty.")

    state = _json_safe(dict(strategy.__dict__))
    state_json = json.dumps(state, separators=(",", ":"), allow_nan=False)

    _initialise_schema()

    with _connect() as connection:
        connection.execute(
            """
            INSERT INTO account_state (user_key, schema_version, risk_level, strategy_state, updated_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(user_key) DO UPDATE SET
                schema_version=excluded.schema_version,
                risk_level=excluded.risk_level,
                strategy_state=excluded.strategy_state,
                updated_at=excluded.updated_at
            """,
            (
                user_key,
                SCHEMA_VERSION,
                int(risk_level),
                state_json,
                datetime.now().astimezone().isoformat(),
            ),
        )
        connection.commit()
