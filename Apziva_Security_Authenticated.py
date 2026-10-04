#!/usr/bin/env python3
"""Security helpers for the Bitcoin trading application.

This module provides three deliberately small security primitives:

1. Anonymous per-session identifiers for application state isolation.
2. Privacy-conscious audit logging that never records user messages,
   portfolio values, credentials, or secret values.
3. Secret retrieval from environment variables or Streamlit secrets rather
   than source-code configuration files.

The LLM is never given direct access to this module's secret values.
"""

from __future__ import annotations

import hashlib
import logging
import os
import secrets
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional


LOGGER_NAME = "apziva.security"
LOG_DIR = Path(__file__).resolve().parent / "logs"
MAX_LOG_BYTES = 1_000_000
BACKUP_COUNT = 3


class _RedactSecretFilter(logging.Filter):
    """Prevent obvious secret-like fields from reaching the log handler."""

    _SENSITIVE_KEYS = (
        "api_key",
        "api_secret",
        "access_token",
        "refresh_token",
        "password",
        "secret",
        "credential",
        "authorization",
    )

    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage().lower()
        if any(key in message for key in self._SENSITIVE_KEYS):
            record.msg = "Sensitive information was suppressed from the audit log."
            record.args = ()
        return True


def get_session_id() -> str:
    """Return a random anonymous identifier for the current Streamlit session."""
    return secrets.token_urlsafe(24)


def hash_session_id(session_id: str) -> str:
    """Return a non-reversible identifier suitable for logs."""
    return hashlib.sha256(session_id.encode("utf-8")).hexdigest()[:16]


def hash_identity(identity: str) -> str:
    """Return a non-reversible identifier for an authenticated identity."""
    return hashlib.sha256(identity.strip().lower().encode("utf-8")).hexdigest()[:16]


def get_security_logger() -> logging.Logger:
    """Return the privacy-conscious application audit logger."""
    logger = logging.getLogger(LOGGER_NAME)

    if logger.handlers:
        return logger

    LOG_DIR.mkdir(parents=True, exist_ok=True)

    handler = RotatingFileHandler(
        LOG_DIR / "security.log",
        maxBytes=MAX_LOG_BYTES,
        backupCount=BACKUP_COUNT,
        encoding="utf-8",
    )
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s | %(levelname)s | %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S%z",
        )
    )
    handler.addFilter(_RedactSecretFilter())

    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    return logger


def audit_event(
    event: str,
    *,
    session_id: Optional[str] = None,
    status: str = "ok",
    detail: Optional[str] = None,
) -> None:
    """Write a minimal audit event without storing user content or finances."""
    logger = get_security_logger()
    anonymous_session = hash_session_id(session_id) if session_id else "system"

    # ``event`` and ``status`` are application-controlled values. ``detail``
    # is deliberately optional and should contain only non-sensitive metadata.
    message = f"session={anonymous_session} event={event} status={status}"
    if detail:
        message += f" detail={detail}"
    logger.info(message)


def get_secret(name: str, *, required: bool = False) -> Optional[str]:
    """Retrieve a secret without putting it in source code.

    Environment variables are preferred. Streamlit secrets are supported when
    the application is running under Streamlit. The returned value must never
    be logged or passed to the LLM.
    """
    value = os.getenv(name)

    if value:
        return value

    try:
        import streamlit as st

        value = st.secrets.get(name)
    except Exception:
        value = None

    if value:
        return str(value)

    if required:
        raise RuntimeError(f"Required secret '{name}' is not configured.")

    return None
