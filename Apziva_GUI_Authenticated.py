#!/usr/bin/env python3
"""Streamlit GUI for the Bitcoin trading assistant.

The chat is the primary interface. Portfolio/status information is displayed
around the conversation; all consequential actions continue to use the
existing LlamaInterface confirmation workflow.
"""

from __future__ import annotations

import os

import streamlit as st
from llama_cpp import Llama

from Apziva_Security_Authenticated import audit_event, get_secret, get_session_id, hash_identity
from Apziva_Config import load_config

from Apziva_Binance_API import BinanceMarketData
from Apziva_LLM_Interface_Authenticated import LlamaInterface, MODEL_PATH
from Apziva_Risk_Manager_Authenticated import RiskManager
from Apziva_Trading_Strategy_Level1 import TradingStrategyLevel1
from Apziva_Trading_Strategy_Level2_Final import TradingStrategyLevel2
from Apziva_Trading_Strategy_Level3 import TradingStrategyLevel3


st.set_page_config(
    page_title="Bitcoin Trading Assistant",
    page_icon="₿",
    layout="wide",
    initial_sidebar_state="expanded",
)

MAX_USER_MESSAGE_CHARS = 4000


# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------

st.markdown(
    """
    <style>
    .block-container { max-width: 1400px; padding-top: 2rem; }
    .status-card {
        border: 1px solid rgba(128,128,128,.25);
        border-radius: 12px;
        padding: 14px 16px;
        min-height: 92px;
    }
    .status-label {
        font-size: .78rem;
        opacity: .70;
        margin-bottom: 4px;
    }
    .status-value {
        font-size: 1.35rem;
        font-weight: 650;
    }
    .app-title { margin-bottom: 0; }
    .app-subtitle { opacity: .65; margin-top: -8px; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Authentication / web-access policy
# ---------------------------------------------------------------------------


def _setting_enabled(name: str, default: bool = False) -> bool:
    value = get_secret(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _allowed_emails():
    raw = get_secret("ALLOWED_EMAILS")
    if not raw:
        return set()
    return {item.strip().lower() for item in raw.split(",") if item.strip()}


def require_authentication():
    """Require OIDC authentication when explicitly enabled for deployment.

    Local development remains unchanged unless AUTH_REQUIRED is enabled.
    When enabled, authentication is handled by Streamlit/OIDC rather than a
    custom password system. An optional email allowlist provides basic
    application-level authorization for private demos.
    """
    if not _setting_enabled("AUTH_REQUIRED", default=False):
        return

    user = getattr(st, "user", None)
    is_logged_in = bool(getattr(user, "is_logged_in", False)) if user is not None else False

    if not is_logged_in:
        st.title("Bitcoin Trading Assistant")
        st.subheader("Private application")
        st.write("Please sign in to continue.")
        if hasattr(st, "login"):
            st.button("Log in", on_click=st.login, use_container_width=True)
        else:
            st.error("Authentication is required, but this Streamlit version does not provide st.login().")
        st.stop()

    email = str(getattr(user, "email", "")).strip().lower()
    allowed = _allowed_emails()
    if allowed and email not in allowed:
        identity = email or str(getattr(user, "sub", "unknown"))
        audit_event(
            "authorization_denied",
            detail=f"identity={hash_identity(identity)}",
            status="error",
        )
        st.title("Access denied")
        st.write("Your account is authenticated but is not authorised to use this application.")
        if hasattr(st, "logout"):
            st.button("Log out", on_click=st.logout, use_container_width=True)
        st.stop()

    identity = email or str(getattr(user, "sub", "unknown"))
    audit_event(
        "authentication_accepted",
        detail=f"identity={hash_identity(identity)}",
    )

    if hasattr(st, "logout"):
        with st.sidebar:
            st.caption("Authenticated session")
            st.button("Log out", on_click=st.logout, use_container_width=True)


require_authentication()


# ---------------------------------------------------------------------------
# Application bootstrap
# ---------------------------------------------------------------------------


def _level1_factory(budget: float, dca_amount: float):
    return TradingStrategyLevel1(
        budget=budget,
        dca_amount=dca_amount,
    )


def _level2_factory(budget: float, dca_amount: float):
    return TradingStrategyLevel2(
        budget=budget,
        dca_amount=dca_amount,
    )


def _level3_factory(budget: float, dca_amount: float):
    return TradingStrategyLevel3(
        budget=budget,
        dca_amount=dca_amount,
    )


def load_application_config():
    """Load validated application configuration from Google Sheet/cache."""
    spreadsheet_id = get_secret("APZIVA_CONFIG_SPREADSHEET_ID")
    credentials_file = get_secret("GOOGLE_SERVICE_ACCOUNT_FILE")

    config, source = load_config(
        spreadsheet_id=spreadsheet_id,
        credentials_file=credentials_file,
        worksheet_name="Config",
    )

    return config, source


def build_runtime(config):
    """Create the initial application state from validated configuration."""
    factories = {
        1: _level1_factory,
        2: _level2_factory,
        3: _level3_factory,
    }

    selected_level = int(config["default_risk_level"])
    strategy = factories[selected_level](
        budget=0.0,
        dca_amount=float(config["default_dca_amount"]),
    )
    strategy.trading_enabled = bool(config["trading_enabled_by_default"])

    risk_manager = RiskManager(
        current_strategy=strategy,
        strategy_factories=factories,
    )

    return strategy, risk_manager


@st.cache_resource(show_spinner=False)
def get_llm_resources():
    """Load the quantised GGUF model once for the Streamlit process."""
    model_path = os.getenv("LLAMA_MODEL_PATH", MODEL_PATH)
    return Llama(
        model_path=model_path,
        n_ctx=4096,
        n_threads=max(1, min(8, os.cpu_count() or 1)),
        verbose=False,
    )


def initialise_session():
    """Create isolated mutable state for this browser session only."""
    if "session_id" not in st.session_state:
        st.session_state.session_id = get_session_id()
        audit_event("session_created", session_id=st.session_state.session_id)

    config, config_source = load_application_config()
    st.session_state.config = config
    st.session_state.config_source = config_source

    if "strategy" not in st.session_state or "risk_manager" not in st.session_state:
        strategy, risk_manager = build_runtime(config)
        st.session_state.strategy = strategy
        st.session_state.risk_manager = risk_manager
        audit_event(
            "session_state_initialized",
            session_id=st.session_state.session_id,
            detail=f"config_source={config_source}",
        )

    if "interface" not in st.session_state:
        model = get_llm_resources()
        st.session_state.interface = LlamaInterface(
            trading_strategy=st.session_state.strategy,
            market_data=BinanceMarketData(
                symbol=config["market_symbol"],
                base_url="https://api.binance.com",
                default_interval=config["market_data_interval"],
                default_limit=int(config["market_data_limit"]),
            ),
            risk_manager=st.session_state.risk_manager,
            model=model,
            device="cpu",
            session_id=st.session_state.session_id,
        )

    return (
        st.session_state.strategy,
        st.session_state.risk_manager,
        st.session_state.interface,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def money(value):
    return f"£{float(value):,.2f}"


def card(label, value):
    st.markdown(
        f"""
        <div class="status-card">
            <div class="status-label">{label}</div>
            <div class="status-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def refresh_portfolio(strategy, market_data):
    try:
        return {
            "portfolio": LlamaInterface.get_portfolio,
        }
    except Exception:
        return None


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": r"""# Welcome to the Bitcoin Trading Agent

This interface provides a natural-language control layer for a simulated Bitcoin trading system. You can ask the agent about your portfolio, performance, strategy, risk level, and trading status, or make changes to your account configuration.

**No real trades are placed.** Market data is read-only, and all trading activity takes place within the simulated portfolio.

## Risk Levels

The system has three risk levels. Your selected risk level determines which trading strategy is active:

### Low Risk — Level 1
**Regime-Adaptive DCA**

Dollar-Cost Averaging (DCA) involves investing a predetermined amount at regular intervals rather than attempting to time the market.

The strategy adapts the frequency of scheduled DCA purchases according to the detected Bitcoin market regime:

- **Bullish:** DCA every 10 days
- **Neutral:** DCA every 14 days
- **Bearish:** DCA every 21 days

A portfolio drawdown safeguard can temporarily stop new purchases during significant declines.

### Medium Risk — Level 2
**Volatility-Aware Trading**

Includes everything in Level 1, with additional responses to significant daily Bitcoin price movements.

The strategy responds to significant daily Bitcoin price movements:

- A daily downside move of 4% or more doubles the next scheduled DCA purchase.
- A daily upside move of 5% or more triggers a sale equal to 15% of the gain attributable to that day's uptick.

If the relevant volatility conditions are not triggered, the underlying Level 1 strategy continues to operate.

### High Risk — Level 3
**ML-Enhanced Trading**

Includes everything in Level 2, with an additional machine-learning accumulation layer.

Two independent models — a tree-based model and an LSTM — must agree before an additional opportunistic purchase is made.

If the ML conditions are not met, the underlying Level 2 strategy continues to operate. If Level 2's volatility conditions are not triggered, it continues according to the underlying Level 1 strategy.

In this way, the higher risk levels add additional trading behaviour on top of the lower levels rather than replacing them.

## What You Can Configure

You can use natural language to:

- **Add or withdraw simulated capital**
- **Set your DCA amount**
- **Change your risk level**
- **Start or stop trading**
- **View your portfolio and performance**
- **Check your current DCA regime and schedule**
- **Sell some or all of your Bitcoin**

The **DCA frequency is determined automatically by the trading strategy** and cannot be configured manually.

## Important

The agent separates **strategy decisions from account controls**. Changing your risk level changes which trading strategy is active; it does not manually tell the system when to buy or sell.

Actions that change your simulated portfolio or account configuration require explicit confirmation before they are executed.

You can simply ask questions or give instructions in ordinary language — for example:

> “What is my current portfolio?”

> “Set my risk level to high.”

> “Add £5,000 to the account.”

> “Set my DCA amount to £250.”

> “How is the current DCA schedule determined?”

> “Stop trading.”

The system will explain what it intends to do before executing any action that changes your account.
"""
        }
    ]

strategy, risk_manager, interface = initialise_session()

# Risk changes replace the strategy object. Always read the authoritative
# strategy from the session's risk manager after each Streamlit rerun.
strategy = risk_manager.current_strategy
st.session_state.strategy = strategy
interface.trading_strategy = strategy


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.markdown('<h1 class="app-title">₿ Bitcoin Trading Assistant</h1>', unsafe_allow_html=True)
st.markdown(
    '<p class="app-subtitle">Conversational control interface for the simulated trading system</p>',
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Status strip
# ---------------------------------------------------------------------------

market_data = BinanceMarketData()

try:
    portfolio = interface.get_portfolio()
except Exception as exc:
    portfolio = {
        "cash": float(strategy.cash),
        "btc_held": float(strategy.btc_held),
        "btc_price": 0.0,
        "portfolio_value": float(strategy.portfolio_value(0.0)),
    }
    audit_event(
        "market_data_unavailable",
        session_id=st.session_state.session_id,
        status="error",
    )
    st.sidebar.warning("Live market data is currently unavailable. Using the local portfolio state.")

risk = risk_manager.get_status()
trading = interface.get_trading_status()
dca = interface.get_dca_status()

cols = st.columns(6)
with cols[0]:
    card("Portfolio", money(portfolio["portfolio_value"]))
with cols[1]:
    card("Cash", money(portfolio["cash"]))
with cols[2]:
    card("BTC held", f"{portfolio['btc_held']:.6f}")
with cols[3]:
    card("BTC price", money(portfolio["btc_price"]))
with cols[4]:
    card("Risk", risk["name"])
with cols[5]:
    card("Trading", "ON" if trading["trading_enabled"] else "OFF")


st.divider()


# ---------------------------------------------------------------------------
# Main layout
# ---------------------------------------------------------------------------

chat_col, info_col = st.columns([2.8, 1], gap="large")

with chat_col:
    st.subheader("Trading Assistant")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    prompt = st.chat_input("Ask about your portfolio or tell me what you want to change…")

    if prompt:
        if len(prompt) > MAX_USER_MESSAGE_CHARS:
            st.error(f"Please keep messages below {MAX_USER_MESSAGE_CHARS:,} characters.")
            st.stop()

        audit_event(
            "chat_request_received",
            session_id=st.session_state.session_id,
        )
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking…"):
                try:
                    response = interface.chat(prompt)
                except Exception as exc:
                    audit_event(
                        "chat_request_failed",
                        session_id=st.session_state.session_id,
                        status="error",
                    )
                    response = "I couldn't process that request. Please try again."
            st.markdown(response)

        st.session_state.messages.append({"role": "assistant", "content": response})
        st.rerun()


with info_col:
    st.subheader("Account")

    st.divider()
    st.subheader("Privacy")
    st.caption("Chat messages are kept only in this browser session and are not written to the application audit log.")

    if st.button("Clear conversation", use_container_width=True):
        # Clearing the conversation also removes any unconfirmed actions so
        # a user cannot accidentally lose the visible confirmation context.
        st.session_state.messages = st.session_state.messages[:1]
        st.session_state.interface.pending_action = None
        st.session_state.interface.pending_actions = []
        audit_event(
            "conversation_cleared",
            session_id=st.session_state.session_id,
        )
        st.rerun()

    st.divider()

with info_col:
    st.subheader("Account")

    st.metric("DCA amount", money(dca["dca_amount"]))
    st.metric("Regime DCA interval", f"Every {dca['effective_dca_frequency_days']} days")

    last_dca = dca["last_dca_day"]
    st.caption(f"Last DCA: {last_dca if last_dca else 'Not yet executed'}")

    st.divider()

    st.markdown(
        f"**Risk Level**  \n{risk['name']} — {risk['description']}"
    )

    st.markdown(
        f"**Trading**  \n{'Enabled' if trading['trading_enabled'] else 'Disabled'}"
    )

    if interface.pending_action is not None:
        st.warning("An action is awaiting confirmation. Reply `yes` or `no` in the chat.")

    st.divider()
    st.caption("All state-changing actions require explicit confirmation.")
    st.caption("The system is configured for simulated/read-only trading activity.")
