# dashboard.py
# 🦅 ARGUS MISSION CONTROL - V3.7 (CORTEX WIRED + FLEXIBLE RUNTIME ROOT)

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import time
import requests
import json
import sys
from pathlib import Path
from dotenv import load_dotenv

# ---------------------------
# RUNTIME ROOT DETECTION
# ---------------------------

current_file = Path(__file__).resolve()
repo_root = current_file.parent  # where dashboard.py lives


def _detect_runtime_root(repo_root: Path) -> Path:
    """
    Find the Argus runtime root where:
      - src/real_broker.py lives
      - flight_recorder.csv, cortex.json, argus.log live (on server)
      - OR runtime/argus/src/real_broker.py (on local mono-repo)
    """

    # 1) Server layout: dashboard.py and src/ under same root (/opt/argus)
    direct_src = repo_root / "src" / "real_broker.py"
    if direct_src.exists():
        return repo_root

    # 2) Local mono-repo layout: runtime/argus/src/real_broker.py
    candidate = repo_root / "runtime" / "argus"
    if (candidate / "src" / "real_broker.py").exists():
        return candidate

    # 3) Slight variant: maybe you're already in runtime/argus
    if (repo_root / "src" / "real_broker.py").exists():
        return repo_root

    # 4) Last resort: just use repo_root and hope src is importable some other way
    return repo_root


project_root = _detect_runtime_root(repo_root)

# Make sure runtime root is importable
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


# ---------------------------
# ENV LOADING
# ---------------------------

def _find_env_file(start: Path) -> Path | None:
    for p in (start, *start.parents):
        candidate = p / ".env"
        if candidate.exists():
            return candidate
    return None


_env = _find_env_file(project_root)
if _env is not None:
    load_dotenv(_env)
else:
    load_dotenv()

# ---------------------------
# IMPORT REALBROKER
# ---------------------------

try:
    from src.real_broker import RealBroker
except ImportError as e:
    broker_import_err = str(e)

    # Optional: if mono-repo is installed as a package, try that
    try:
        import importlib

        RealBroker = importlib.import_module(
            "iteradynamics_monorepo.src.real_broker"
        ).RealBroker  # type: ignore[attr-defined]
    except Exception:
        st.error(
            "❌ Could not import RealBroker.\n\n"
            "Checked:\n"
            f"  • {project_root / 'src' / 'real_broker.py'}\n"
            f"  • mono-repo package 'iteradynamics_monorepo.src.real_broker'\n\n"
            "Make sure that:\n"
            "  1) On the server: `dashboard.py` and `src/real_broker.py` live under the same root (e.g., /opt/argus).\n"
            "  2) On your local mono-repo: `runtime/argus/src/real_broker.py` exists.\n\n"
            f"Debug detail: {broker_import_err}"
        )
        st.stop()

# ---------------------------
# STREAMLIT CONFIG / STYLES
# ---------------------------

st.set_page_config(
    page_title="Argus Commander",
    page_icon="🦅",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
<style>
    .stApp { background-color: #0e1117; }
    h1, h2, h3, h4, h5, h6 {
        color: #ffffff !important;
        font-family: 'Monospace';
    }
    [data-testid="stMetricValue"] {
        font-size: 26px;
        font-family: 'Monospace';
        color: #00ff00;
    }
    div[data-testid="stMetricLabel"] > label {
        color: #ffffff !important;
        font-size: 14px;
        font-weight: bold;
    }
    .log-box {
        font-family: 'Courier New', monospace;
        font-size: 14px;
        line-height: 1.5;
        color: #00ff00 !important;
        background-color: #000000;
        padding: 15px;
        border: 1px solid #333;
        border-radius: 5px;
        height: 350px;
        overflow-y: scroll;
        white-space: pre-wrap;
    }
    .log-box * {
        color: #00ff00 !important;
        opacity: 1 !important;
    }
</style>
""",
    unsafe_allow_html=True,
)

# ---------------------------
# LIVE DATA FUNCTIONS
# ---------------------------

@st.cache_resource
def get_broker():
    return RealBroker()


def get_live_price():
    try:
        url = "https://api.coinbase.com/v2/prices/BTC-USD/spot"
        resp = requests.get(url, timeout=3)
        return float(resp.json()["data"]["amount"])
    except Exception:
        return 0.0


def load_market_data():
    csv_path = project_root / "flight_recorder.csv"
    if csv_path.exists():
        try:
            df = pd.read_csv(csv_path)
            if "Close" not in df.columns:
                df = pd.read_csv(
                    csv_path,
                    names=["Timestamp", "Open", "High", "Low", "Close", "Volume"],
                    header=0,
                )
            df["Timestamp"] = pd.to_datetime(df["Timestamp"])
            df.sort_values("Timestamp", inplace=True)
            return df
        except Exception:
            pass
    return pd.DataFrame()


def read_logs():
    log_path = project_root / "argus.log"
    if log_path.exists():
        try:
            with open(log_path, "r", encoding="utf-8") as f:
                return "".join(f.readlines()[-300:])
        except Exception:
            return "Error reading log."
    return "Waiting for logs..."


# ---------------------------
# 🎯 CORTEX TELEMETRY
# ---------------------------

def get_cortex_state():
    """
    Read the latest brain/regime snapshot from cortex.json.

    This is written atomically by apex_core.signal_generator via _atomic_write_json,
    so dashboard should always see a fully-formed JSON payload or nothing.
    """
    cortex_path = project_root / "cortex.json"

    state = {
        "timestamp_utc": None,
        "regime": "Searching...",
        "risk_mult": 0.0,
        "severity": 0.0,
        "raw_signal": "N/A",
        "wallet_verified": False,
        "cash_usd": None,
        "btc": None,
        "btc_notional_usd": None,
        "emergency_exit": False,
        "conviction_score": 0,
    }

    if not cortex_path.exists():
        return state

    try:
        with open(cortex_path, "r", encoding="utf-8") as f:
            payload = json.load(f)

        regime = payload.get("regime", state["regime"])
        risk_mult = float(payload.get("risk_mult", state["risk_mult"]) or 0.0)
        severity = float(payload.get("severity", state["severity"]) or 0.0)
        raw_signal = payload.get("raw_signal", state["raw_signal"])
        wallet_verified = bool(payload.get("wallet_verified", False))
        cash_usd = payload.get("cash_usd")
        btc = payload.get("btc")
        btc_notional = payload.get("btc_notional_usd")
        emergency_exit = bool(payload.get("emergency_exit", False))
        ts = payload.get("timestamp_utc")

        conviction_score = int(max(0.0, min(1.0, risk_mult)) * 100)

        state.update(
            {
                "timestamp_utc": ts,
                "regime": regime,
                "risk_mult": risk_mult,
                "severity": severity,
                "raw_signal": raw_signal,
                "wallet_verified": wallet_verified,
                "cash_usd": cash_usd,
                "btc": btc,
                "btc_notional_usd": btc_notional,
                "emergency_exit": emergency_exit,
                "conviction_score": conviction_score,
            }
        )
    except Exception:
        # If cortex.json is corrupt or mid-write, fall back to defaults
        pass

    return state


def get_auto_entry():
    """Pulls from trade_state.json for the dashboard metrics."""
    path = project_root / "trade_state.json"
    if path.exists():
        try:
            with open(path, "r") as f:
                return float(json.load(f).get("entry_price", 90163.00))
        except Exception:
            pass
    return 90163.00


# ---------------------------
# MAIN DASHBOARD
# ---------------------------

def main():
    st.title("🦅 ARGUS // LIVE COMMANDER")

    # --- Broker + market snapshot ---
    try:
        broker = get_broker()
        current_price = get_live_price()
        df = load_market_data()
        avg_entry = get_auto_entry()

        # 🔑 NEW: derive balances from RealBroker.get_wallet_snapshot()
        cash, btc = broker.get_wallet_snapshot()
        btc_exposure_usd = btc * current_price
        equity = cash + btc_exposure_usd

        if btc > 0:
            unrealized_pnl_usd = (current_price - avg_entry) * btc
            pnl_pct = (unrealized_pnl_usd / (avg_entry * btc)) * 100
            breakeven = avg_entry * 1.002
        else:
            unrealized_pnl_usd, pnl_pct, breakeven = 0.0, 0.0, 0.0
    except Exception as e:
        st.error(f"System Error: {e}")
        return

    # --- ROW 1: STATUS ---
    st.markdown("### 🏦 Liquid Status")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Net Liquid Equity", f"${equity:,.2f}", delta=f"{unrealized_pnl_usd:+.2f}")
    k2.metric("Dry Powder (USD)", f"${cash:,.2f}")
    k3.metric(
        "BTC Exposure",
        f"${btc_exposure_usd:,.2f}",
        f"{btc:.6f} BTC",
    )
    k4.metric("Market Price", f"${current_price:,.2f}")

    # --- ROW 2: ANALYSIS ---
    st.markdown("### 📊 Active Position Analysis")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Auto-Entry Price", f"${avg_entry:,.2f}")
    m2.metric("Unrealized P&L ($)", f"${unrealized_pnl_usd:+.2f}")
    color_mode = "normal" if pnl_pct >= 0 else "inverse"
    m3.metric("Unrealized P&L (%)", f"{pnl_pct:+.2f}%", delta=pnl_pct, delta_color=color_mode)
    m4.metric("Breakeven Price", f"${breakeven:,.2f}")

    st.markdown("---")

    # --- ROW 3: CHARTS & GAUGE ---
    c1, c2 = st.columns([3, 1])

    with c1:
        st.subheader("Performance Curve")
        if not df.empty:
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=df["Timestamp"],
                    y=df["Close"],
                    mode="lines",
                    line=dict(color="#00ff00", width=2),
                )
            )
            fig.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                height=350,
            )
            st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.subheader("Cortex State")
        cortex = get_cortex_state()

        fig_gauge = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=cortex["conviction_score"],
                title={"text": "Risk Multiplier"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": "darkblue"},
                    "steps": [
                        {"range": [0, 40], "color": "red"},
                        {"range": [40, 70], "color": "orange"},
                        {"range": [70, 100], "color": "green"},
                    ],
                },
            )
        )
        fig_gauge.update_layout(
            height=280,
            paper_bgcolor="rgba(0,0,0,0)",
            font={"color": "white"},
            margin=dict(l=20, r=20, t=40, b=20),
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

        regime_html = f"<p style='text-align: center; color: gray;'>Regime: {cortex['regime']}</p>"
        signal_html = f"<p style='text-align: center; color: gray;'>Raw Signal: {cortex['raw_signal']}</p>"
        sev_html = f"<p style='text-align: center; color: gray;'>Severity: {cortex['severity']:.2f}</p>"
        st.markdown(regime_html + signal_html + sev_html, unsafe_allow_html=True)

    # --- ROW 4: LOGS ---
    st.subheader("📜 System Logs")
    st.markdown(f"<div class='log-box'>{read_logs()}</div>", unsafe_allow_html=True)

    # Auto-refresh every 10 seconds
    time.sleep(10)
    st.rerun()


if __name__ == "__main__":
    main()
