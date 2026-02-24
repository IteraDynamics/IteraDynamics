# dashboard.py
# 🦅 ARGUS MISSION CONTROL - V4.2 (PRIME-NATIVE UI + BRIGHT LABELS + SAFE LOG RENDER + NO add_vline ANYWHERE)

from __future__ import annotations

import os
import sys
import json
import time
import html
import requests
import pandas as pd
import plotly.graph_objects as go

import streamlit as st
from pathlib import Path
from dotenv import load_dotenv
from datetime import datetime, timezone

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
    if (repo_root / "src" / "real_broker.py").exists():
        return repo_root

    candidate = repo_root / "runtime" / "argus"
    if (candidate / "src" / "real_broker.py").exists():
        return candidate

    if (repo_root / "src" / "real_broker.py").exists():
        return repo_root

    return repo_root


project_root = _detect_runtime_root(repo_root)

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
    load_dotenv(_env, override=False)
else:
    load_dotenv(override=False)


def _env_bool(name: str, default: bool = False) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return v.strip().lower() in {"1", "true", "yes", "y", "on"}


# ---------------------------
# IMPORT REALBROKER
# ---------------------------

RealBroker = None
broker_import_err = None

try:
    from src.real_broker import RealBroker
except ImportError as e:
    broker_import_err = str(e)

if RealBroker is None:
    # Fallback: try direct file import from detected project_root
    try:
        import importlib.util
        broker_path = project_root / "src" / "real_broker.py"
        if broker_path.exists():
            spec = importlib.util.spec_from_file_location("real_broker", broker_path)
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                RealBroker = module.RealBroker
    except Exception as e2:
        broker_import_err = f"{broker_import_err}; fallback also failed: {e2}"

# Product and namespaced paths (ARGUS_PRODUCT_ID)
try:
    from config import (
        PRODUCT_ID,
        FLIGHT_RECORDER_PATH,
        PRIME_STATE_LIVE_PATH,
        PRIME_STATE_PAPER_PATH,
        TRADE_STATE_PATH,
        CORTEX_PATH,
        LOG_PATH,
    )
except ImportError:
    PRODUCT_ID = os.getenv("ARGUS_PRODUCT_ID", "BTC-USD")
    FLIGHT_RECORDER_PATH = project_root / "flight_recorder.csv"
    PRIME_STATE_LIVE_PATH = project_root / "prime_state.json"
    PRIME_STATE_PAPER_PATH = project_root / "paper_prime_state.json"
    TRADE_STATE_PATH = project_root / "trade_state.json"
    CORTEX_PATH = project_root / "cortex.json"
    LOG_PATH = project_root / "argus.log"

if RealBroker is None:
    st.error(
        "❌ Could not import RealBroker.\n\n"
        "Checked:\n"
        f"  • sys.path import 'src.real_broker'\n"
        f"  • Direct file: {project_root / 'src' / 'real_broker.py'}\n\n"
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

# iOS Home Screen Icon - Replace URL with your hosted icon location
# Options: GitHub raw URL, your server's static path, or CDN
APPLE_TOUCH_ICON_URL = "https://raw.githubusercontent.com/IteraDynamics/IteraDynamics/main/runtime/argus/apple-touch-icon.png"

if APPLE_TOUCH_ICON_URL:
    st.markdown(
        f'<link rel="apple-touch-icon" href="{APPLE_TOUCH_ICON_URL}">',
        unsafe_allow_html=True
    )

st.markdown(
    """
<style>
    .stApp { background-color: #0e1117; }

    h1, h2, h3, h4, h5, h6 {
        color: #ffffff !important;
        font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
        letter-spacing: 0.5px;
    }

    /* Metric values */
    [data-testid="stMetricValue"] {
        font-size: 26px;
        font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
        color: #00ff00 !important;
        opacity: 1 !important;
    }

    /* Metric labels (fix: too dark on mobile) */
    div[data-testid="stMetricLabel"],
    div[data-testid="stMetricLabel"] > div,
    div[data-testid="stMetricLabel"] > label,
    div[data-testid="stMetricLabel"] * {
        color: #d7deee !important;
        opacity: 1 !important;
        font-size: 14px !important;
        font-weight: 700 !important;
        letter-spacing: 0.2px;
    }

    /* Catch-all for some Streamlit builds */
    [data-testid="stMetric"] label,
    [data-testid="stMetric"] small,
    [data-testid="stMetric"] p {
        color: #d7deee !important;
        opacity: 1 !important;
    }

    .subtle {
        color: #b9c2d8;
        font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
        font-size: 13px;
        opacity: 1;
    }

    .badge {
        display: inline-block;
        padding: 6px 10px;
        border-radius: 999px;
        border: 1px solid #2a2f3a;
        background: rgba(0,0,0,0.25);
        font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
        color: #d7deee;
        font-size: 12px;
        margin-right: 8px;
    }

    .badge-green { border-color: rgba(0,255,0,0.35); }
    .badge-amber { border-color: rgba(255,165,0,0.35); }
    .badge-red { border-color: rgba(255,0,0,0.35); }
    .badge-black { border-color: rgba(255,255,255,0.20); color: #f2f2f2; }

    .log-box {
        font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
        font-size: 13px;
        line-height: 1.45;
        color: #00ff00 !important;
        background-color: #000000;
        padding: 14px;
        border: 1px solid #333;
        border-radius: 6px;
        height: 350px;
        overflow-y: auto;
        white-space: pre;
        font-variant-ligatures: none;
        -webkit-font-smoothing: antialiased;
        text-rendering: geometricPrecision;
    }
    .log-box * {
        color: #00ff00 !important;
        opacity: 1 !important;
        font-variant-ligatures: none;
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


@st.cache_data(ttl=5)
def get_live_price() -> float:
    """
    Updates dynamically because we rerun every ~10s.
    Cached 5s to reduce API hits.
    """
    try:
        url = f"https://api.coinbase.com/v2/prices/{PRODUCT_ID}/spot"
        resp = requests.get(url, timeout=3)
        return float(resp.json()["data"]["amount"])
    except Exception:
        return 0.0


@st.cache_data(ttl=10)
def _load_market_data_cached(_mtime: float, path_str: str) -> pd.DataFrame:
    # cache key includes file mtime so the chart updates immediately when CSV changes
    csv_path = Path(path_str)
    if not csv_path.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(csv_path)
        if "Close" not in df.columns:
            df = pd.read_csv(
                csv_path,
                names=["Timestamp", "Open", "High", "Low", "Close", "Volume"],
                header=0,
            )
        df["Timestamp"] = pd.to_datetime(df["Timestamp"], utc=False, errors="coerce")
        df.dropna(subset=["Timestamp"], inplace=True)
        df.sort_values("Timestamp", inplace=True)
        return df
    except Exception:
        return pd.DataFrame()


def load_market_data() -> pd.DataFrame:
    csv_path = FLIGHT_RECORDER_PATH
    try:
        mtime = os.path.getmtime(str(csv_path)) if csv_path.exists() else 0.0
    except Exception:
        mtime = 0.0
    return _load_market_data_cached(mtime, str(csv_path))


def _safe_read_tail(path: Path, n_lines: int = 300) -> str:
    try:
        if not path.exists():
            return ""
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
        return "".join(lines[-n_lines:])
    except Exception:
        return ""


def read_logs() -> str:
    log_path = LOG_PATH
    txt = _safe_read_tail(log_path, n_lines=300)
    return txt if txt else "Waiting for logs..."


@st.cache_data(ttl=5)
def _load_json_cached(_mtime: float, path_str: str) -> dict | None:
    # cache key includes file mtime so cortex updates immediately when writer replaces file
    p = Path(path_str)
    try:
        if not p.exists():
            return None
        with open(p, "r", encoding="utf-8") as f:
            obj = json.load(f)
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def _safe_load_json(path: Path) -> dict | None:
    try:
        mtime = os.path.getmtime(str(path)) if path.exists() else 0.0
    except Exception:
        mtime = 0.0
    return _load_json_cached(mtime, str(path))


def _parse_iso(ts: str | None) -> datetime | None:
    if not ts:
        return None
    try:
        dt = datetime.fromisoformat(ts)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except Exception:
        return None


def _fmt_dt(dt: datetime | None) -> str:
    if not dt:
        return "—"
    try:
        return dt.astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
    except Exception:
        return dt.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def _human_td(seconds: float) -> str:
    if seconds <= 0:
        return "0m"
    m = int(seconds // 60)
    h = m // 60
    m = m % 60
    d = h // 24
    h = h % 24
    if d > 0:
        return f"{d}d {h}h {m}m"
    if h > 0:
        return f"{h}h {m}m"
    return f"{m}m"


# ---------------------------
# 🎯 CORTEX TELEMETRY
# ---------------------------


def get_cortex_state() -> dict:
    cortex_path = CORTEX_PATH

    state = {
        "timestamp_utc": None,
        "mode": None,
        # legacy
        "regime": "Searching...",
        "risk_mult": 0.0,
        "severity": 0.0,
        "raw_signal": "N/A",
        "wallet_verified": False,
        "cash_usd": None,
        "btc": None,
        "btc_notional_usd": None,
        "emergency_exit": False,
        # prime (optional)
        "p_long": None,
        "conf_min": None,
        "dd_band": None,
        "drawdown_frac": None,
        "planned_exit_ts": None,
        "last_decision": None,
        "equity_usd": None,
        "peak_equity_usd": None,
        # derived
        "conviction_score": 0,
    }

    payload = _safe_load_json(cortex_path)
    if not payload:
        return state

    try:
        state["timestamp_utc"] = payload.get("timestamp_utc")
        state["mode"] = payload.get("mode")

        state["regime"] = payload.get("regime", state["regime"])
        state["risk_mult"] = float(payload.get("risk_mult", state["risk_mult"]) or 0.0)
        state["severity"] = float(payload.get("severity", state["severity"]) or 0.0)
        state["raw_signal"] = payload.get("raw_signal", state["raw_signal"])
        state["wallet_verified"] = bool(payload.get("wallet_verified", False))

        if payload.get("mode") == "prime":
            state["p_long"] = payload.get("p_long")
            state["conf_min"] = payload.get("conf_min")
            state["dd_band"] = payload.get("dd_band")
            state["drawdown_frac"] = payload.get("drawdown_frac")
            state["planned_exit_ts"] = payload.get("planned_exit_ts")
            state["last_decision"] = payload.get("last_decision")
            state["equity_usd"] = payload.get("equity_usd")
            state["peak_equity_usd"] = payload.get("peak_equity_usd")

        if payload.get("mode") == "prime" and state["p_long"] is not None and state["conf_min"] is not None:
            try:
                margin = float(state["p_long"]) - float(state["conf_min"])
                score = max(0.0, min(100.0, (margin + 0.2) / 0.4 * 100.0))
                state["conviction_score"] = int(score)
            except Exception:
                state["conviction_score"] = 0
        else:
            state["conviction_score"] = int(max(0.0, min(1.0, state["risk_mult"])) * 100)

    except Exception:
        pass

    return state


# ---------------------------
# PRIME STATE (PAPER + LIVE)
# ---------------------------


def get_prime_state() -> dict | None:
    paper = PRIME_STATE_PAPER_PATH
    live = PRIME_STATE_LIVE_PATH

    dry = _env_bool("ARGUS_DRY_RUN", default=False) or _env_bool("PRIME_DRY_RUN", default=False)

    if dry and paper.exists():
        return _safe_load_json(paper)
    if live.exists():
        return _safe_load_json(live)
    if paper.exists():
        return _safe_load_json(paper)
    return None


def get_auto_entry_legacy() -> float:
    path = TRADE_STATE_PATH
    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                return float(json.load(f).get("entry_price", 0.0))
        except Exception:
            pass
    return 0.0


def _dd_badge(dd_band: str | None) -> tuple[str, str]:
    if not dd_band:
        return ("DD: —", "badge")
    b = str(dd_band).strip().lower()
    if b == "ok":
        return ("DD: OK", "badge badge-green")
    if b == "soft":
        return ("DD: SOFT", "badge badge-amber")
    if b == "hard":
        return ("DD: HARD", "badge badge-red")
    if b == "kill":
        return ("DD: KILL", "badge badge-black")
    return (f"DD: {dd_band}", "badge")


def _add_vline_shape(fig: go.Figure, x_dt: datetime, label: str, color: str) -> None:
    """
    DO NOT use fig.add_vline() at all (your Plotly build errors on it).
    Add a vertical line as a shape at x0=x1, and annotate with a normal annotation.
    """
    x_iso = x_dt.astimezone(timezone.utc).isoformat()

    fig.add_shape(
        type="line",
        xref="x",
        yref="paper",
        x0=x_iso,
        x1=x_iso,
        y0=0,
        y1=1,
        line=dict(color=color, width=2, dash="dash"),
    )
    fig.add_annotation(
        x=x_iso,
        y=1.02,
        xref="x",
        yref="paper",
        text=label,
        showarrow=False,
        font=dict(size=10, color="#d7deee"),
        bgcolor="rgba(0,0,0,0.35)",
        bordercolor="rgba(255,255,255,0.10)",
        borderwidth=1,
        align="center",
    )


# ---------------------------
# MAIN DASHBOARD
# ---------------------------


def main():
    st.title("🦅 ARGUS // LIVE COMMANDER")

    try:
        broker = get_broker()
        current_price = get_live_price()
        df = load_market_data()

        cash, btc = broker.get_wallet_snapshot()
        btc_exposure_usd = btc * current_price
        equity = cash + btc_exposure_usd

        cortex = get_cortex_state()
        prime_state = get_prime_state()

        if cortex.get("mode") == "prime":
            inferred_mode = "prime"
        elif prime_state is not None:
            inferred_mode = "prime"
        else:
            env_mode = (os.getenv("ARGUS_MODE") or "").strip().lower()
            inferred_mode = "prime" if env_mode == "prime" else "legacy"

        dry = _env_bool("ARGUS_DRY_RUN", default=False) or _env_bool("PRIME_DRY_RUN", default=False)

        legacy_entry = get_auto_entry_legacy()
        if btc > 0 and legacy_entry > 0:
            unrealized_pnl_usd = (current_price - legacy_entry) * btc
            pnl_pct = (unrealized_pnl_usd / (legacy_entry * btc)) * 100
        else:
            unrealized_pnl_usd, pnl_pct = 0.0, 0.0

    except Exception as e:
        st.error(f"System Error: {e}")
        return

    # Status badges
    mode_badge = "badge badge-green" if inferred_mode == "prime" else "badge"
    run_badge = "badge badge-amber" if dry else "badge badge-green"
    run_label = "DRY-RUN" if dry else "LIVE"
    st.markdown(
        f"<span class='{mode_badge}'>MODE: {inferred_mode.upper()}</span>"
        f"<span class='{run_badge}'>RUN: {run_label}</span>",
        unsafe_allow_html=True,
    )

    # Row 1
    st.markdown("### 🏦 Liquid Status")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Net Liquid Equity", f"${equity:,.2f}")
    k2.metric("Dry Powder (USD)", f"${cash:,.2f}")
    k3.metric("BTC Exposure", f"${btc_exposure_usd:,.2f}", f"{btc:.6f} BTC")
    k4.metric("Market Price", f"${current_price:,.2f}")

    # Row 2
    st.markdown("---")

    if inferred_mode == "prime":
        st.markdown("### 🧠 PRIME POSITION")

        ps = prime_state or {}

        in_position = bool(ps.get("in_position", False))
        entry_px = ps.get("entry_px")
        qty_btc = ps.get("qty_btc")
        entry_ts = _parse_iso(ps.get("entry_ts"))
        planned_exit_ts = _parse_iso(ps.get("planned_exit_ts"))

        p_long = cortex.get("p_long")
        conf_min = cortex.get("conf_min")
        dd_band = cortex.get("dd_band")
        dd_frac = cortex.get("drawdown_frac")
        last_decision = cortex.get("last_decision") or ps.get("last_decision")

        now = datetime.now(timezone.utc)
        time_remaining_s = (planned_exit_ts - now).total_seconds() if planned_exit_ts else None

        dd_label, dd_class = _dd_badge(dd_band)

        status_str = "LONG" if in_position else "FLAT"
        paper_str = " (PAPER)" if dry else ""
        st.markdown(
            f"<span class='badge badge-green'>STATUS: {status_str}{paper_str}</span>"
            f"<span class='{dd_class}'>{dd_label}</span>"
            f"<span class='badge'>LAST: {last_decision or '—'}</span>",
            unsafe_allow_html=True,
        )

        a, b, c, d = st.columns(4)

        a.metric("Entry Price", f"${float(entry_px):,.2f}" if entry_px else "—")
        a.metric("Entry Time", _fmt_dt(entry_ts))
        a.metric("Planned Exit", _fmt_dt(planned_exit_ts))

        b.metric("Time Remaining", _human_td(time_remaining_s) if time_remaining_s is not None else "—")
        if p_long is not None and conf_min is not None:
            try:
                b.metric("Confidence", f"{float(p_long):.3f}", f"min {float(conf_min):.2f}")
            except Exception:
                b.metric("Confidence", "—")
        else:
            b.metric("Confidence", "—")

        peak_eq = ps.get("peak_equity_usd")
        last_eq = ps.get("last_equity_usd")
        c.metric("Equity (Last)", f"${float(last_eq):,.2f}" if last_eq is not None else f"${equity:,.2f}")
        c.metric("Equity (Peak)", f"${float(peak_eq):,.2f}" if peak_eq is not None else "—")
        if dd_frac is not None:
            try:
                c.metric("Drawdown", f"{float(dd_frac) * 100:.2f}%")
            except Exception:
                c.metric("Drawdown", "—")
        else:
            c.metric("Drawdown", "—")

        if qty_btc and entry_px:
            try:
                notional = float(qty_btc) * float(entry_px)
                d.metric("Position Size", f"${notional:,.2f}")
            except Exception:
                d.metric("Position Size", "—")
        else:
            d.metric("Position Size", "—")
        d.metric("BTC Qty", f"{float(qty_btc):.8f}" if qty_btc else "—")

        st.markdown(
            "<div class='subtle'>PnL is de-emphasized in Prime (time-boxed exposure). Shown for sanity only.</div>",
            unsafe_allow_html=True,
        )
        p1, p2, p3 = st.columns(3)
        p1.metric("Spot PnL ($)", f"${unrealized_pnl_usd:+.2f}")
        p2.metric("Spot PnL (%)", f"{pnl_pct:+.2f}%")
        p3.metric(
            "Spot Price vs Entry",
            "—" if not entry_px else f"{(current_price / float(entry_px) - 1) * 100:+.2f}%",
        )

    else:
        st.markdown("### 📊 Active Position Analysis (Legacy)")
        avg_entry = legacy_entry if legacy_entry > 0 else 0.0
        breakeven = (avg_entry * 1.002) if (btc > 0 and avg_entry > 0) else 0.0

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Auto-Entry Price", f"${avg_entry:,.2f}" if avg_entry else "—")
        m2.metric("Unrealized P&L ($)", f"${unrealized_pnl_usd:+.2f}")
        color_mode = "normal" if pnl_pct >= 0 else "inverse"
        m3.metric("Unrealized P&L (%)", f"{pnl_pct:+.2f}%", delta=pnl_pct, delta_color=color_mode)
        m4.metric("Breakeven Price", f"${breakeven:,.2f}" if breakeven else "—")

    # Row 3
    st.markdown("---")
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
                    name=PRODUCT_ID,
                )
            )

            if inferred_mode == "prime" and prime_state is not None:
                entry_line = _parse_iso(prime_state.get("entry_ts"))
                exit_line = _parse_iso(prime_state.get("planned_exit_ts"))
                if entry_line:
                    _add_vline_shape(fig, entry_line, "ENTRY", "rgba(0,255,0,0.45)")
                if exit_line:
                    _add_vline_shape(fig, exit_line, "PLANNED EXIT", "rgba(255,165,0,0.45)")

            fig.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                height=350,
                margin=dict(l=10, r=10, t=10, b=10),
                showlegend=False,
            )
            st.plotly_chart(fig, width="stretch")
        else:
            st.markdown(f"<div class='subtle'>No market data found ({FLIGHT_RECORDER_PATH.name} missing).</div>", unsafe_allow_html=True)

    with c2:
        st.subheader("Risk / Cortex")

        if inferred_mode == "prime":
            fig_gauge = go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=int(cortex.get("conviction_score", 0) or 0),
                    title={"text": "Confidence Margin"},
                    gauge={
                        "axis": {"range": [0, 100]},
                        "bar": {"color": "darkblue"},
                        "steps": [
                            {"range": [0, 33], "color": "red"},
                            {"range": [33, 66], "color": "orange"},
                            {"range": [66, 100], "color": "green"},
                        ],
                    },
                )
            )
            fig_gauge.update_layout(
                height=260,
                paper_bgcolor="rgba(0,0,0,0)",
                font={"color": "white"},
                margin=dict(l=20, r=20, t=40, b=20),
            )
            st.plotly_chart(fig_gauge, width="stretch")

            p_long = cortex.get("p_long")
            conf_min = cortex.get("conf_min")
            dd_band = cortex.get("dd_band") or "—"
            dd_frac = cortex.get("drawdown_frac")
            planned_exit = cortex.get("planned_exit_ts") or (prime_state or {}).get("planned_exit_ts")

            line1 = f"<p style='text-align:center; color:#d7deee;'>DD Band: <b style='color:white'>{dd_band}</b></p>"
            if p_long is not None and conf_min is not None:
                line2 = (
                    f"<p style='text-align:center; color:#d7deee;'>"
                    f"p_long: <b style='color:white'>{float(p_long):.3f}</b> &nbsp;|&nbsp; "
                    f"conf_min: <b style='color:white'>{float(conf_min):.2f}</b></p>"
                )
            else:
                line2 = "<p style='text-align:center; color:#d7deee;'>p_long / conf_min: —</p>"

            if dd_frac is not None:
                try:
                    line3 = f"<p style='text-align:center; color:#d7deee;'>Drawdown: <b style='color:white'>{float(dd_frac)*100:.2f}%</b></p>"
                except Exception:
                    line3 = "<p style='text-align:center; color:#d7deee;'>Drawdown: —</p>"
            else:
                line3 = "<p style='text-align:center; color:#d7deee;'>Drawdown: —</p>"

            dt_exit = _parse_iso(planned_exit)
            if dt_exit:
                remaining = (dt_exit - datetime.now(timezone.utc)).total_seconds()
                line4 = f"<p style='text-align:center; color:#d7deee;'>Time to Exit: <b style='color:white'>{_human_td(remaining)}</b></p>"
            else:
                line4 = "<p style='text-align:center; color:#d7deee;'>Time to Exit: —</p>"

            st.markdown(line1 + line2 + line3 + line4, unsafe_allow_html=True)

        else:
            fig_gauge = go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=int(cortex.get("conviction_score", 0) or 0),
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
            st.plotly_chart(fig_gauge, width="stretch")

            regime_html = f"<p style='text-align:center; color:#d7deee;'>Regime: <b style='color:white'>{cortex.get('regime','—')}</b></p>"
            signal_html = f"<p style='text-align:center; color:#d7deee;'>Raw Signal: <b style='color:white'>{cortex.get('raw_signal','—')}</b></p>"
            sev_html = f"<p style='text-align:center; color:#d7deee;'>Severity: <b style='color:white'>{float(cortex.get('severity',0.0)):.2f}</b></p>"
            st.markdown(regime_html + signal_html + sev_html, unsafe_allow_html=True)

    # Row 4: Logs (escaped + <pre> to avoid weird glyph rendering)
    st.subheader("📜 System Logs")
    logs_raw = read_logs()
    logs_safe = html.escape(logs_raw)
    st.markdown(f"<div class='log-box'><pre style='margin:0'>{logs_safe}</pre></div>", unsafe_allow_html=True)

    time.sleep(10)
    st.rerun()


if __name__ == "__main__":
    main()
