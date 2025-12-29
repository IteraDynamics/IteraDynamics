# runtime/argus/apex_core/signal_generator.py
# 🦅 ARGUS LIVE PILOT - V3.5 (SMARTER DISCRETE DECISION LOGIC)

from __future__ import annotations

import sys
import os
import json
import joblib
import requests
import pandas as pd
import pandas_ta as ta

from pathlib import Path
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv

# ---------------------------
# Path / env resolution (runtime/argus as root)
# ---------------------------

current_file = Path(__file__).resolve()
project_root = current_file.parent.parent  # .../runtime/argus

if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


def _find_env_file(start: Path) -> Path | None:
    """
    Find the first .env walking up from `start` through parents.
    This avoids brittle assumptions about where .env lives (repo root vs runtime root).
    """
    for p in (start, *start.parents):
        candidate = p / ".env"
        if candidate.exists():
            return candidate
    return None


_env = _find_env_file(project_root)
if _env is not None:
    load_dotenv(_env)
else:
    # Still allow environment-only execution.
    load_dotenv()

# ---------------------------
# Runtime assets / artifacts
# ---------------------------

MODELS_DIR = project_root / "models"
MODEL_FILE = os.getenv("ARGUS_MODEL_FILE", "random_forest.pkl")

DATA_FILE = project_root / "flight_recorder.csv"
STATE_FILE = project_root / "trade_state.json"

CORTEX_FILE = project_root / "cortex.json"
CORTEX_TMP = project_root / "cortex.json.tmp"

PRODUCT_ID = "BTC-USD"

# Policy thresholds (env-overridable)
MIN_NOTIONAL_USD = float(os.getenv("ARGUS_MIN_NOTIONAL_USD", "5.0"))
MIN_HOLD_HOURS = float(os.getenv("ARGUS_MIN_HOLD_HOURS", "4.0"))

# Slippage-aware cushion via hurdle (default 0.35%).
PROFIT_HURDLE_PCT = float(os.getenv("ARGUS_PROFIT_HURDLE_PCT", "0.0035"))

# Emergency exit threshold (severity in [0, 1]).
EMERGENCY_SEVERITY_THRESHOLD = float(os.getenv("ARGUS_EMERGENCY_SEVERITY_THRESHOLD", "0.85"))

# --- NEW: discrete decision tuning knobs ---
# Hard stop-loss to cap downside on a single trade (e.g. 0.02 = -2%)
STOP_LOSS_PCT = float(os.getenv("ARGUS_STOP_LOSS_PCT", "0.02"))

# Hard max-hold to avoid zombie positions that never quite hit hurdle (in hours)
MAX_HOLD_HOURS = float(os.getenv("ARGUS_MAX_HOLD_HOURS", "72.0"))

# If severity exceeds this, we relax profit hurdle a bit to exit sooner
HURDLE_RELIEF_SEVERITY = float(os.getenv("ARGUS_HURDLE_RELIEF_SEVERITY", "0.60"))

# Factor to shrink hurdle when in bad regimes / high severity (e.g. 0.5 = cut in half)
HURDLE_RELIEF_FACTOR = float(os.getenv("ARGUS_HURDLE_RELIEF_FACTOR", "0.5"))

# ---------------------------
# Broker import
# ---------------------------

try:
    from src.real_broker import RealBroker
except ImportError as e:
    print(f"❌ CRITICAL IMPORT ERROR: {e}")
    sys.exit(1)

_broker = RealBroker()


def _utc_ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def _atomic_write_json(path: Path, tmp: Path, payload: dict) -> None:
    tmp.parent.mkdir(parents=True, exist_ok=True)
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, separators=(",", ":"), sort_keys=True)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def update_market_data():
    """Fetch latest hourly candles (UTC) and append only new rows."""
    try:
        url = "https://api.exchange.coinbase.com/products/BTC-USD/candles"
        resp = requests.get(url, params={"granularity": 3600}, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        if not isinstance(data, list):
            raise ValueError(f"Unexpected candles payload type: {type(data)}")

        data.sort(key=lambda x: x[0])  # ascending epoch seconds

        if DATA_FILE.exists():
            df_existing = pd.read_csv(DATA_FILE)
            if "Timestamp" not in df_existing.columns:
                raise ValueError("flight_recorder.csv missing Timestamp header (corrupt or headerless file).")
            last_ts = pd.to_datetime(df_existing["Timestamp"], utc=True, errors="coerce").max()
            if pd.isna(last_ts):
                last_ts = pd.Timestamp.min.tz_localize("UTC")
        else:
            pd.DataFrame(columns=["Timestamp", "Open", "High", "Low", "Close", "Volume"]).to_csv(
                DATA_FILE, index=False
            )
            last_ts = pd.Timestamp.min.tz_localize("UTC")

        new_rows = []
        for c in data:
            # Coinbase candles: [ time, low, high, open, close, volume ]
            ts = pd.to_datetime(c[0], unit="s", utc=True)
            if ts > last_ts:
                new_rows.append(
                    {
                        "Timestamp": ts.strftime("%Y-%m-%d %H:%M:%S"),
                        "Open": c[3],
                        "High": c[2],
                        "Low": c[1],
                        "Close": c[4],
                        "Volume": c[5],
                    }
                )

        if new_rows:
            pd.DataFrame(new_rows).to_csv(DATA_FILE, mode="a", header=False, index=False)
            print(f"   >> ✅ Data Updated. Newest(UTC): {new_rows[-1]['Timestamp']}")
    except Exception as e:
        print(f"   >> ⚠️ Data Update Glitch: {e}")


def detect_regime(df: pd.DataFrame):
    """
    Returns: (regime_label, risk_mult, severity, emergency_exit)
    - risk_mult governs BUY sizing (0 disables buys)
    - emergency_exit triggers SELL bypass path
    """
    sma_50 = ta.sma(df["Close"], length=50)
    sma_200 = ta.sma(df["Close"], length=200)
    atr = ta.atr(df["High"], df["Low"], df["Close"], length=14)
    vol = (atr / df["Close"]).astype(float)

    vol_t = vol.rolling(100).mean().iloc[-1]
    cp = float(df["Close"].iloc[-1])
    s50 = float(sma_50.iloc[-1]) if not pd.isna(sma_50.iloc[-1]) else cp
    s200_val = sma_200.iloc[-1]
    s200 = float(s200_val) if not pd.isna(s200_val) else s50

    label = "⚠️ UNKNOWN"
    risk_mult = 0.0
    severity = 0.0
    emergency_exit = False

    v_now = float(vol.iloc[-1]) if not pd.isna(vol.iloc[-1]) else 0.0
    v_t = float(vol_t) if not pd.isna(vol_t) and vol_t > 0 else max(v_now, 1e-9)

    def _clamp01(x: float) -> float:
        return max(0.0, min(1.0, x))

    below_200 = cp < s200
    below_50 = cp < s50

    vol_spike = _clamp01((v_now / v_t - 1.0) / 1.5)  # ~0 at normal, ~1 at +150% spike
    trend_bad = 1.0 if (below_200 and below_50) else (0.6 if below_200 else (0.2 if below_50 else 0.0))
    severity = _clamp01(0.65 * trend_bad + 0.35 * vol_spike)

    if cp > s200:
        if cp > s50:
            if v_now < v_t:
                label, risk_mult = "🐂 BULL QUIET", 0.90
            else:
                label, risk_mult = "🐎 BULL VOLATILE", 0.50
        else:
            label, risk_mult = "⚠️ PULLBACK (Warning)", 0.0
    else:
        if cp > s50:
            label, risk_mult = "🐯 RECOVERY", 0.25
        else:
            if (v_now / v_t) >= 2.0 or severity >= EMERGENCY_SEVERITY_THRESHOLD:
                label, risk_mult = "🩸 BEAR VOLATILE (Emergency)", 0.0
                emergency_exit = True
            else:
                label, risk_mult = "🐻 BEAR QUIET", 0.0

    if severity >= EMERGENCY_SEVERITY_THRESHOLD:
        emergency_exit = True

    return label, float(risk_mult), float(severity), bool(emergency_exit)


def _load_trade_state():
    """
    Returns (state_dict, status_str).
    status_str in {"OK", "MISSING", "CORRUPT"}.
    """
    if not STATE_FILE.exists():
        return None, "MISSING"
    try:
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            state = json.load(f)

        if not isinstance(state, dict):
            return None, "CORRUPT"
        if "entry_timestamp" not in state or "entry_price" not in state:
            return None, "CORRUPT"

        entry_time = datetime.fromisoformat(state["entry_timestamp"])
        if entry_time.tzinfo is None:
            # Backward compat for older naive timestamps: treat as UTC.
            entry_time = entry_time.replace(tzinfo=timezone.utc)

        float(state["entry_price"])
        state["_entry_time"] = entry_time
        return state, "OK"
    except Exception:
        return None, "CORRUPT"


def generate_signals():
    cycle_start = _utc_ts()

    update_market_data()

    # ---------------------------
    # Model + data load (BUY fail-closed)
    # ---------------------------
    try:
        model_path = MODELS_DIR / MODEL_FILE
        model = joblib.load(model_path)
        df = pd.read_csv(DATA_FILE)
    except Exception as e:
        print(f"   >> [DECISION] HOLD | Reason: MODEL_OR_DATA_LOAD_FAIL | Error: {e}")
        return

    if df.empty or len(df) < 210:
        print("   >> [DECISION] HOLD | Reason: INSUFFICIENT_HISTORY")
        return

    # ---------------------------
    # Feature Engineering (BUY fail-closed on NaNs)
    # ---------------------------
    try:
        df["RSI"] = ta.rsi(df["Close"], length=14)
        bband = ta.bbands(df["Close"], length=20, std=2)
        if bband is None or bband.shape[1] < 3:
            print("   >> [DECISION] HOLD | Reason: BBANDS_UNAVAILABLE")
            return

        df["BB_Pos"] = (df["Close"] - bband.iloc[:, 0]) / (bband.iloc[:, 2] - bband.iloc[:, 0])

        vol_mean = df["Volume"].rolling(20).mean()
        vol_std = df["Volume"].rolling(20).std()
        df["Vol_Z"] = (df["Volume"] - vol_mean) / vol_std.replace(0, pd.NA)

        feat_row = [df["RSI"].iloc[-1], df["BB_Pos"].iloc[-1], df["Vol_Z"].iloc[-1]]
        if any(pd.isna(x) for x in feat_row):
            print("   >> [DECISION] HOLD | Reason: FEATURES_NAN")
            return

        feat = pd.DataFrame([feat_row], columns=["RSI", "BB_Pos", "Vol_Z"])
        price = float(df["Close"].iloc[-1])
        if not (price > 0):
            print("   >> [DECISION] HOLD | Reason: BAD_PRICE")
            return
    except Exception as e:
        print(f"   >> [DECISION] HOLD | Reason: FEATURE_ENGINEERING_FAIL | Error: {e}")
        return

    # ---------------------------
    # Regime + signal
    # ---------------------------
    try:
        regime, risk_mult, severity, emergency_exit = detect_regime(df)
        raw_signal = "BUY" if int(model.predict(feat)[0]) == 1 else "SELL"
    except Exception as e:
        print(f"   >> [DECISION] HOLD | Reason: REGIME_OR_MODEL_PREDICT_FAIL | Error: {e}")
        return

    # ---------------------------
    # Wallet snapshot (BUY fail-closed, SELL fail-open only after verify)
    # ---------------------------
    wallet_verified = False
    cash = 0.0
    btc = 0.0
    wallet_err = None
    try:
        cash, btc = _broker.get_wallet_snapshot()
        wallet_verified = True
    except Exception as e:
        wallet_err = str(e)

    btc_notional = btc * price

    # Observability (always)
    print(f"   >> [BRAIN] Raw Signal: {raw_signal}")
    print(
        f"   >> [REGIME] {regime} | Risk Multiplier: {risk_mult:.2f} | Severity: {severity:.2f} | EmergencyExit: {str(emergency_exit)}"
    )

    if wallet_verified:
        print(f"   >> [WALLET] VERIFIED | Cash: ${cash:.2f} | BTC: {btc:.8f} | BTC Notional: ${btc_notional:.2f}")
    else:
        print(f"   >> [WALLET] UNVERIFIED | Error: {wallet_err}")

    # Write dashboard artifact atomically
    try:
        _atomic_write_json(
            CORTEX_FILE,
            CORTEX_TMP,
            {
                "timestamp_utc": cycle_start,
                "regime": regime,
                "risk_mult": float(risk_mult),
                "severity": float(severity),
                "raw_signal": raw_signal,
                "wallet_verified": bool(wallet_verified),
                "cash_usd": float(cash) if wallet_verified else None,
                "btc": float(btc) if wallet_verified else None,
                "btc_notional_usd": float(btc_notional) if wallet_verified else None,
                "emergency_exit": bool(emergency_exit),
                "model_file": str(MODEL_FILE),
                "model_path": str(model_path),
            },
        )
    except Exception as e:
        print(f"   >> ⚠️ Dashboard write glitch: {e}")

    # ---------------------------
    # Decision engine
    # ---------------------------

    # BUY path: fail-closed on any uncertainty
    if raw_signal == "BUY":
        if not wallet_verified:
            print("   >> [DECISION] HOLD | Reason: WALLET_UNVERIFIED_FAIL_CLOSED_BUY")
            return
        if risk_mult <= 0.0:
            print(f"   >> [DECISION] HOLD | Reason: RISK_MULT_ZERO | Regime: {regime}")
            return

        target_usd = cash * risk_mult
        if target_usd < MIN_NOTIONAL_USD:
            print(f"   >> [DECISION] HOLD | Reason: TARGET_BELOW_MIN_NOTIONAL | Target: ${target_usd:.2f}")
            return

        btc_qty = target_usd / price
        print(
            f"   >> [EXECUTION] ROUTE BUY | TargetUSD: ${target_usd:.2f} | QtyBTC: {btc_qty:.8f} | Price: ${price:.2f}"
        )

        _broker.execute_trade("BUY", btc_qty, price)
        return

    # SELL path: fail-open only after verifying live BTC balance
    if raw_signal == "SELL":
        if not wallet_verified:
            print("   >> [DECISION] HOLD | Reason: WALLET_UNVERIFIED_CANNOT_CONFIRM_POSITION")
            return

        if btc_notional < MIN_NOTIONAL_USD:
            print("   >> [DECISION] HOLD | Reason: NO_POSITION_OR_BELOW_MIN_NOTIONAL")
            return

        # Emergency exit bypass path (ignores guardrails)
        if emergency_exit:
            print("   >> [EXECUTION] ROUTE SELL | Reason: EMERGENCY_EXIT_BYPASS_GUARDRAILS")
            _broker.execute_trade("SELL", btc, price)
            return

        # Guardrail path (state-dependent)
        state, state_status = _load_trade_state()

        if state_status == "CORRUPT":
            print("   >> [EXECUTION] ROUTE SELL | Reason: STATE_CORRUPT_GUARDRAILS_SKIPPED_FAIL_OPEN_SELL")
            _broker.execute_trade("SELL", btc, price)
            return

        if state_status == "MISSING":
            print("   >> [EXECUTION] ROUTE SELL | Reason: STATE_MISSING_GUARDRAILS_SKIPPED_FAIL_OPEN_SELL")
            _broker.execute_trade("SELL", btc, price)
            return

        entry_time = state["_entry_time"]
        entry_price = float(state["entry_price"])

        now_utc = datetime.now(timezone.utc)
        hold_time = now_utc - entry_time
        hold_hours = hold_time.total_seconds() / 3600.0
        profit_pct = (price - entry_price) / entry_price

        # --- 1) Hard stop-loss: dominates all other guardrails ---
        if profit_pct <= -STOP_LOSS_PCT:
            print(
                f"   >> [EXECUTION] ROUTE SELL | Reason: STOP_LOSS_TRIGGERED | Loss: {profit_pct:.3%} | "
                f"Entry: ${entry_price:.2f} | Now: ${price:.2f}"
            )
            _broker.execute_trade("SELL", btc, price)
            return

        # --- 2) Max-hold failsafe to avoid zombie positions ---
        if hold_hours >= MAX_HOLD_HOURS:
            print(
                f"   >> [EXECUTION] ROUTE SELL | Reason: MAX_HOLD_EXCEEDED | Held: {hold_hours:.2f}h | "
                f"Entry: ${entry_price:.2f} | Now: ${price:.2f} | PnL: {profit_pct:.3%}"
            )
            _broker.execute_trade("SELL", btc, price)
            return

        # --- 3) Min-hold: only block sells before MIN_HOLD_HOURS unless stop-loss already triggered ---
        if hold_hours < MIN_HOLD_HOURS:
            print(
                f"   >> [DECISION] HOLD | Reason: MIN_HOLD_NOT_MET | Held: {hold_hours:.2f}h | "
                f"Min: {MIN_HOLD_HOURS:.2f}h"
            )
            return

        # --- 4) Dynamic profit hurdle after min-hold ---
        effective_hurdle = PROFIT_HURDLE_PCT

        bad_regime = ("BEAR" in regime) or ("RECOVERY" in regime)
        if severity >= HURDLE_RELIEF_SEVERITY or bad_regime:
            effective_hurdle = PROFIT_HURDLE_PCT * HURDLE_RELIEF_FACTOR

        if profit_pct < effective_hurdle:
            print(
                f"   >> [DECISION] HOLD | Reason: PROFIT_HURDLE_NOT_MET | Profit: {profit_pct:.3%} | "
                f"HurdleEff: {effective_hurdle:.3%} | BaseHurdle: {PROFIT_HURDLE_PCT:.3%}"
            )
            return

        print(
            f"   >> [EXECUTION] ROUTE SELL | Profit: {profit_pct:.3%} | "
            f"Held: {hold_hours:.2f}h | HurdleEff: {effective_hurdle:.3%} | "
            f"Entry: ${entry_price:.2f} | Now: ${price:.2f}"
        )
        _broker.execute_trade("SELL", btc, price)
        return

    print("   >> [DECISION] HOLD | Reason: UNKNOWN_SIGNAL_STATE")


if __name__ == "__main__":
    generate_signals()
