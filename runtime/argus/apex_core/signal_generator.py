# runtime/argus/apex_core/signal_generator.py
# 🦅 ARGUS SIGNAL GENERATOR — REFACTORED (Strategy Intent + SG Gates/Execution)
#
# Design:
#   - Strategy produces an "intent" (ENTER_LONG / EXIT_LONG / FLAT / HOLD)
#   - SG applies safety gates (wallet verification, min notional, drawdown governors, horizon, etc.)
#   - SG executes through RealBroker and writes state + cortex.json
#
# Modes:
#   ARGUS_MODE="prime"   -> PrimeModelStrategy (predict_proba + horizon + DD governors)
#   else                -> LegacyModelStrategy (predict + legacy regime risk_mult + sell guardrails)
#
# OPTIONAL external strategy override:
#   ARGUS_STRATEGY_MODULE="research.strategies.sg_stub_strategy"
#   ARGUS_STRATEGY_FUNC="generate_intent"
#
# Prime extra generic gates added:
#   - PRIME_REENTRY_COOLDOWN_H: blocks new entries for N hours after an exit (generic churn killer)
#   - PRIME_EXIT_MIN_HOLD_H   : blocks non-panic exits for first N hours after entry
#     (panic exits are allowed if strategy marks meta.exit_kind=="panic" or reason contains "vol_panic")
#
# IMPORTANT:
# - DRY-RUN:
#     - Prime uses paper_prime_state.json (never touches prime_state.json)
#     - Legacy still reads trade_state.json (if present) but will not execute real trades if broker is dry-run

from __future__ import annotations

import os
import sys
import json
import joblib
import requests
import importlib
import pandas as pd
import pandas_ta as ta

from dataclasses import dataclass
from pathlib import Path
from datetime import datetime, timezone, timedelta
from typing import Any, Callable
from dotenv import load_dotenv


# ---------------------------
# Path / env resolution
# ---------------------------

_CURRENT_FILE = Path(__file__).resolve()
_PROJECT_ROOT = _CURRENT_FILE.parent.parent  # runtime/argus

if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


def _find_env_file(start: Path) -> Path | None:
    for p in (start, *start.parents):
        candidate = p / ".env"
        if candidate.exists():
            return candidate
    return None


_env = _find_env_file(_PROJECT_ROOT)
if _env is not None:
    load_dotenv(_env, override=False)
else:
    load_dotenv(override=False)


# ---------------------------
# Broker import
# ---------------------------

try:
    from src.real_broker import RealBroker
except ImportError as e:
    print(f"❌ CRITICAL IMPORT ERROR: {e}")
    raise SystemExit(1)

_broker = RealBroker()


# ---------------------------
# Constants / Runtime assets
# ---------------------------

# Product and namespaced paths (configurable via ARGUS_PRODUCT_ID)
from config import (
    PRODUCT_ID,
    FLIGHT_RECORDER_PATH,
    TRADE_STATE_PATH,
    CORTEX_PATH,
    CORTEX_TMP_PATH,
    PRIME_STATE_LIVE_PATH,
    PRIME_STATE_LIVE_TMP_PATH,
    PRIME_STATE_PAPER_PATH,
    PRIME_STATE_PAPER_TMP_PATH,
)

MODELS_DIR = _PROJECT_ROOT / "models"
MODEL_FILE = os.getenv("ARGUS_MODEL_FILE", "random_forest.pkl")
DATA_FILE = FLIGHT_RECORDER_PATH
STATE_FILE = TRADE_STATE_PATH
CORTEX_FILE = CORTEX_PATH
CORTEX_TMP = CORTEX_TMP_PATH
PRIME_STATE_FILE_LIVE = PRIME_STATE_LIVE_PATH
PRIME_STATE_TMP_LIVE = PRIME_STATE_LIVE_TMP_PATH
PRIME_STATE_FILE_PAPER = PRIME_STATE_PAPER_PATH
PRIME_STATE_TMP_PAPER = PRIME_STATE_PAPER_TMP_PATH

ARGUS_MODE = os.getenv("ARGUS_MODE", "legacy").strip().lower()


# ---------------------------
# Helpers
# ---------------------------

_PROVENANCE_PRINTED = False


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _utc_ts_str() -> str:
    return _utc_now().strftime("%Y-%m-%d %H:%M:%S")


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def _safe_float(x, default: float = 0.0) -> float:
    try:
        v = float(x)
        if v != v:  # NaN
            return default
        return v
    except Exception:
        return default


def _parse_bool_env(name: str, default: bool = False) -> bool:
    v = os.getenv(name, "")
    if v == "":
        return default
    return v.strip().lower() in {"1", "true", "yes", "y", "on"}


def _is_dry_run() -> bool:
    return _parse_bool_env("ARGUS_DRY_RUN", False) or _parse_bool_env("PRIME_DRY_RUN", False)


def _atomic_write_json(path: Path, tmp: Path, payload: dict) -> None:
    tmp.parent.mkdir(parents=True, exist_ok=True)
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, separators=(",", ":"), sort_keys=True)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _load_external_strategy() -> Callable[..., Any] | None:
    mod = os.getenv("ARGUS_STRATEGY_MODULE", "").strip()
    fn = os.getenv("ARGUS_STRATEGY_FUNC", "").strip()
    if not mod or not fn:
        return None
    try:
        m = importlib.import_module(mod)
        f = getattr(m, fn)
        if not callable(f):
            raise TypeError("Strategy func is not callable")
        return f
    except Exception as e:
        print(f"   >> [STRATEGY] HOLD | Reason: STRATEGY_LOAD_FAIL | module={mod} func={fn} err={e}")
        return None


def _call_external_strategy(fn: Callable[..., Any], df: pd.DataFrame, ctx: "StrategyContext") -> Any:
    try:
        return fn(df=df, ctx=ctx)
    except TypeError:
        pass
    try:
        return fn(df, ctx)
    except TypeError:
        pass
    return fn(df=df, ctx=ctx, mode=ctx.mode, now_utc=ctx.now_utc)


# ---------------------------
# Market data update
# ---------------------------

def update_market_data() -> None:
    """Fetch latest hourly candles (UTC) and append only new rows."""
    try:
        url = f"https://api.exchange.coinbase.com/products/{PRODUCT_ID}/candles"
        resp = requests.get(url, params={"granularity": 3600}, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        if not isinstance(data, list):
            raise ValueError(f"Unexpected candles payload type: {type(data)}")

        data.sort(key=lambda x: x[0])

        if DATA_FILE.exists():
            df_existing = pd.read_csv(DATA_FILE)
            if "Timestamp" not in df_existing.columns:
                raise ValueError("flight_recorder.csv missing Timestamp header.")
            last_ts = pd.to_datetime(df_existing["Timestamp"], utc=True, errors="coerce").max()
            if pd.isna(last_ts):
                last_ts = pd.Timestamp.min.tz_localize("UTC")
        else:
            pd.DataFrame(columns=["Timestamp", "Open", "High", "Low", "Close", "Volume"]).to_csv(DATA_FILE, index=False)
            last_ts = pd.Timestamp.min.tz_localize("UTC")

        new_rows: list[dict] = []
        for c in data:
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


# ============================================================
# STRATEGY INTERFACE (Intent)
# ============================================================

class Action:
    ENTER_LONG = "ENTER_LONG"
    EXIT_LONG = "EXIT_LONG"
    FLAT = "FLAT"
    HOLD = "HOLD"


@dataclass
class StrategyContext:
    mode: str
    dry_run: bool
    model_file: str
    model_path: str
    now_utc: datetime


@dataclass
class StrategyIntent:
    action: str
    confidence: float | None = None
    desired_exposure_frac: float | None = None
    horizon_hours: int | None = None
    reason: str = "n/a"
    meta: dict | None = None


# ============================================================
# PRIME (state + governors)
# ============================================================

PRIME_CONF_MIN = _safe_float(os.getenv("PRIME_CONF_MIN", "0.64"), 0.64)
PRIME_HORIZON_H = int(_safe_float(os.getenv("PRIME_HORIZON", "48"), 48))
PRIME_MAX_EXPOSURE = _safe_float(os.getenv("PRIME_MAX_EXPOSURE", "0.25"), 0.25)

PRIME_DD_SOFT = _safe_float(os.getenv("PRIME_DD_SOFT", "0.03"), 0.03)
PRIME_DD_HARD = _safe_float(os.getenv("PRIME_DD_HARD", "0.06"), 0.06)
PRIME_DD_KILL = _safe_float(os.getenv("PRIME_DD_KILL", "0.10"), 0.10)
PRIME_DD_SOFT_MULT = _safe_float(os.getenv("PRIME_DD_SOFT_MULT", "0.5"), 0.5)

PRIME_MIN_NOTIONAL_USD = _safe_float(os.getenv("PRIME_MIN_NOTIONAL_USD", "5.00"), 5.00)

# Generic churn-control gates (Prime only)
PRIME_REENTRY_COOLDOWN_H = _safe_float(os.getenv("PRIME_REENTRY_COOLDOWN_H", "0"), 0.0)
PRIME_EXIT_MIN_HOLD_H = _safe_float(os.getenv("PRIME_EXIT_MIN_HOLD_H", "0"), 0.0)

PRIME_STATE_FILE_LIVE = _PROJECT_ROOT / "prime_state.json"
PRIME_STATE_TMP_LIVE = _PROJECT_ROOT / "prime_state.json.tmp"
PRIME_STATE_FILE_PAPER = _PROJECT_ROOT / "paper_prime_state.json"
PRIME_STATE_TMP_PAPER = _PROJECT_ROOT / "paper_prime_state.json.tmp"


def _prime_state_paths() -> tuple[Path, Path, str]:
    if _is_dry_run():
        return PRIME_STATE_FILE_PAPER, PRIME_STATE_TMP_PAPER, "paper"
    return PRIME_STATE_FILE_LIVE, PRIME_STATE_TMP_LIVE, "live"


def _load_prime_state() -> dict:
    state_path, _tmp, _mode = _prime_state_paths()
    if not state_path.exists():
        return {
            "killed": False,
            "peak_equity_usd": None,
            "last_equity_usd": None,
            "in_position": False,
            "entry_ts": None,
            "entry_px": None,
            "qty_btc": None,
            "planned_exit_ts": None,
            "last_exit_ts": None,   # <- added
            "last_decision": None,
        }
    try:
        with open(state_path, "r", encoding="utf-8") as f:
            st = json.load(f)
        if not isinstance(st, dict):
            raise ValueError("prime_state is not a dict")
        # forward-fill new keys
        st.setdefault("last_exit_ts", None)
        st.setdefault("last_decision", None)
        return st
    except Exception:
        return {"killed": True, "corrupt": True}


def _save_prime_state(st: dict) -> None:
    state_path, tmp_path, _mode = _prime_state_paths()
    _atomic_write_json(state_path, tmp_path, st)


def _dd_status(equity: float, peak: float) -> tuple[float, str, float]:
    if peak <= 0:
        return 0.0, "ok", 1.0

    dd = equity / peak - 1.0
    soft = -abs(PRIME_DD_SOFT)
    hard = -abs(PRIME_DD_HARD)
    kill = -abs(PRIME_DD_KILL)

    if dd <= kill:
        return dd, "kill", 0.0
    if dd <= hard:
        return dd, "hard", 0.0
    if dd <= soft:
        return dd, "soft", _clamp01(PRIME_DD_SOFT_MULT)
    return dd, "ok", 1.0


def _build_prime_features(df: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    df = df.copy()
    df["RSI"] = ta.rsi(df["Close"], length=14)

    bband = ta.bbands(df["Close"], length=20, std=2)
    if bband is None or bband.shape[1] < 3:
        raise RuntimeError("BBANDS_UNAVAILABLE")

    lower = bband.iloc[:, 0]
    upper = bband.iloc[:, 2]
    denom = (upper - lower).replace(0, pd.NA)
    df["BB_Pos"] = (df["Close"] - lower) / denom

    vol_mean = df["Volume"].rolling(20).mean()
    vol_std = df["Volume"].rolling(20).std().replace(0, pd.NA)
    df["Vol_Z"] = (df["Volume"] - vol_mean) / vol_std

    feat_row = [df["RSI"].iloc[-1], df["BB_Pos"].iloc[-1], df["Vol_Z"].iloc[-1]]
    if any(pd.isna(x) for x in feat_row):
        raise RuntimeError("FEATURES_NAN")

    feat = pd.DataFrame([feat_row], columns=["RSI", "BB_Pos", "Vol_Z"])
    price = float(df["Close"].iloc[-1])
    if not (price > 0):
        raise RuntimeError("BAD_PRICE")
    return feat, price


# ============================================================
# LEGACY (unchanged)
# ============================================================

MIN_NOTIONAL_USD = float(os.getenv("ARGUS_MIN_NOTIONAL_USD", "5.0"))
MIN_HOLD_HOURS = float(os.getenv("ARGUS_MIN_HOLD_HOURS", "4.0"))
PROFIT_HURDLE_PCT = float(os.getenv("ARGUS_PROFIT_HURDLE_PCT", "0.0035"))
EMERGENCY_SEVERITY_THRESHOLD = float(os.getenv("ARGUS_EMERGENCY_SEVERITY_THRESHOLD", "0.85"))
STOP_LOSS_PCT = float(os.getenv("ARGUS_STOP_LOSS_PCT", "0.02"))
MAX_HOLD_HOURS = float(os.getenv("ARGUS_MAX_HOLD_HOURS", "72.0"))
HURDLE_RELIEF_SEVERITY = float(os.getenv("ARGUS_HURDLE_RELIEF_SEVERITY", "0.60"))
HURDLE_RELIEF_FACTOR = float(os.getenv("ARGUS_HURDLE_RELIEF_FACTOR", "0.5"))


def detect_regime_legacy(df: pd.DataFrame):
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

    below_200 = cp < s200
    below_50 = cp < s50

    vol_spike = _clamp01((v_now / v_t - 1.0) / 1.5)
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


def _load_trade_state_legacy():
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
            entry_time = entry_time.replace(tzinfo=timezone.utc)

        float(state["entry_price"])
        state["_entry_time"] = entry_time
        return state, "OK"
    except Exception:
        return None, "CORRUPT"


# ============================================================
# STRATEGIES (produce intents)
# ============================================================

class PrimeModelStrategy:
    def __init__(self, model):
        self.model = model

    def get_intent(self, df: pd.DataFrame, ctx: StrategyContext) -> tuple[StrategyIntent, float]:
        feat, price = _build_prime_features(df)

        if not hasattr(self.model, "predict_proba"):
            return StrategyIntent(action=Action.HOLD, reason="MODEL_NO_PREDICT_PROBA"), price

        p_long = float(self.model.predict_proba(feat)[0][1])
        want_long = p_long >= PRIME_CONF_MIN

        if want_long:
            return StrategyIntent(
                action=Action.ENTER_LONG,
                confidence=p_long,
                desired_exposure_frac=PRIME_MAX_EXPOSURE,
                horizon_hours=PRIME_HORIZON_H,
                reason=f"p_long>=conf ({p_long:.3f}>={PRIME_CONF_MIN:.2f})",
                meta={"p_long": p_long, "conf_min": PRIME_CONF_MIN},
            ), price

        return StrategyIntent(
            action=Action.FLAT,
            confidence=p_long,
            reason=f"p_long<conf ({p_long:.3f}<{PRIME_CONF_MIN:.2f})",
            meta={"p_long": p_long, "conf_min": PRIME_CONF_MIN},
        ), price


class LegacyModelStrategy:
    def __init__(self, model):
        self.model = model

    def get_intent(self, df: pd.DataFrame, ctx: StrategyContext) -> tuple[StrategyIntent, float, dict]:
        df2 = df.copy()
        df2["RSI"] = ta.rsi(df2["Close"], length=14)
        bband = ta.bbands(df2["Close"], length=20, std=2)
        if bband is None or bband.shape[1] < 3:
            price = float(df2["Close"].iloc[-1])
            return StrategyIntent(action=Action.HOLD, reason="BBANDS_UNAVAILABLE"), price, {}

        lower = bband.iloc[:, 0]
        upper = bband.iloc[:, 2]
        denom = (upper - lower).replace(0, pd.NA)
        df2["BB_Pos"] = (df2["Close"] - lower) / denom

        vol_mean = df2["Volume"].rolling(20).mean()
        vol_std = df2["Volume"].rolling(20).std().replace(0, pd.NA)
        df2["Vol_Z"] = (df2["Volume"] - vol_mean) / vol_std

        feat_row = [df2["RSI"].iloc[-1], df2["BB_Pos"].iloc[-1], df2["Vol_Z"].iloc[-1]]
        if any(pd.isna(x) for x in feat_row):
            price = float(df2["Close"].iloc[-1])
            return StrategyIntent(action=Action.HOLD, reason="FEATURES_NAN"), price, {}

        feat = pd.DataFrame([feat_row], columns=["RSI", "BB_Pos", "Vol_Z"])
        price = float(df2["Close"].iloc[-1])
        if not (price > 0):
            return StrategyIntent(action=Action.HOLD, reason="BAD_PRICE"), 0.0, {}

        regime, risk_mult, severity, emergency_exit = detect_regime_legacy(df2)
        raw_signal = "BUY" if int(self.model.predict(feat)[0]) == 1 else "SELL"

        meta = {
            "regime": regime,
            "risk_mult": float(risk_mult),
            "severity": float(severity),
            "emergency_exit": bool(emergency_exit),
            "raw_signal": raw_signal,
        }

        if raw_signal == "BUY":
            return StrategyIntent(
                action=Action.ENTER_LONG,
                confidence=None,
                desired_exposure_frac=_clamp01(float(risk_mult)),
                reason=f"LEGACY_BUY ({regime})",
                meta=meta,
            ), price, meta

        return StrategyIntent(
            action=Action.EXIT_LONG,
            confidence=None,
            reason=f"LEGACY_SELL ({regime})",
            meta=meta,
        ), price, meta


# ============================================================
# SG EXECUTION LAYER
# ============================================================

def _load_model() -> tuple[object | None, str]:
    try:
        model_path = MODELS_DIR / MODEL_FILE
        model = joblib.load(model_path)
        return model, str(model_path)
    except Exception as e:
        return None, f"{e}"


def _load_data() -> pd.DataFrame | None:
    try:
        df = pd.read_csv(DATA_FILE)
        return df
    except Exception:
        return None


def _wallet_snapshot(price: float) -> tuple[bool, float, float, float, str | None]:
    try:
        cash, btc = _broker.get_wallet_snapshot()
        equity = float(cash) + float(btc) * float(price)
        return True, float(cash), float(btc), float(equity), None
    except Exception as e:
        return False, 0.0, 0.0, 0.0, str(e)


def _write_cortex(payload: dict) -> None:
    try:
        _atomic_write_json(CORTEX_FILE, CORTEX_TMP, payload)
    except Exception:
        pass


def _debug_asserts_enabled() -> bool:
    return _parse_bool_env("ARGUS_DEBUG_ASSERTS", False)


def _build_cortex_payload(
    *,
    cycle_start: str,
    prime_state: dict,
    execution_branch: str,
    intent: "StrategyIntent",
    price: float,
    wallet_verified: bool,
    cash: float,
    btc: float,
    equity: float,
    peak: float,
    dd: float,
    dd_band: str,
    model_path_str: str,
    state_mode: str,
    state_path: "Path",
    dry: bool,
    wallet_err: str | None = None,
) -> dict:
    """
    Assemble cortex.json payload deterministically from authoritative sources:
      - prime_state: freshly loaded from disk (authoritative for position state)
      - wallet snapshot: authoritative for balances
      - execution_branch: authoritative for last_decision (what actually happened THIS cycle)
      - intent: current cycle's strategy intent
    
    HARD RULES:
      - in_position is derived from prime_state
      - planned_exit_ts is only included if in_position == True
      - last_decision is the execution_branch from THIS cycle
      - btc/cash/equity are from current wallet snapshot
    """
    in_position = bool(prime_state.get("in_position", False))
    
    # planned_exit_ts: only meaningful if in_position
    if in_position:
        planned_exit_ts = prime_state.get("planned_exit_ts")
    else:
        planned_exit_ts = None
    
    # DEBUG assertions (only fire if ARGUS_DEBUG_ASSERTS=1)
    if _debug_asserts_enabled():
        # If not in position, planned_exit_ts in state must be None
        if not prime_state.get("in_position", False):
            assert prime_state.get("planned_exit_ts") is None, (
                f"Invariant violation: in_position=False but planned_exit_ts={prime_state.get('planned_exit_ts')}"
            )
        # If we're writing a non-null planned_exit_ts, in_position must be True
        if planned_exit_ts is not None:
            assert in_position is True, (
                f"Invariant violation: planned_exit_ts={planned_exit_ts} but in_position={in_position}"
            )
    
    p_long = (intent.meta or {}).get("p_long", None)
    btc_notional = btc * price if wallet_verified else None
    
    _ext_mod = os.getenv("ARGUS_STRATEGY_MODULE", "").strip()
    _ext_fn = os.getenv("ARGUS_STRATEGY_FUNC", "").strip()
    _is_external = bool(_ext_mod and _ext_fn)
    
    return {
        "timestamp_utc": cycle_start,
        "mode": "prime",
        "in_position": in_position,
        "intent_action": intent.action,
        "intent_reason": intent.reason,
        "execution_branch": execution_branch,
        "last_decision": execution_branch,  # authoritative: what THIS cycle did
        "p_long": p_long,
        "conf_min": PRIME_CONF_MIN,
        "horizon_h": intent.horizon_hours or PRIME_HORIZON_H,
        "max_exposure": PRIME_MAX_EXPOSURE,
        "equity_usd": float(equity) if wallet_verified else None,
        "cash_usd": float(cash) if wallet_verified else None,
        "btc": float(btc) if wallet_verified else None,
        "btc_notional_usd": float(btc_notional) if btc_notional is not None else None,
        "peak_equity_usd": float(peak),
        "drawdown_frac": float(dd),
        "dd_band": dd_band,
        "planned_exit_ts": planned_exit_ts,  # None if not in_position
        "entry_ts": prime_state.get("entry_ts") if in_position else None,
        "entry_px": prime_state.get("entry_px") if in_position else None,
        "model_file": str(MODEL_FILE),
        "model_path": str(model_path_str),
        "dry_run": bool(dry),
        "prime_state_mode": state_mode,
        "prime_state_file": str(state_path),
        "wallet_verified": bool(wallet_verified),
        "wallet_err": wallet_err,
        "prime_reentry_cooldown_h": float(PRIME_REENTRY_COOLDOWN_H),
        "prime_exit_min_hold_h": float(PRIME_EXIT_MIN_HOLD_H),
        "external_strategy": _is_external,
        "external_strategy_module": _ext_mod if _is_external else None,
        "external_strategy_func": _ext_fn if _is_external else None,
        "intent_source": "external" if _is_external else "internal",
    }


def _execute_buy(target_usd: float, price: float) -> float:
    qty = target_usd / price
    _broker.execute_trade("BUY", qty, price)
    return qty


def _execute_sell(qty_btc: float, price: float) -> None:
    _broker.execute_trade("SELL", qty_btc, price)


def _is_panic_exit(intent: StrategyIntent) -> bool:
    r = (intent.reason or "").lower()
    if "vol_panic" in r or "panic" in r:
        return True
    meta = intent.meta or {}
    if isinstance(meta, dict) and meta.get("exit_kind") == "panic":
        return True
    return False


def _run_prime_engine(df: pd.DataFrame, model, model_path_str: str) -> None:
    cycle_start = _utc_ts_str()
    dry = _is_dry_run()
    now = _utc_now()

    # FRESH state load from disk — authoritative for position state
    st = _load_prime_state()
    state_path, _tmp, state_mode = _prime_state_paths()

    ctx = StrategyContext(
        mode="prime",
        dry_run=dry,
        model_file=MODEL_FILE,
        model_path=model_path_str,
        now_utc=now,
    )

    # Helper to write cortex with current state
    def write_cortex_for_branch(execution_branch: str) -> None:
        cortex_payload = _build_cortex_payload(
            cycle_start=cycle_start,
            prime_state=st,
            execution_branch=execution_branch,
            intent=intent,
            price=price,
            wallet_verified=wallet_verified,
            cash=cash,
            btc=btc,
            equity=current_equity,
            peak=peak,
            dd=dd,
            dd_band=dd_band,
            model_path_str=model_path_str,
            state_mode=state_mode,
            state_path=state_path,
            dry=dry,
            wallet_err=wallet_err,
        )
        _write_cortex(cortex_payload)

    # Initialize defaults for early-exit branches
    price = float(df["Close"].iloc[-1]) if len(df) > 0 else 0.0
    intent = StrategyIntent(action=Action.HOLD, reason="INIT")
    wallet_verified, cash, btc, equity, wallet_err = _wallet_snapshot(price=price)
    btc_notional = btc * price
    current_equity = equity if wallet_verified else _safe_float(st.get("last_equity_usd"), 0.0)
    peak = _safe_float(st.get("peak_equity_usd"), current_equity)
    if peak <= 0:
        peak = current_equity
    dd, dd_band, dd_expo_mult = _dd_status(equity=current_equity, peak=peak)

    try:
        external = _load_external_strategy()
        if external:
            out = _call_external_strategy(external, df=df, ctx=ctx)
            if isinstance(out, StrategyIntent):
                intent = out
            elif isinstance(out, dict):
                intent = StrategyIntent(**out)
            else:
                raise TypeError(f"Unsupported external strategy return type: {type(out)}")
            price = float(df["Close"].iloc[-1])
        else:
            strat = PrimeModelStrategy(model)
            intent, price = strat.get_intent(df, ctx)

        p_long = (intent.meta or {}).get("p_long", None)
    except Exception as e:
        print(f"   >> [PRIME] HOLD | Reason: STRATEGY_INTENT_FAIL | Error: {e}")
        st["last_decision"] = f"HOLD_STRATEGY_FAIL err={e}"
        _save_prime_state(st)
        write_cortex_for_branch(f"HOLD_STRATEGY_FAIL err={e}")
        return

    # Re-fetch wallet with updated price (in case price changed)
    wallet_verified, cash, btc, equity, wallet_err = _wallet_snapshot(price=price)
    btc_notional = btc * price
    current_equity = equity if wallet_verified else _safe_float(st.get("last_equity_usd"), 0.0)
    if current_equity > peak:
        peak = current_equity

    dd, dd_band, dd_expo_mult = _dd_status(equity=current_equity, peak=peak)

    if st.get("corrupt") is True:
        print("   >> [PRIME] HOLD | Reason: PRIME_STATE_CORRUPT_FAILSAFE (treated as killed)")
        st["last_decision"] = "HOLD_STATE_CORRUPT"
        write_cortex_for_branch("HOLD_STATE_CORRUPT")
        return

    if st.get("killed") is True:
        print("   >> [PRIME] KILLED STATE ACTIVE -> no entries until reset state file")
        if wallet_verified and btc_notional >= PRIME_MIN_NOTIONAL_USD and btc > 0:
            print("   >> [PRIME] KILLED -> LIQUIDATING REMAINING BTC (best-effort)")
            _execute_sell(btc, price)
        st["last_decision"] = "KILLED_STATE_ACTIVE"
        write_cortex_for_branch("KILLED_STATE_ACTIVE")
        return

    # In-position authority
    if dry:
        paper_qty = _safe_float(st.get("qty_btc"), 0.0)
        paper_in_pos = bool(st.get("in_position", False))
        paper_notional = paper_qty * price
        in_pos = bool(paper_in_pos and (paper_notional >= PRIME_MIN_NOTIONAL_USD) and (paper_qty > 0))
        qty_for_exit = paper_qty
    else:
        in_pos = btc_notional >= PRIME_MIN_NOTIONAL_USD
        qty_for_exit = btc

    # Parse entry_ts / planned_exit_ts / last_exit_ts
    entry_ts = None
    if st.get("entry_ts"):
        try:
            entry_ts = datetime.fromisoformat(st["entry_ts"])
            if entry_ts.tzinfo is None:
                entry_ts = entry_ts.replace(tzinfo=timezone.utc)
        except Exception:
            entry_ts = None

    planned_exit_ts = None
    if st.get("planned_exit_ts"):
        try:
            planned_exit_ts = datetime.fromisoformat(st["planned_exit_ts"])
            if planned_exit_ts.tzinfo is None:
                planned_exit_ts = planned_exit_ts.replace(tzinfo=timezone.utc)
        except Exception:
            planned_exit_ts = None

    last_exit_ts = None
    if st.get("last_exit_ts"):
        try:
            last_exit_ts = datetime.fromisoformat(st["last_exit_ts"])
            if last_exit_ts.tzinfo is None:
                last_exit_ts = last_exit_ts.replace(tzinfo=timezone.utc)
        except Exception:
            last_exit_ts = None

    # Kill rule
    if dd_band == "kill":
        if wallet_verified and in_pos and qty_for_exit > 0:
            print(f"   >> [PRIME] DD_KILL TRIGGERED (dd={dd:.3%}) -> FORCED LIQUIDATION")
            _execute_sell(qty_for_exit, price)
        execution_branch = f"KILL_DD dd={dd:.6f}"
        st["killed"] = True
        st["in_position"] = False
        st["planned_exit_ts"] = None
        st["last_decision"] = execution_branch
        st["peak_equity_usd"] = peak
        st["last_equity_usd"] = current_equity
        _save_prime_state(st)
        write_cortex_for_branch(execution_branch)
        return

    # ----------------------------
    # IN POSITION: exits + horizon + min-hold gate
    # ----------------------------
    if in_pos:
        # Horizon schedule
        if planned_exit_ts is None:
            horizon_h = intent.horizon_hours or PRIME_HORIZON_H
            planned_exit_ts = now + timedelta(hours=int(horizon_h))
            st["planned_exit_ts"] = planned_exit_ts.isoformat()

        # Strategy exit gate (with min-hold)
        if intent.action == Action.EXIT_LONG and wallet_verified:
            hold_hours = None
            if entry_ts is not None:
                hold_hours = (now - entry_ts).total_seconds() / 3600.0

            panic = _is_panic_exit(intent)
            if (not panic) and PRIME_EXIT_MIN_HOLD_H > 0 and hold_hours is not None and hold_hours < PRIME_EXIT_MIN_HOLD_H:
                # Block early non-panic exits
                execution_branch = f"MIN_HOLD_BLOCK exit intent={intent.reason} held={hold_hours:.2f}h min={PRIME_EXIT_MIN_HOLD_H:.2f}h"
                st["in_position"] = True
                st["peak_equity_usd"] = peak
                st["last_equity_usd"] = current_equity
                st["last_decision"] = execution_branch
                _save_prime_state(st)
                print(f"   >> [PRIME] HOLD (MIN_HOLD_BLOCK) | held={hold_hours:.2f}h < min={PRIME_EXIT_MIN_HOLD_H:.2f}h | reason={intent.reason}")
                write_cortex_for_branch(execution_branch)
            else:
                # Execute strategy exit
                execution_branch = f"EXIT_STRATEGY reason={intent.reason}"
                print(f"   >> [PRIME] EXIT: STRATEGY_EXIT -> SELL ALL | reason={intent.reason} p={p_long} conf={PRIME_CONF_MIN:.2f}")
                if qty_for_exit > 0:
                    _execute_sell(qty_for_exit, price)

                st.update(
                    {
                        "in_position": False,
                        "entry_ts": None,
                        "entry_px": None,
                        "qty_btc": None,
                        "planned_exit_ts": None,
                        "last_exit_ts": now.isoformat(),
                        "last_decision": execution_branch,
                        "peak_equity_usd": peak,
                        "last_equity_usd": current_equity,
                    }
                )
                _save_prime_state(st)
                write_cortex_for_branch(execution_branch)
            return

        # Horizon exit
        if wallet_verified and planned_exit_ts is not None and now >= planned_exit_ts:
            execution_branch = "EXIT_HORIZON"
            print(f"   >> [PRIME] EXIT: HORIZON_REACHED -> SELL ALL | p={p_long} conf={PRIME_CONF_MIN:.2f}")
            if qty_for_exit > 0:
                _execute_sell(qty_for_exit, price)
            st.update(
                {
                    "in_position": False,
                    "entry_ts": None,
                    "entry_px": None,
                    "qty_btc": None,
                    "planned_exit_ts": None,
                    "last_exit_ts": now.isoformat(),
                    "last_decision": execution_branch,
                    "peak_equity_usd": peak,
                    "last_equity_usd": current_equity,
                }
            )
            _save_prime_state(st)
            write_cortex_for_branch(execution_branch)
            return

        # HOLD
        execution_branch = f"HOLD_IN_POSITION exec=HOLD intent={intent.action} p={p_long}"
        st["in_position"] = True
        st["peak_equity_usd"] = peak
        st["last_equity_usd"] = current_equity
        st["last_decision"] = execution_branch
        _save_prime_state(st)

        print(
            f"   >> [PRIME] HOLD (IN POSITION) | exec=HOLD | "
            f"intent={intent.action} | equity=${current_equity:.2f} dd={dd:.3%} band={dd_band} | "
            f"exit_at={planned_exit_ts.isoformat() if planned_exit_ts else 'n/a'}"
        )

        write_cortex_for_branch(execution_branch)
        return

    # ----------------------------
    # NOT IN POSITION: entry gates
    # ----------------------------
    if dd_band == "hard":
        execution_branch = f"HARD_DD_BLOCK dd={dd:.6f}"
        print(f"   >> [PRIME] HARD_DD (dd={dd:.3%}) -> entries blocked")
        st["in_position"] = False
        st["planned_exit_ts"] = None
        st["peak_equity_usd"] = peak
        st["last_equity_usd"] = current_equity
        st["last_decision"] = execution_branch
        _save_prime_state(st)
        write_cortex_for_branch(execution_branch)
        return

    # If strategy isn't asking to enter, remain flat
    if intent.action not in (Action.ENTER_LONG,):
        execution_branch = f"FLAT_NO_ENTRY intent={intent.action}"
        st["in_position"] = False
        st["planned_exit_ts"] = None
        st["peak_equity_usd"] = peak
        st["last_equity_usd"] = current_equity
        st["last_decision"] = execution_branch
        _save_prime_state(st)
        print(f"   >> [PRIME] FLAT | intent={intent.action} | reason={intent.reason}")
        write_cortex_for_branch(execution_branch)
        return

    # Cooldown gate (generic)
    if PRIME_REENTRY_COOLDOWN_H > 0 and last_exit_ts is not None:
        since_exit_h = (now - last_exit_ts).total_seconds() / 3600.0
        if since_exit_h < PRIME_REENTRY_COOLDOWN_H:
            execution_branch = f"REENTRY_COOLDOWN_BLOCK since_exit={since_exit_h:.2f}h < {PRIME_REENTRY_COOLDOWN_H:.2f}h"
            st["in_position"] = False
            st["planned_exit_ts"] = None
            st["peak_equity_usd"] = peak
            st["last_equity_usd"] = current_equity
            st["last_decision"] = execution_branch
            _save_prime_state(st)
            print(f"   >> [PRIME] HOLD | Reason: REENTRY_COOLDOWN_BLOCK since_exit={since_exit_h:.2f}h < {PRIME_REENTRY_COOLDOWN_H:.2f}h")
            write_cortex_for_branch(execution_branch)
            return

    if not wallet_verified:
        execution_branch = f"WALLET_UNVERIFIED_FAIL_CLOSED_ENTRY err={wallet_err}"
        print(f"   >> [PRIME] HOLD | Reason: WALLET_UNVERIFIED_FAIL_CLOSED_ENTRY | err={wallet_err}")
        st["last_decision"] = execution_branch
        _save_prime_state(st)
        write_cortex_for_branch(execution_branch)
        return

    effective_max_exposure = PRIME_MAX_EXPOSURE * dd_expo_mult
    if effective_max_exposure <= 0:
        execution_branch = f"BLOCKED_EXPO band={dd_band}"
        print(f"   >> [PRIME] HOLD | Reason: EXPOSURE_ZERO (dd_band={dd_band})")
        st["in_position"] = False
        st["planned_exit_ts"] = None
        st["peak_equity_usd"] = peak
        st["last_equity_usd"] = current_equity
        st["last_decision"] = execution_branch
        _save_prime_state(st)
        write_cortex_for_branch(execution_branch)
        return

    target_usd = equity * effective_max_exposure
    target_usd = min(target_usd, cash)

    if target_usd < PRIME_MIN_NOTIONAL_USD:
        execution_branch = "TARGET_TOO_SMALL"
        print(f"   >> [PRIME] HOLD | Reason: TARGET_BELOW_MIN_NOTIONAL | target=${target_usd:.2f}")
        st["in_position"] = False
        st["planned_exit_ts"] = None
        st["peak_equity_usd"] = peak
        st["last_equity_usd"] = current_equity
        st["last_decision"] = execution_branch
        _save_prime_state(st)
        write_cortex_for_branch(execution_branch)
        return

    horizon_h = int(intent.horizon_hours or PRIME_HORIZON_H)
    new_planned_exit_ts = now + timedelta(hours=horizon_h)

    print(
        f"   >> [PRIME] ENTER LONG | p={p_long} conf={PRIME_CONF_MIN:.2f} | "
        f"intent={intent.action} | equity=${equity:.2f} cash=${cash:.2f} dd={dd:.3%} band={dd_band} | "
        f"effExpo={effective_max_exposure:.3f} target=${target_usd:.2f} px=${price:.2f} | "
        f"exit_at={new_planned_exit_ts.isoformat()}"
    )

    qty = _execute_buy(target_usd, price)

    execution_branch = f"ENTER_LONG exec=BUY p={p_long}"
    st.update(
        {
            "killed": False,
            "peak_equity_usd": peak,
            "last_equity_usd": equity,
            "in_position": True,
            "entry_ts": now.isoformat(),
            "entry_px": price,
            "qty_btc": qty,
            "planned_exit_ts": new_planned_exit_ts.isoformat(),
            "last_decision": execution_branch,
        }
    )
    _save_prime_state(st)

    write_cortex_for_branch(execution_branch)


def _run_legacy_engine(df: pd.DataFrame, model, model_path_str: str) -> None:
    cycle_start = _utc_ts_str()
    dry = _is_dry_run()
    now = _utc_now()

    ctx = StrategyContext(
        mode="legacy",
        dry_run=dry,
        model_file=MODEL_FILE,
        model_path=model_path_str,
        now_utc=now,
    )

    try:
        external = _load_external_strategy()
        if external:
            out = _call_external_strategy(external, df=df, ctx=ctx)
            if isinstance(out, StrategyIntent):
                intent = out
                meta = intent.meta or {}
            elif isinstance(out, dict):
                intent = StrategyIntent(**out)
                meta = intent.meta or {}
            else:
                raise TypeError(f"Unsupported external strategy return type: {type(out)}")
            price = float(df["Close"].iloc[-1])
        else:
            strat = LegacyModelStrategy(model)
            intent, price, meta = strat.get_intent(df, ctx)
    except Exception as e:
        print(f"   >> [LEGACY] HOLD | Reason: STRATEGY_INTENT_FAIL | Error: {e}")
        return

    wallet_verified, cash, btc, equity, wallet_err = _wallet_snapshot(price=price)
    btc_notional = btc * price

    regime = (meta or {}).get("regime", "UNKNOWN")
    risk_mult = float((meta or {}).get("risk_mult", 0.0) or 0.0)
    severity = float((meta or {}).get("severity", 0.0) or 0.0)
    emergency_exit = bool((meta or {}).get("emergency_exit", False))
    raw_signal = (meta or {}).get("raw_signal", None)

    _ext_mod = os.getenv("ARGUS_STRATEGY_MODULE", "").strip()
    _ext_fn = os.getenv("ARGUS_STRATEGY_FUNC", "").strip()
    _is_external = bool(_ext_mod and _ext_fn)
    _write_cortex(
        {
            "timestamp_utc": cycle_start,
            "mode": "legacy",
            "intent_action": intent.action,
            "intent_reason": intent.reason,
            "raw_signal": raw_signal,
            "regime": regime,
            "risk_mult": float(risk_mult),
            "severity": float(severity),
            "emergency_exit": bool(emergency_exit),
            "wallet_verified": bool(wallet_verified),
            "wallet_err": wallet_err,
            "cash_usd": float(cash) if wallet_verified else None,
            "btc": float(btc) if wallet_verified else None,
            "btc_notional_usd": float(btc_notional) if wallet_verified else None,
            "model_file": str(MODEL_FILE),
            "model_path": str(model_path_str),
            "dry_run": bool(dry),
            "external_strategy": _is_external,
            "external_strategy_module": _ext_mod if _is_external else None,
            "external_strategy_func": _ext_fn if _is_external else None,
            "intent_source": "external" if _is_external else "internal",
        }
    )

    if intent.action == Action.ENTER_LONG:
        if not wallet_verified:
            print("   >> [LEGACY] HOLD | Reason: WALLET_UNVERIFIED_FAIL_CLOSED_BUY")
            return

        if risk_mult <= 0.0:
            print(f"   >> [LEGACY] HOLD | Reason: RISK_MULT_ZERO | Regime: {regime}")
            return

        if btc_notional >= MIN_NOTIONAL_USD:
            print(f"   >> [LEGACY] HOLD | Reason: ALREADY_IN_POSITION_NO_PYRAMID | BTC_Notional=${btc_notional:.2f}")
            return

        target_usd = cash * float(risk_mult)
        if target_usd < MIN_NOTIONAL_USD:
            print(f"   >> [LEGACY] HOLD | Reason: TARGET_BELOW_MIN_NOTIONAL | Target=${target_usd:.2f}")
            return

        print(f"   >> [LEGACY] BUY | target=${target_usd:.2f} px=${price:.2f} regime={regime}")
        _execute_buy(target_usd, price)
        return

    if intent.action == Action.EXIT_LONG:
        if not wallet_verified:
            print("   >> [LEGACY] HOLD | Reason: WALLET_UNVERIFIED_CANNOT_CONFIRM_POSITION")
            return

        if btc_notional < MIN_NOTIONAL_USD:
            print("   >> [LEGACY] HOLD | Reason: NO_POSITION_OR_BELOW_MIN_NOTIONAL")
            return

        if emergency_exit:
            print("   >> [LEGACY] SELL | Reason: EMERGENCY_EXIT_BYPASS_GUARDRAILS")
            _execute_sell(btc, price)
            return

        state, state_status = _load_trade_state_legacy()

        if state_status in ("CORRUPT", "MISSING"):
            print(f"   >> [LEGACY] SELL | Reason: STATE_{state_status}_FAIL_OPEN_SELL")
            _execute_sell(btc, price)
            return

        entry_time = state["_entry_time"]
        entry_price = float(state["entry_price"])

        hold_hours = (now - entry_time).total_seconds() / 3600.0
        profit_pct = (price - entry_price) / entry_price

        if profit_pct <= -STOP_LOSS_PCT:
            print(f"   >> [LEGACY] SELL | Reason: STOP_LOSS_TRIGGERED | pnl={profit_pct:.3%}")
            _execute_sell(btc, price)
            return

        if hold_hours >= MAX_HOLD_HOURS:
            print(f"   >> [LEGACY] SELL | Reason: MAX_HOLD_EXCEEDED | held={hold_hours:.2f}h pnl={profit_pct:.3%}")
            _execute_sell(btc, price)
            return

        if hold_hours < MIN_HOLD_HOURS:
            print(f"   >> [LEGACY] HOLD | Reason: MIN_HOLD_NOT_MET | held={hold_hours:.2f}h min={MIN_HOLD_HOURS:.2f}h")
            return

        effective_hurdle = PROFIT_HURDLE_PCT
        bad_regime = ("BEAR" in regime) or ("RECOVERY" in regime)
        if severity >= HURDLE_RELIEF_SEVERITY or bad_regime:
            effective_hurdle = PROFIT_HURDLE_PCT * HURDLE_RELIEF_FACTOR

        if profit_pct < effective_hurdle:
            print(
                f"   >> [LEGACY] HOLD | Reason: PROFIT_HURDLE_NOT_MET | pnl={profit_pct:.3%} hurdle={effective_hurdle:.3%}"
            )
            return

        print(f"   >> [LEGACY] SELL | pnl={profit_pct:.3%} held={hold_hours:.2f}h hurdle={effective_hurdle:.3%}")
        _execute_sell(btc, price)
        return

    print(f"   >> [LEGACY] {intent.action} | reason={intent.reason} | regime={regime}")


# ============================================================
# ENTRYPOINT
# ============================================================

def generate_signals() -> None:
    global _PROVENANCE_PRINTED

    if not _PROVENANCE_PRINTED:
        state_path, _tmp, state_mode = _prime_state_paths()
        print("   >> [PROVENANCE] signal_generator loaded from:", str(_CURRENT_FILE))
        print("   >> [PROVENANCE] project_root:", str(_PROJECT_ROOT))
        print("   >> [PROVENANCE] ARGUS_MODE:", ARGUS_MODE)
        print("   >> [PROVENANCE] DRY_RUN:", str(_is_dry_run()))
        print("   >> [PROVENANCE] MODEL_FILE:", str(MODEL_FILE))
        print("   >> [PROVENANCE] DATA_FILE:", str(DATA_FILE))
        print("   >> [PROVENANCE] PRIME_STATE_MODE:", state_mode)
        print("   >> [PROVENANCE] PRIME_STATE_FILE:", str(state_path))
        if ARGUS_MODE == "prime":
            print("   >> [PROVENANCE] PRIME_REENTRY_COOLDOWN_H:", PRIME_REENTRY_COOLDOWN_H)
            print("   >> [PROVENANCE] PRIME_EXIT_MIN_HOLD_H:", PRIME_EXIT_MIN_HOLD_H)

        mod = os.getenv("ARGUS_STRATEGY_MODULE", "").strip()
        fn = os.getenv("ARGUS_STRATEGY_FUNC", "").strip()
        if mod and fn:
            print("   >> [PROVENANCE] EXTERNAL_STRATEGY_MODULE:", mod)
            print("   >> [PROVENANCE] EXTERNAL_STRATEGY_FUNC:", fn)
        else:
            print("   >> [PROVENANCE] EXTERNAL_STRATEGY: (none)")

        _PROVENANCE_PRINTED = True

    update_market_data()

    model, model_path_or_err = _load_model()
    if model is None:
        print(f"   >> [DECISION] HOLD | Reason: MODEL_LOAD_FAIL | Error: {model_path_or_err}")
        return

    df = _load_data()
    if df is None or df.empty or len(df) < 210:
        print("   >> [DECISION] HOLD | Reason: INSUFFICIENT_HISTORY_OR_DATA_LOAD_FAIL")
        return

    if ARGUS_MODE == "prime":
        _run_prime_engine(df, model, model_path_or_err)
    else:
        _run_legacy_engine(df, model, model_path_or_err)


if __name__ == "__main__":
    generate_signals()
