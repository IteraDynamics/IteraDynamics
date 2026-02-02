# /opt/argus/apex_core/signal_generator.py
# 🦅 ARGUS SIGNAL ENGINE (LEGACY + ARGUS PRIME) — FULL FILE REPLACEMENT
#
# ARGUS_MODE:
#   - "prime"  -> Argus Prime live pilot (conf-gated proba + horizon hold + DD governors + sizing)
#   - else     -> Legacy engine (your existing v3.6 behavior)
#
# Prime risk controls:
#   1) PRIME_DD_SOFT: reduce exposure (default multiplier 0.5)
#   2) PRIME_DD_HARD: block new entries, allow exits
#   3) PRIME_DD_KILL: liquidate and set killed flag (requires manual reset)
#
# Prime position sizing:
#   size_usd = equity_usd * min(PRIME_MAX_EXPOSURE, dd_adjusted_exposure)
#
# Prime: single position, no overlap, hold exactly PRIME_HORIZON hours.
#
# IMPORTANT:
# - In DRY-RUN, Prime writes paper state to paper_prime_state.json (never prime_state.json)
# - In LIVE, Prime writes to prime_state.json

from __future__ import annotations

import sys
import os
import json
import joblib
import requests
import pandas as pd
import pandas_ta as ta
import numpy as np  # <-- ADDED (quantile support)

from collections import deque  # <-- ADDED (rolling p_long window)
from pathlib import Path
from datetime import datetime, timezone, timedelta
from dotenv import load_dotenv

# ---------------------------
# Path / env resolution (/opt/argus as project root)
# ---------------------------

_CURRENT_FILE = Path(__file__).resolve()
_PROJECT_ROOT = _CURRENT_FILE.parent.parent  # /opt/argus

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
    sys.exit(1)

_broker = RealBroker()

# ---------------------------
# Helpers
# ---------------------------

_PROVENANCE_PRINTED = False

def _utc_now() -> datetime:
    return datetime.now(timezone.utc)

def _utc_ts() -> str:
    return _utc_now().strftime("%Y-%m-%d %H:%M:%S")

def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))

def _atomic_write_json(path: Path, tmp: Path, payload: dict) -> None:
    tmp.parent.mkdir(parents=True, exist_ok=True)
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, separators=(",", ":"), sort_keys=True)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)

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
    # Primary switch is ARGUS_DRY_RUN (since RealBroker prints its mode from this)
    # PRIME_DRY_RUN exists to force dry-run when ARGUS_DRY_RUN isn't already set.
    return _parse_bool_env("ARGUS_DRY_RUN", False) or _parse_bool_env("PRIME_DRY_RUN", False)

# ---------------------------
# Shared runtime assets
# ---------------------------

MODELS_DIR = _PROJECT_ROOT / "models"
MODEL_FILE = os.getenv("ARGUS_MODEL_FILE", "random_forest.pkl")

DATA_FILE = _PROJECT_ROOT / "flight_recorder.csv"

# Legacy state + cortex (kept for compatibility)
STATE_FILE = _PROJECT_ROOT / "trade_state.json"
CORTEX_FILE = _PROJECT_ROOT / "cortex.json"
CORTEX_TMP = _PROJECT_ROOT / "cortex.json.tmp"

PRODUCT_ID = "BTC-USD"

# ---------------------------
# PRIME config (env)
# ---------------------------

ARGUS_MODE = os.getenv("ARGUS_MODE", "legacy").strip().lower()

PRIME_CONF_MIN = _safe_float(os.getenv("PRIME_CONF_MIN", "0.64"), 0.64)
PRIME_HORIZON_H = int(_safe_float(os.getenv("PRIME_HORIZON", "48"), 48))
PRIME_MAX_EXPOSURE = _safe_float(os.getenv("PRIME_MAX_EXPOSURE", "0.25"), 0.25)

PRIME_DD_SOFT = _safe_float(os.getenv("PRIME_DD_SOFT", "0.03"), 0.03)
PRIME_DD_HARD = _safe_float(os.getenv("PRIME_DD_HARD", "0.06"), 0.06)
PRIME_DD_KILL = _safe_float(os.getenv("PRIME_DD_KILL", "0.10"), 0.10)

# When below soft DD, cut exposure by this factor (0.5 = half size)
PRIME_DD_SOFT_MULT = _safe_float(os.getenv("PRIME_DD_SOFT_MULT", "0.5"), 0.5)

# Minimum notional to avoid dust
PRIME_MIN_NOTIONAL_USD = _safe_float(os.getenv("PRIME_MIN_NOTIONAL_USD", "5.00"), 5.00)

# Prime state files (paper vs live)
PRIME_STATE_FILE_LIVE = _PROJECT_ROOT / "prime_state.json"
PRIME_STATE_TMP_LIVE  = _PROJECT_ROOT / "prime_state.json.tmp"

PRIME_STATE_FILE_PAPER = _PROJECT_ROOT / "paper_prime_state.json"
PRIME_STATE_TMP_PAPER  = _PROJECT_ROOT / "paper_prime_state.json.tmp"

# ---------------------------
# PRIME dynamic conf (quantile) config (env)  <-- ADDED
# ---------------------------

PRIME_CONF_MODE = os.getenv("PRIME_CONF_MODE", "fixed").strip().lower()  # fixed | quantile
PRIME_CONF_Q = _safe_float(os.getenv("PRIME_CONF_Q", "0.95"), 0.95)      # quantile in [0,1]
PRIME_CONF_WINDOW = int(_safe_float(os.getenv("PRIME_CONF_WINDOW", "400"), 400))
PRIME_CONF_FLOOR = _safe_float(os.getenv("PRIME_CONF_FLOOR", "0.45"), 0.45)
PRIME_CONF_MIN_SAMPLES = int(_safe_float(os.getenv("PRIME_CONF_MIN_SAMPLES", "50"), 50))

# Rolling p_long history buffer (filled once, then appended each cycle)  <-- ADDED
_P_HIST = deque(maxlen=max(10, PRIME_CONF_WINDOW))

# If PRIME_DRY_RUN is set, enforce dry-run (without overriding systemd if already set)
if _parse_bool_env("PRIME_DRY_RUN", default=False):
    if os.getenv("ARGUS_DRY_RUN") is None:
        os.environ["ARGUS_DRY_RUN"] = "1"

# ---------------------------
# Market data update
# ---------------------------

def update_market_data() -> None:
    """Fetch latest hourly candles (UTC) and append only new rows."""
    try:
        url = "https://api.exchange.coinbase.com/products/BTC-USD/candles"
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

# ---------------------------
# PRIME state
# ---------------------------

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
            "last_decision": None,
        }
    try:
        with open(state_path, "r", encoding="utf-8") as f:
            st = json.load(f)
        if not isinstance(st, dict):
            raise ValueError("prime_state is not a dict")
        return st
    except Exception:
        # Corrupt -> fail safe: do not enter new trades
        return {"killed": True, "corrupt": True}

def _save_prime_state(st: dict) -> None:
    state_path, tmp_path, _mode = _prime_state_paths()
    _atomic_write_json(state_path, tmp_path, st)

# ---------------------------
# PRIME sizing + DD governors
# ---------------------------

def _dd_status(equity: float, peak: float) -> tuple[float, str, float]:
    """
    Returns (dd_frac_negative, dd_band, exposure_mult)
    dd_frac_negative: negative number (e.g. -0.04 means -4% drawdown)
    dd_band: "ok" | "soft" | "hard" | "kill"
    exposure_mult: multiplier applied to PRIME_MAX_EXPOSURE for new entries
    """
    if peak <= 0:
        return 0.0, "ok", 1.0
    dd = equity / peak - 1.0  # <= 0

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

# ---------------------------
# PRIME features + proba
# ---------------------------

def _build_prime_features(df: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    """
    Returns (feature_df, last_price)
    Feature columns must match training: RSI, BB_Pos, Vol_Z
    """
    df = df.copy()
    df["RSI"] = ta.rsi(df["Close"], length=14)

    bband = ta.bbands(df["Close"], length=20, std=2)
    if bband is None or bband.shape[1] < 3:
        raise RuntimeError("BBANDS_UNAVAILABLE")
    df["BB_Pos"] = (df["Close"] - bband.iloc[:, 0]) / (bband.iloc[:, 2] - bband.iloc[:, 0])

    vol_mean = df["Volume"].rolling(20).mean()
    vol_std = df["Volume"].rolling(20).std()
    df["Vol_Z"] = (df["Volume"] - vol_mean) / vol_std.replace(0, pd.NA)

    feat_row = [df["RSI"].iloc[-1], df["BB_Pos"].iloc[-1], df["Vol_Z"].iloc[-1]]
    if any(pd.isna(x) for x in feat_row):
        raise RuntimeError("FEATURES_NAN")

    feat = pd.DataFrame([feat_row], columns=["RSI", "BB_Pos", "Vol_Z"])
    price = float(df["Close"].iloc[-1])
    if not (price > 0):
        raise RuntimeError("BAD_PRICE")
    return feat, price

# ---------------------------
# PRIME dynamic conf helpers  <-- ADDED
# ---------------------------

def _backfill_p_hist(model, df: pd.DataFrame) -> None:
    """
    Fill rolling p_long history once from recent bars using the exact same feature pipeline.
    Skips bars where feature build fails.
    """
    global _P_HIST
    _P_HIST.clear()

    # Ensure we keep enough history for indicators (df length must be >= 210 already checked).
    start = max(210, len(df) - PRIME_CONF_WINDOW)
    for i in range(start, len(df)):
        try:
            sub = df.iloc[: i + 1]
            feat_i, _ = _build_prime_features(sub)
            p_i = float(model.predict_proba(feat_i)[0][1])
            _P_HIST.append(p_i)
        except Exception:
            continue

def _conf_from_hist() -> tuple[float | None, dict]:
    """
    Returns (quantile_value_or_None, stats_dict). If insufficient samples, returns (None, {"n": n}).
    """
    n = len(_P_HIST)
    if n < PRIME_CONF_MIN_SAMPLES:
        return None, {"n": n}

    arr = np.array(_P_HIST, dtype=float)
    qv = float(np.quantile(arr, PRIME_CONF_Q))
    return qv, {
        "n": n,
        "min": float(arr.min()),
        "mean": float(arr.mean()),
        "max": float(arr.max()),
        "q": qv,
    }

# ---------------------------
# PRIME engine
# ---------------------------

def _run_prime() -> None:
    global _PROVENANCE_PRINTED

    cycle_start = _utc_ts()
    dry = _is_dry_run()
    state_path, _tmp, state_mode = _prime_state_paths()

    if not _PROVENANCE_PRINTED:
        print("   >> [PROVENANCE] signal_generator loaded from:", str(_CURRENT_FILE))
        print("   >> [PROVENANCE] project_root:", str(_PROJECT_ROOT))
        print("   >> [PROVENANCE] ARGUS_MODE:", ARGUS_MODE)
        print("   >> [PROVENANCE] DRY_RUN:", str(dry))
        print("   >> [PROVENANCE] PRIME_STATE_MODE:", state_mode)
        print("   >> [PROVENANCE] PRIME_STATE_FILE:", str(state_path))
        print("   >> [PROVENANCE] MODEL_FILE:", str(MODEL_FILE))
        print("   >> [PROVENANCE] DATA_FILE:", str(DATA_FILE))
        print("   >> [PROVENANCE] PRIME_CONF_MIN:", PRIME_CONF_MIN)
        print("   >> [PROVENANCE] PRIME_CONF_MODE:", PRIME_CONF_MODE)
        print("   >> [PROVENANCE] PRIME_CONF_Q:", PRIME_CONF_Q)
        print("   >> [PROVENANCE] PRIME_CONF_WINDOW:", PRIME_CONF_WINDOW)
        print("   >> [PROVENANCE] PRIME_CONF_FLOOR:", PRIME_CONF_FLOOR)
        print("   >> [PROVENANCE] PRIME_HORIZON_H:", PRIME_HORIZON_H)
        print("   >> [PROVENANCE] PRIME_MAX_EXPOSURE:", PRIME_MAX_EXPOSURE)
        print("   >> [PROVENANCE] PRIME_DD_SOFT/HARD/KILL:", PRIME_DD_SOFT, PRIME_DD_HARD, PRIME_DD_KILL)
        _PROVENANCE_PRINTED = True

    update_market_data()

    # Load model + data
    try:
        model_path = MODELS_DIR / MODEL_FILE
        print ("   >>[PROVENANCE] MODEL_PATH:", str(model_path))
        model = joblib.load(model_path)
        df = pd.read_csv(DATA_FILE)
    except Exception as e:
        print(f"   >> [PRIME] HOLD | Reason: MODEL_OR_DATA_LOAD_FAIL | Error: {e}")
        return

    if df.empty or len(df) < 210:
        print("   >> [PRIME] HOLD | Reason: INSUFFICIENT_HISTORY")
        return

    # Features + proba
    try:
        feat, price = _build_prime_features(df)
        if not hasattr(model, "predict_proba"):
            raise RuntimeError("MODEL_NO_PREDICT_PROBA")
        p_long = float(model.predict_proba(feat)[0][1])
    except Exception as e:
        print(f"   >> [PRIME] HOLD | Reason: FEATURES_OR_PROBA_FAIL | Error: {e}")
        return

    # ---------------------------
    # Dynamic confidence threshold (quantile mode)  <-- ADDED
    # ---------------------------

    conf_mode_eff = PRIME_CONF_MODE
    conf_dyn = None
    pstats = None
    conf_eff = PRIME_CONF_MIN

    if conf_mode_eff == "quantile":
        try:
            # Backfill once per process start
            if len(_P_HIST) == 0:
                _backfill_p_hist(model, df)

            # Append current cycle p_long
            _P_HIST.append(p_long)

            conf_dyn, pstats = _conf_from_hist()
            if conf_dyn is None:
                # Not enough samples -> fallback to fixed (avoid bricking)
                conf_mode_eff = "fixed"
                conf_eff = PRIME_CONF_MIN
            else:
                conf_eff = max(PRIME_CONF_FLOOR, float(conf_dyn))
        except Exception as e:
            print(f"   >> [PRIME] HOLD | Reason: CONF_DYNAMIC_FAIL | Error: {e}")
            # fail-closed: fallback to fixed (still allows trading if fixed is reachable)
            conf_mode_eff = "fixed"
            conf_eff = PRIME_CONF_MIN

    # Wallet snapshot (Prime always fail-closed on entry if wallet unverified)
    wallet_verified = False
    cash = 0.0
    btc = 0.0
    wallet_err = None
    try:
        cash, btc = _broker.get_wallet_snapshot()
        wallet_verified = True
    except Exception as e:
        wallet_err = str(e)

    equity = cash + btc * price
    btc_notional = btc * price

    st = _load_prime_state()

    # ---------------------------
    # Dashboard snapshot writer (Prime)  <-- ADDED / expanded to all cycles
    # ---------------------------

    def _write_prime_cortex_snapshot() -> None:
        payload = {
            "timestamp_utc": cycle_start,
            "mode": "prime",
            "p_long": float(p_long),
            "conf_min": float(conf_eff),
            "conf_mode": str(conf_mode_eff),
            "conf_q": float(PRIME_CONF_Q),
            "conf_window": int(PRIME_CONF_WINDOW),
            "conf_floor": float(PRIME_CONF_FLOOR),
            "p_long_q": float(conf_dyn) if conf_dyn is not None else None,
            "p_long_hist_n": int(pstats["n"]) if isinstance(pstats, dict) and "n" in pstats else int(len(_P_HIST)),
            "p_long_hist_min": float(pstats["min"]) if isinstance(pstats, dict) and "min" in pstats else None,
            "p_long_hist_mean": float(pstats["mean"]) if isinstance(pstats, dict) and "mean" in pstats else None,
            "p_long_hist_max": float(pstats["max"]) if isinstance(pstats, dict) and "max" in pstats else None,
            "horizon_h": int(PRIME_HORIZON_H),
            "max_exposure": float(PRIME_MAX_EXPOSURE),
            "equity_usd": float(equity),
            "cash_usd": float(cash) if wallet_verified else None,
            "btc": float(btc) if wallet_verified else None,
            "btc_notional_usd": float(btc_notional) if wallet_verified else None,
            "peak_equity_usd": float(_safe_float(st.get("peak_equity_usd"), equity)),
            "drawdown_frac": float(_safe_float(st.get("last_drawdown_frac"), 0.0)) if "last_drawdown_frac" in st else None,
            "dd_band": st.get("last_dd_band"),
            "planned_exit_ts": st.get("planned_exit_ts"),
            "last_decision": st.get("last_decision"),
            "model_file": str(MODEL_FILE),
            "model_path": str((MODELS_DIR / MODEL_FILE)),
            "dry_run": bool(dry),
            "prime_state_file": str(state_path),
        }
        try:
            _atomic_write_json(CORTEX_FILE, CORTEX_TMP, payload)
        except Exception:
            pass

    if st.get("killed") is True:
        print("   >> [PRIME] KILLED STATE ACTIVE -> no entries until reset state file")

        # In LIVE, liquidate from wallet. In DRY-RUN, liquidate from paper state (best-effort).
        if wallet_verified:
            if dry:
                paper_qty = _safe_float(st.get("qty_btc"), 0.0)
                paper_in_pos = bool(st.get("in_position", False))
                paper_notional = paper_qty * price
                if paper_in_pos and paper_notional >= PRIME_MIN_NOTIONAL_USD and paper_qty > 0:
                    print("   >> [PRIME] KILLED (DRY) -> LIQUIDATING PAPER BTC (best-effort)")
                    _broker.execute_trade("SELL", paper_qty, price)
            else:
                if btc_notional >= PRIME_MIN_NOTIONAL_USD and btc > 0:
                    print("   >> [PRIME] KILLED -> LIQUIDATING REMAINING BTC (best-effort)")
                    _broker.execute_trade("SELL", btc, price)
        _write_prime_cortex_snapshot()
        return

    if st.get("corrupt") is True:
        print("   >> [PRIME] HOLD | Reason: PRIME_STATE_CORRUPT_FAILSAFE (treated as killed)")
        _write_prime_cortex_snapshot()
        return

    # Update peak equity
    peak = _safe_float(st.get("peak_equity_usd"), equity)
    if peak <= 0:
        peak = equity
    if equity > peak:
        peak = equity

    dd, dd_band, dd_expo_mult = _dd_status(equity=equity, peak=peak)

    # record dd metadata for dashboard snapshot
    st["last_drawdown_frac"] = float(dd)
    st["last_dd_band"] = str(dd_band)

    # Prime decision: desired = long if p_long >= conf, else flat
    want_long = p_long >= conf_eff  # <-- CHANGED (was PRIME_CONF_MIN)

    # ---------------------------
    # Position authority
    # LIVE: wallet BTC is authoritative
    # DRY : paper prime state is authoritative (prevents repeated re-entries)
    # ---------------------------
    btc_for_exit = btc  # default
    if dry:
        paper_qty = _safe_float(st.get("qty_btc"), 0.0)
        paper_in_pos = bool(st.get("in_position", False))
        paper_notional = paper_qty * price
        in_pos = bool(paper_in_pos and (paper_notional >= PRIME_MIN_NOTIONAL_USD) and (paper_qty > 0))
        btc_for_exit = paper_qty
    else:
        in_pos = btc_notional >= PRIME_MIN_NOTIONAL_USD
        btc_for_exit = btc

    now = _utc_now()
    planned_exit_ts = None
    if st.get("planned_exit_ts"):
        try:
            planned_exit_ts = datetime.fromisoformat(st["planned_exit_ts"])
            if planned_exit_ts.tzinfo is None:
                planned_exit_ts = planned_exit_ts.replace(tzinfo=timezone.utc)
        except Exception:
            planned_exit_ts = None

    # Kill rule
    if dd_band == "kill":
        if wallet_verified and in_pos and btc_for_exit > 0:
            print(f"   >> [PRIME] DD_KILL TRIGGERED (dd={dd:.3%}) -> FORCED LIQUIDATION")
            _broker.execute_trade("SELL", btc_for_exit, price)
        st["killed"] = True
        st["last_decision"] = f"KILL_DD dd={dd:.6f}"
        st["peak_equity_usd"] = peak
        st["last_equity_usd"] = equity
        _save_prime_state(st)
        _write_prime_cortex_snapshot()
        return

    # Horizon exit
    if wallet_verified and in_pos and planned_exit_ts is not None and now >= planned_exit_ts:
        print(f"   >> [PRIME] EXIT: HORIZON_REACHED -> SELL ALL | p={p_long:.3f} conf={conf_eff:.3f}")
        if btc_for_exit > 0:
            _broker.execute_trade("SELL", btc_for_exit, price)
        st.update(
            {
                "in_position": False,
                "entry_ts": None,
                "entry_px": None,
                "qty_btc": None,
                "planned_exit_ts": None,
                "last_decision": "EXIT_HORIZON",
                "peak_equity_usd": peak,
                "last_equity_usd": equity,
            }
        )
        _save_prime_state(st)
        _write_prime_cortex_snapshot()
        return

    # Hard DD: no new entries
    if dd_band == "hard":
        print(f"   >> [PRIME] HARD_DD (dd={dd:.3%}) -> entries blocked (exits allowed)")
        st["peak_equity_usd"] = peak
        st["last_equity_usd"] = equity
        st["last_decision"] = f"HARD_DD_BLOCK dd={dd:.6f}"
        _save_prime_state(st)
        _write_prime_cortex_snapshot()
        return

    # In position: do nothing until horizon
    if in_pos:
        if planned_exit_ts is None:
            planned_exit_ts = now + timedelta(hours=PRIME_HORIZON_H)
            st["planned_exit_ts"] = planned_exit_ts.isoformat()
        st["in_position"] = True
        st["peak_equity_usd"] = peak
        st["last_equity_usd"] = equity
        st["last_decision"] = f"HOLD_IN_POSITION p={p_long:.4f}"
        _save_prime_state(st)

        # Optional per-cycle conf diagnostics (only if quantile mode is active)
        if conf_mode_eff == "quantile" and isinstance(pstats, dict) and pstats.get("n", 0) > 0:
            print(
                f"   >> [PRIME] CONF_GATE | mode=quantile q={PRIME_CONF_Q:.2f} n={pstats['n']} "
                f"min/mean/max={pstats['min']:.3f}/{pstats['mean']:.3f}/{pstats['max']:.3f} "
                f"dyn={pstats['q']:.3f} eff={conf_eff:.3f}"
            )

        print(
            f"   >> [PRIME] HOLD (IN POSITION) | p={p_long:.3f} conf={conf_eff:.3f} | "
            f"equity=${equity:.2f} dd={dd:.3%} band={dd_band} | exit_at={planned_exit_ts.isoformat() if planned_exit_ts else 'n/a'}"
        )
        _write_prime_cortex_snapshot()
        return

    # Flat: no entry
    if not want_long:
        st["in_position"] = False
        st["peak_equity_usd"] = peak
        st["last_equity_usd"] = equity
        st["last_decision"] = f"FLAT_NO_SIGNAL p={p_long:.4f}"
        _save_prime_state(st)

        if conf_mode_eff == "quantile" and isinstance(pstats, dict) and pstats.get("n", 0) > 0:
            print(
                f"   >> [PRIME] CONF_GATE | mode=quantile q={PRIME_CONF_Q:.2f} n={pstats['n']} "
                f"min/mean/max={pstats['min']:.3f}/{pstats['mean']:.3f}/{pstats['max']:.3f} "
                f"dyn={pstats['q']:.3f} eff={conf_eff:.3f}"
            )

        print(f"   >> [PRIME] FLAT | No entry signal | p={p_long:.3f} conf={conf_eff:.3f}")
        _write_prime_cortex_snapshot()
        return

    # Entry requires verified wallet snapshot (even in DRY-RUN, we still want the balance sanity check)
    if not wallet_verified:
        print(f"   >> [PRIME] HOLD | Reason: WALLET_UNVERIFIED_FAIL_CLOSED_ENTRY | err={wallet_err}")
        _write_prime_cortex_snapshot()
        return

    effective_max_exposure = PRIME_MAX_EXPOSURE * dd_expo_mult

    if effective_max_exposure <= 0:
        print(f"   >> [PRIME] HOLD | Reason: EXPOSURE_ZERO (dd_band={dd_band})")
        st["peak_equity_usd"] = peak
        st["last_equity_usd"] = equity
        st["last_decision"] = f"BLOCKED_EXPO band={dd_band}"
        _save_prime_state(st)
        _write_prime_cortex_snapshot()
        return

    target_usd = equity * effective_max_exposure
    target_usd = min(target_usd, cash)

    if target_usd < PRIME_MIN_NOTIONAL_USD:
        print(f"   >> [PRIME] HOLD | Reason: TARGET_BELOW_MIN_NOTIONAL | target=${target_usd:.2f}")
        st["peak_equity_usd"] = peak
        st["last_equity_usd"] = equity
        st["last_decision"] = "TARGET_TOO_SMALL"
        _save_prime_state(st)
        _write_prime_cortex_snapshot()
        return

    qty = target_usd / price
    planned_exit_ts = now + timedelta(hours=PRIME_HORIZON_H)

    if conf_mode_eff == "quantile" and isinstance(pstats, dict) and pstats.get("n", 0) > 0:
        print(
            f"   >> [PRIME] CONF_GATE | mode=quantile q={PRIME_CONF_Q:.2f} n={pstats['n']} "
            f"min/mean/max={pstats['min']:.3f}/{pstats['mean']:.3f}/{pstats['max']:.3f} "
            f"dyn={pstats['q']:.3f} eff={conf_eff:.3f}"
        )

    print(
        f"   >> [PRIME] ENTER LONG | p={p_long:.3f} conf={conf_eff:.3f} | "
        f"equity=${equity:.2f} cash=${cash:.2f} dd={dd:.3%} band={dd_band} | "
        f"maxExpo={PRIME_MAX_EXPOSURE:.3f} effExpo={effective_max_exposure:.3f} | "
        f"target=${target_usd:.2f} qty={qty:.8f} px=${price:.2f} | exit_at={planned_exit_ts.isoformat()}"
    )

    _broker.execute_trade("BUY", qty, price)

    st.update(
        {
            "killed": False,
            "peak_equity_usd": peak,
            "last_equity_usd": equity,
            "in_position": True,
            "entry_ts": now.isoformat(),
            "entry_px": price,
            "qty_btc": qty,
            "planned_exit_ts": planned_exit_ts.isoformat(),
            "last_decision": f"ENTER_LONG p={p_long:.4f}",
        }
    )
    _save_prime_state(st)
    _write_prime_cortex_snapshot()
    return


# ---------------------------
# LEGACY engine (your existing V3.6)
# ---------------------------

MIN_NOTIONAL_USD = float(os.getenv("ARGUS_MIN_NOTIONAL_USD", "5.0"))
MIN_HOLD_HOURS = float(os.getenv("ARGUS_MIN_HOLD_HOURS", "4.0"))
PROFIT_HURDLE_PCT = float(os.getenv("ARGUS_PROFIT_HURDLE_PCT", "0.0035"))
EMERGENCY_SEVERITY_THRESHOLD = float(os.getenv("ARGUS_EMERGENCY_SEVERITY_THRESHOLD", "0.85"))
STOP_LOSS_PCT = float(os.getenv("ARGUS_STOP_LOSS_PCT", "0.02"))
MAX_HOLD_HOURS = float(os.getenv("ARGUS_MAX_HOLD_HOURS", "72.0"))
HURDLE_RELIEF_SEVERITY = float(os.getenv("ARGUS_HURDLE_RELIEF_SEVERITY", "0.60"))
HURDLE_RELIEF_FACTOR = float(os.getenv("ARGUS_HURDLE_RELIEF_FACTOR", "0.5"))

def detect_regime(df: pd.DataFrame):
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

def _load_trade_state():
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

def _run_legacy() -> None:
    global _PROVENANCE_PRINTED

    cycle_start = _utc_ts()

    if not _PROVENANCE_PRINTED:
        try:
            print("   >> [PROVENANCE] signal_generator loaded from:", str(_CURRENT_FILE))
            print("   >> [PROVENANCE] project_root:", str(_PROJECT_ROOT))
            print("   >> [PROVENANCE] MODELS_DIR:", str(MODELS_DIR))
            print("   >> [PROVENANCE] MODEL_FILE:", str(MODEL_FILE))
            print("   >> [PROVENANCE] DATA_FILE:", str(DATA_FILE))
            print("   >> [PROVENANCE] STATE_FILE:", str(STATE_FILE))
            print("   >> [PROVENANCE] PRODUCT_ID:", str(PRODUCT_ID))
        except Exception:
            pass
        _PROVENANCE_PRINTED = True

    update_market_data()

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

    try:
        regime, risk_mult, severity, emergency_exit = detect_regime(df)
        raw_signal = "BUY" if int(model.predict(feat)[0]) == 1 else "SELL"
    except Exception as e:
        print(f"   >> [DECISION] HOLD | Reason: REGIME_OR_MODEL_PREDICT_FAIL | Error: {e}")
        return

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

    print(f"   >> [BRAIN] Raw Signal: {raw_signal}")
    print(
        f"   >> [REGIME] {regime} | Risk Multiplier: {risk_mult:.2f} | Severity: {severity:.2f} | EmergencyExit: {str(emergency_exit)}"
    )
    if wallet_verified:
        print(f"   >> [WALLET] VERIFIED | Cash: ${cash:.2f} | BTC: {btc:.8f} | BTC Notional: ${btc_notional:.2f}")
    else:
        print(f"   >> [WALLET] UNVERIFIED | Error: {wallet_err}")

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

    if raw_signal == "BUY":
        if not wallet_verified:
            print("   >> [DECISION] HOLD | Reason: WALLET_UNVERIFIED_FAIL_CLOSED_BUY")
            return
        if risk_mult <= 0.0:
            print(f"   >> [DECISION] HOLD | Reason: RISK_MULT_ZERO | Regime: {regime}")
            return
        if btc_notional >= MIN_NOTIONAL_USD:
            print(
                f"   >> [DECISION] HOLD | Reason: ALREADY_IN_POSITION_NO_PYRAMID | "
                f"BTC_Notional: ${btc_notional:.2f}"
            )
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

    if raw_signal == "SELL":
        if not wallet_verified:
            print("   >> [DECISION] HOLD | Reason: WALLET_UNVERIFIED_CANNOT_CONFIRM_POSITION")
            return

        if btc_notional < MIN_NOTIONAL_USD:
            print("   >> [DECISION] HOLD | Reason: NO_POSITION_OR_BELOW_MIN_NOTIONAL")
            return

        if emergency_exit:
            print("   >> [EXECUTION] ROUTE SELL | Reason: EMERGENCY_EXIT_BYPASS_GUARDRAILS")
            _broker.execute_trade("SELL", btc, price)
            return

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

        if profit_pct <= -STOP_LOSS_PCT:
            print(
                f"   >> [EXECUTION] ROUTE SELL | Reason: STOP_LOSS_TRIGGERED | Loss: {profit_pct:.3%} | "
                f"Entry: ${entry_price:.2f} | Now: ${price:.2f}"
            )
            _broker.execute_trade("SELL", btc, price)
            return

        if hold_hours >= MAX_HOLD_HOURS:
            print(
                f"   >> [EXECUTION] ROUTE SELL | Reason: MAX_HOLD_EXCEEDED | Held: {hold_hours:.2f}h | "
                f"Entry: ${entry_price:.2f} | Now: ${price:.2f} | PnL: {profit_pct:.3%}"
            )
            _broker.execute_trade("SELL", btc, price)
            return

        if hold_hours < MIN_HOLD_HOURS:
            print(
                f"   >> [DECISION] HOLD | Reason: MIN_HOLD_NOT_MET | Held: {hold_hours:.2f}h | "
                f"Min: {MIN_HOLD_HOURS:.2f}h"
            )
            return

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

# ---------------------------
# Entry point
# ---------------------------

def generate_signals() -> None:
    if ARGUS_MODE == "prime":
        _run_prime()
    else:
        _run_legacy()

if __name__ == "__main__":
    generate_signals()
