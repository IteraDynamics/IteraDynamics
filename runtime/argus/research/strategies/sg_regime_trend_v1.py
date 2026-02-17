"""
SG Regime Trend v1 (external strategy module)
=============================================

Layer 2 Strategy: Pluggable, deterministic, per-regime alpha module.

This module MUST be compatible with two callers:

1) Backtest harness expects:
   - _get_env_cfg() -> dict with keys: ema_fast, ema_slow, atr_len, adx_len, etc.
   - generate_intent_from_values(...) called with keyword args like:
       generate_intent_from_values(
          ema_fast=..., ema_slow=..., atr_pct_raw=..., atr_pct_s=...,
          trend_strength_s=..., adx=..., adx_rel=..., spike_ratio=..., cfg=cfg
       )

2) Signal generator external strategy expects:
   - generate_intent(df, ctx) -> dict compatible with StrategyIntent(**dict)


TIMELINE SAFETY (non-negotiable):
- closed_only=True (default): ALWAYS drop the last row before computing indicators
- All decisions based on CLOSED candles only
- Calls Layer 1 regime engine with closed_only=True


ENVIRONMENT VARIABLES REFERENCE
===============================

All env vars are optional; defaults shown below.

INDICATOR LENGTHS:
  SGRT_EMA_FAST=20           Fast EMA period
  SGRT_EMA_SLOW=50           Slow EMA period
  SGRT_ATR_LEN=14            ATR calculation period
  SGRT_ADX_LEN=14            ADX calculation period

ENTRY THRESHOLDS:
  SGRT_TREND_MIN=0.25        Min trend strength (|EMA diff|/ATR) for entry
  SGRT_VOL_MIN=0.003         Min smoothed ATR% for entry (avoid dead markets)
  SGRT_VOL_MAX=0.020         Max smoothed ATR% for entry (avoid chaos)
  SGRT_SPIKE_RATIO_MAX=1.8   Max ratio of raw vs smoothed ATR% (spike filter)

EXIT THRESHOLDS:
  SGRT_VOL_EXIT_MAX=0.030    Emergency exit if raw ATR% exceeds this

POSITION SIZING:
  SGRT_EXPO_FRAC=0.25        Desired exposure fraction [0-1]
  SGRT_HORIZON_H=36          Trade horizon in hours (time-based exit)

SMOOTHING WINDOWS:
  SGRT_VOL_WIN=24            Rolling window for smoothed ATR%
  SGRT_TREND_WIN=24          Rolling window for smoothed trend strength

ADX GATING (strategy-level entry filter):
  SGRT_ADX_MIN=0             Min absolute ADX for entry (0 = disabled)
  SGRT_ADX_REL_WIN=168       Window for ADX rolling median (relative calc)
  SGRT_ADX_REL_MIN=1.00      Min ADX relative to its median (adx/median >= this)
  SGRT_ADX_MODE=soft         Gate mode: off|soft|hard
                             - off:  bypass ADX checks entirely
                             - soft: apply confidence penalty if ADX fails
                             - hard: block entry if ADX fails
  SGRT_ADX_SOFT_PENALTY=0.10 Confidence penalty when ADX fails (soft mode)

CHURN KILLERS (consumed by backtest executor/position sim layer):
  PRIME_REENTRY_COOLDOWN_H=0   Hours to wait after exit before re-entry
  PRIME_EXIT_MIN_HOLD_H=0      Minimum hold time before exit allowed

p_long CONFIRMATION (optional, from external model):
  SGRT_P_LONG_MIN=0.55       Min p_long to avoid confidence penalty
  SGRT_P_LONG_BOOST_60=0.05  Confidence boost if p_long >= 0.60
  SGRT_P_LONG_BOOST_70=0.10  Confidence boost if p_long >= 0.70

BACKTEST DATA:
  BT_DATASET=<path>          Override dataset CSV path (default: auto-detect)


VALIDATION:
  - vol_min <= vol_max <= vol_exit_max
  - spike_ratio_max > 0
  - adx_rel_win > 0
  - adx_rel_min >= 0
  - adx_mode in (off, soft, hard)
  - expo_frac in (0, 1]
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import pandas as pd

try:
    import pandas_ta as ta
except Exception:
    ta = None

# Layer 1 import: Regime Engine
# Must resolve when sys.path includes runtime/argus
try:
    from research.regime.regime_engine import classify_regime, RegimeState, RegimeLabel
    _REGIME_ENGINE_AVAILABLE = True
except ImportError:
    _REGIME_ENGINE_AVAILABLE = False
    RegimeState = None
    RegimeLabel = None


# Module-level flag for one-time config logging
_cfg_logged = False


# -------------------------
# Env helpers
# -------------------------

def _get_env_float(name: str, default: float) -> float:
    v = os.getenv(name, "")
    if v == "":
        return float(default)
    try:
        return float(v)
    except Exception:
        return float(default)


def _get_env_int(name: str, default: int) -> int:
    v = os.getenv(name, "")
    if v == "":
        return int(default)
    try:
        return int(float(v))
    except Exception:
        return int(default)


def _get_env_str(name: str, default: str) -> str:
    v = os.getenv(name, "")
    return v.strip() if v.strip() else default


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _safe_float(x: Any, default: float = float("nan")) -> float:
    try:
        return float(x)
    except Exception:
        return default


def _safe_last(s: pd.Series, default: float = float("nan")) -> float:
    try:
        return float(s.iloc[-1])
    except Exception:
        return default


# -------------------------
# Config validation
# -------------------------

def _validate_cfg(cfg: dict) -> None:
    """
    Validate env config values are sane. Raises ValueError on bad config.
    Called at strategy init to catch ghost config issues early.
    """
    errors = []

    # vol_min <= vol_max
    vol_min = float(cfg.get("vol_min", 0.003))
    vol_max = float(cfg.get("vol_max", 0.020))
    if vol_min > vol_max:
        errors.append(f"vol_min ({vol_min}) > vol_max ({vol_max})")

    # vol_max <= vol_exit_max
    vol_exit_max = float(cfg.get("vol_exit_max", 0.030))
    if vol_max > vol_exit_max:
        errors.append(f"vol_max ({vol_max}) > vol_exit_max ({vol_exit_max})")

    # spike_ratio_max > 0
    spike_ratio_max = float(cfg.get("spike_ratio_max", 1.8))
    if spike_ratio_max <= 0:
        errors.append(f"spike_ratio_max ({spike_ratio_max}) must be > 0")

    # adx_rel_win > 0
    adx_rel_win = int(cfg.get("adx_rel_win", 168))
    if adx_rel_win <= 0:
        errors.append(f"adx_rel_win ({adx_rel_win}) must be > 0")

    # adx_rel_min >= 0
    adx_rel_min = float(cfg.get("adx_rel_min", 1.0))
    if adx_rel_min < 0:
        errors.append(f"adx_rel_min ({adx_rel_min}) must be >= 0")

    # adx_mode in valid set
    adx_mode = str(cfg.get("adx_mode", "soft")).lower()
    if adx_mode not in ("off", "soft", "hard"):
        errors.append(f"adx_mode ({adx_mode}) must be one of: off, soft, hard")

    # trend_min >= 0
    trend_min = float(cfg.get("trend_min", 0.25))
    if trend_min < 0:
        errors.append(f"trend_min ({trend_min}) must be >= 0")

    # expo_frac in (0, 1]
    expo_frac = float(cfg.get("expo_frac", 0.25))
    if not (0 < expo_frac <= 1.0):
        errors.append(f"expo_frac ({expo_frac}) must be in (0, 1]")

    if errors:
        raise ValueError(f"[SG_REGIME_TREND_V1] Invalid config:\n  " + "\n  ".join(errors))


def _log_effective_cfg(cfg: dict, prefix: str = "[SGRT]") -> None:
    """Log effective config for debugging / reproducibility."""
    import sys
    print(f"{prefix} Effective config:", file=sys.stderr)
    keys = [
        "ema_fast", "ema_slow", "atr_len", "adx_len",
        "trend_min", "vol_min", "vol_max", "vol_exit_max",
        "horizon_h", "expo_frac",
        "vol_win", "trend_win", "spike_ratio_max",
        "adx_min", "adx_rel_win", "adx_rel_min", "adx_mode", "adx_soft_penalty",
        "reentry_cooldown_h", "exit_min_hold_h",
    ]
    for k in keys:
        if k in cfg:
            print(f"{prefix}   {k}={cfg[k]}", file=sys.stderr)


# -------------------------
# REQUIRED by backtest harness
# -------------------------

def _get_env_cfg() -> Dict[str, Any]:
    """
    Backtest harness compatibility layer.

    Provides canonical keys expected by the harness:
      cfg["ema_fast"], cfg["ema_slow"], cfg["atr_len"], cfg["adx_len"], etc.

    Note: reentry_cooldown_h and exit_min_hold_h are *executor-owned* controls.
    They are included here for visibility, but the position simulation layer should
    override/own them.
    """
    ema_fast = _get_env_int("SGRT_EMA_FAST", 20)
    ema_slow = _get_env_int("SGRT_EMA_SLOW", 50)
    atr_len = _get_env_int("SGRT_ATR_LEN", 14)
    adx_len = _get_env_int("SGRT_ADX_LEN", 14)

    cfg = {
        # ---- harness-critical indicator lens keys ----
        "ema_fast": ema_fast,
        "ema_slow": ema_slow,
        "atr_len": atr_len,
        "adx_len": adx_len,

        # ---- aliases (safe) ----
        "ema_fast_n": ema_fast,
        "ema_slow_n": ema_slow,

        # ---- strategy thresholds (BTParams keys) ----
        "trend_min": _get_env_float("SGRT_TREND_MIN", 0.25),
        "vol_max": _get_env_float("SGRT_VOL_MAX", 0.020),
        "horizon_h": _get_env_int("SGRT_HORIZON_H", 36),
        "vol_exit_max": _get_env_float("SGRT_VOL_EXIT_MAX", 0.030),

        # ---- churn killers (executor-owned; included for visibility) ----
        "reentry_cooldown_h": _get_env_float("PRIME_REENTRY_COOLDOWN_H", 0.0),
        "exit_min_hold_h": _get_env_float("PRIME_EXIT_MIN_HOLD_H", 0.0),

        # ---- smoothing / spike handling ----
        "vol_win": _get_env_int("SGRT_VOL_WIN", 24),
        "trend_win": _get_env_int("SGRT_TREND_WIN", 24),
        "spike_ratio_max": _get_env_float("SGRT_SPIKE_RATIO_MAX", 1.8),

        # ---- additional thresholds ----
        "vol_min": _get_env_float("SGRT_VOL_MIN", 0.003),
        "expo_frac": _get_env_float("SGRT_EXPO_FRAC", 0.25),

        # ---- ADX relative gate ----
        "adx_min": _get_env_float("SGRT_ADX_MIN", 0.0),
        "adx_rel_win": _get_env_int("SGRT_ADX_REL_WIN", 168),
        "adx_rel_min": _get_env_float("SGRT_ADX_REL_MIN", 1.0),
        "adx_mode": _get_env_str("SGRT_ADX_MODE", "soft").lower(),
        "adx_soft_penalty": _get_env_float("SGRT_ADX_SOFT_PENALTY", 0.10),
    }

    # Validate config (raises ValueError on bad values)
    _validate_cfg(cfg)

    # Optional: log config once per process (set SGRT_LOG_CFG=1 to enable)
    global _cfg_logged
    if not _cfg_logged and _get_env_str("SGRT_LOG_CFG", "") == "1":
        _log_effective_cfg(cfg)
        _cfg_logged = True

    return cfg


# -------------------------
# Column normalization
# -------------------------

def _normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """
    Accepts common schemas:
      Timestamp/Open/High/Low/Close/Volume
    or lowercase variants.
    Produces columns: Timestamp, Open, High, Low, Close, Volume.
    """
    if df is None or df.empty:
        return df

    cols = {c: c for c in df.columns}
    lower_map = {c.lower(): c for c in df.columns}

    def pick(*names: str) -> Optional[str]:
        for n in names:
            if n in cols:
                return cols[n]
            if n.lower() in lower_map:
                return lower_map[n.lower()]
        return None

    ts = pick("Timestamp", "timestamp", "time", "datetime", "date")
    op = pick("Open", "open", "o")
    hi = pick("High", "high", "h")
    lo = pick("Low", "low", "l")
    cl = pick("Close", "close", "c")
    vol = pick("Volume", "volume", "v")

    missing = [
        k for k, v in [
            ("Timestamp", ts), ("Open", op), ("High", hi),
            ("Low", lo), ("Close", cl), ("Volume", vol),
        ] if v is None
    ]
    if missing:
        return df

    out = df[[ts, op, hi, lo, cl, vol]].copy()
    out.columns = ["Timestamp", "Open", "High", "Low", "Close", "Volume"]

    try:
        out["Timestamp"] = pd.to_datetime(out["Timestamp"], utc=True, errors="coerce")
    except Exception:
        pass

    return out


# -------------------------
# Core computations
# -------------------------

@dataclass(frozen=True)
class _Computed:
    price: float
    ema_fast: float
    ema_slow: float
    atr: float
    atr_pct_raw: float
    atr_pct_s: float
    trend_strength_raw: float
    trend_strength_s: float
    adx: float
    adx_rel: float
    spike_ratio: float


def _compute_indicators(df: pd.DataFrame, *, closed_only: bool = True) -> Tuple[Optional[_Computed], Optional[str], Dict[str, Any]]:
    """
    Compute indicators for strategy decision.

    TIMELINE SAFETY: If closed_only=True, drops the last row before computing.
    """
    meta: Dict[str, Any] = {
        "closed_only": closed_only,
        "dropped_last_row": False,
    }

    df = _normalize_ohlcv(df)
    needed = ["Close", "High", "Low"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        return None, f"invalid_input:missing_cols={missing}", meta

    if ta is None:
        return None, "pandas_ta_unavailable", meta

    # Apply closed_only: drop last row
    if closed_only and len(df) > 1:
        df = df.iloc[:-1].copy()
        meta["dropped_last_row"] = True

    cfg = _get_env_cfg()

    # Check minimum history
    min_rows = max(int(cfg["ema_slow"]), int(cfg["atr_len"]), int(cfg["adx_len"])) + 10
    if len(df) < min_rows:
        return None, f"insufficient_history (need {min_rows}, have {len(df)})", meta

    close = df["Close"].astype(float)
    high = df["High"].astype(float)
    low = df["Low"].astype(float)

    price = float(close.iloc[-1])
    if not (price > 0):
        return None, "bad_price", meta

    ema_fast = ta.ema(close, length=int(cfg["ema_fast"]))
    ema_slow = ta.ema(close, length=int(cfg["ema_slow"]))
    atr = ta.atr(high, low, close, length=int(cfg["atr_len"]))

    if ema_fast is None or ema_slow is None or atr is None:
        return None, "indicator_unavailable", meta

    # ADX
    adx_df = ta.adx(high, low, close, length=int(cfg["adx_len"]))
    adx_col = None
    if adx_df is not None and isinstance(adx_df, pd.DataFrame):
        for c in adx_df.columns:
            if c.upper().startswith("ADX"):
                adx_col = c
                break
    adx = adx_df[adx_col] if (adx_df is not None and adx_col) else None
    if adx is None:
        adx = pd.Series([float("nan")] * len(df), index=df.index)

    atr_pct_raw = (atr / close).astype(float)

    vol_win = max(int(cfg.get("vol_win", 24)), 1)
    atr_pct_s = atr_pct_raw.rolling(window=vol_win, min_periods=1).mean()

    spread = (ema_fast - ema_slow).abs().astype(float)
    trend_strength_raw = (spread / atr.replace(0, pd.NA)).astype(float)

    trend_win = max(int(cfg.get("trend_win", 24)), 1)
    trend_strength_s = trend_strength_raw.rolling(window=trend_win, min_periods=1).mean()

    denom = atr_pct_s.replace(0, pd.NA)
    spike_ratio = (atr_pct_raw / denom).astype(float).fillna(1.0)

    adx_rel_win = max(int(cfg.get("adx_rel_win", 168)), 1)
    adx_med = adx.rolling(window=adx_rel_win, min_periods=1).median().replace(0, pd.NA)
    adx_rel = (adx / adx_med).astype(float).fillna(1.0)

    c = _Computed(
        price=price,
        ema_fast=_safe_last(ema_fast),
        ema_slow=_safe_last(ema_slow),
        atr=_safe_last(atr),
        atr_pct_raw=_safe_last(atr_pct_raw),
        atr_pct_s=_safe_last(atr_pct_s),
        trend_strength_raw=_safe_last(trend_strength_raw),
        trend_strength_s=_safe_last(trend_strength_s),
        adx=_safe_last(adx),
        adx_rel=_safe_last(adx_rel),
        spike_ratio=_safe_last(spike_ratio),
    )

    meta.update(
        {
            "cfg": cfg,
            "price": c.price,
            "ema_fast": c.ema_fast,
            "ema_slow": c.ema_slow,
            "atr": c.atr,
            "atr_pct_raw": c.atr_pct_raw,
            "atr_pct_s": c.atr_pct_s,
            "trend_strength_raw": c.trend_strength_raw,
            "trend_strength_s": c.trend_strength_s,
            "adx": c.adx,
            "adx_rel": c.adx_rel,
            "spike_ratio": c.spike_ratio,
        }
    )

    return c, None, meta


# -------------------------
# Intent mapping
# -------------------------

def _trend_bucket_conf(trend_strength: float) -> float:
    if trend_strength >= 0.60:
        return 0.75
    if trend_strength >= 0.40:
        return 0.65
    return 0.55


def _extract_p_long(ctx: Any) -> Optional[float]:
    """
    Extract p_long confirmation from context.

    Supports:
        - ctx.get("p_long")
        - ctx.get("meta", {}).get("p_long")
        - ctx.p_long attribute
    """
    try:
        if isinstance(ctx, dict):
            v = ctx.get("p_long", None)
            if v is None:
                v = ctx.get("meta", {}).get("p_long", None)
        else:
            v = getattr(ctx, "p_long", None)
        if v is None:
            return None
        pv = float(v)
        if pv != pv:
            return None
        return _clamp(pv, 0.0, 1.0)
    except Exception:
        return None


def _intent_from_values(values: Dict[str, Any], ctx: Any = None, regime_state: Any = None) -> Dict[str, Any]:
    """
    Map computed values to strategy intent.

    Args:
        values: Dict of computed indicator values
        ctx: Optional context with p_long confirmation
        regime_state: Optional RegimeState from Layer 1

    Returns:
        Dict with action, confidence, desired_exposure_frac, horizon_hours, reason, meta
    """
    cfg = values.get("cfg") or _get_env_cfg()

    trend_min = float(cfg.get("trend_min", _get_env_float("SGRT_TREND_MIN", 0.25)))
    vol_min = float(cfg.get("vol_min", _get_env_float("SGRT_VOL_MIN", 0.003)))
    vol_max = float(cfg.get("vol_max", _get_env_float("SGRT_VOL_MAX", 0.020)))
    vol_exit_max = float(cfg.get("vol_exit_max", _get_env_float("SGRT_VOL_EXIT_MAX", 0.030)))
    horizon_h = int(cfg.get("horizon_h", _get_env_int("SGRT_HORIZON_H", 36)))
    expo_frac = float(cfg.get("expo_frac", _get_env_float("SGRT_EXPO_FRAC", 0.25)))

    spike_ratio_max = float(cfg.get("spike_ratio_max", _get_env_float("SGRT_SPIKE_RATIO_MAX", 1.8)))

    adx_min = float(cfg.get("adx_min", _get_env_float("SGRT_ADX_MIN", 0.0)))
    adx_rel_min = float(cfg.get("adx_rel_min", _get_env_float("SGRT_ADX_REL_MIN", 1.0)))
    adx_mode = str(cfg.get("adx_mode", _get_env_str("SGRT_ADX_MODE", "soft"))).lower()
    adx_soft_penalty = float(cfg.get("adx_soft_penalty", _get_env_float("SGRT_ADX_SOFT_PENALTY", 0.10)))

    p_long_min = _get_env_float("SGRT_P_LONG_MIN", 0.55)
    boost_60 = _get_env_float("SGRT_P_LONG_BOOST_60", 0.05)
    boost_70 = _get_env_float("SGRT_P_LONG_BOOST_70", 0.10)
    p_long = _extract_p_long(ctx)

    ema_fast = _safe_float(values.get("ema_fast"))
    ema_slow = _safe_float(values.get("ema_slow"))
    atr_pct_raw = _safe_float(values.get("atr_pct_raw"))
    atr_pct_s = _safe_float(values.get("atr_pct_s"))
    trend_s = _safe_float(values.get("trend_strength_s"))
    adx = _safe_float(values.get("adx"))
    adx_rel = _safe_float(values.get("adx_rel"))
    spike_ratio = _safe_float(values.get("spike_ratio"))

    # Build meta dict
    base_meta = {**values, "p_long": p_long}

    # Include regime state if available
    if regime_state is not None:
        if hasattr(regime_state, "to_dict"):
            base_meta["regime_state"] = regime_state.to_dict()
        elif isinstance(regime_state, dict):
            base_meta["regime_state"] = regime_state

    # Check for PANIC regime from Layer 1
    if regime_state is not None:
        regime_label = None
        if hasattr(regime_state, "label"):
            regime_label = regime_state.label
        elif isinstance(regime_state, dict):
            regime_label = regime_state.get("label")

        if regime_label == "PANIC":
            return {
                "action": "EXIT_LONG",
                "confidence": 0.99,
                "desired_exposure_frac": expo_frac,
                "horizon_hours": horizon_h,
                "reason": "exit_regime_panic",
                "meta": {**base_meta, "exit_kind": "panic"},
            }

    # EXIT rules
    if (ema_fast == ema_fast and ema_slow == ema_slow) and (ema_fast < ema_slow):
        return {
            "action": "EXIT_LONG",
            "confidence": 0.99,
            "desired_exposure_frac": expo_frac,
            "horizon_hours": horizon_h,
            "reason": "exit_trend_break",
            "meta": base_meta,
        }

    if atr_pct_raw == atr_pct_raw and atr_pct_raw > vol_exit_max:
        return {
            "action": "EXIT_LONG",
            "confidence": 0.99,
            "desired_exposure_frac": expo_frac,
            "horizon_hours": horizon_h,
            "reason": "exit_vol_panic",
            "meta": {**base_meta, "exit_kind": "panic"},
        }

    # Entry preconditions
    trend_ok = (
        (ema_fast == ema_fast and ema_slow == ema_slow)
        and (ema_fast > ema_slow)
        and (trend_s == trend_s)
        and (trend_s > trend_min)
    )
    vol_ok = (atr_pct_s == atr_pct_s) and (vol_min <= atr_pct_s <= vol_max)
    spike_ok = (spike_ratio != spike_ratio) or (spike_ratio <= spike_ratio_max)

    # Regime check: only enter if regime is trend-friendly
    regime_entry_ok = True
    if regime_state is not None:
        regime_label = None
        if hasattr(regime_state, "label"):
            regime_label = regime_state.label
        elif isinstance(regime_state, dict):
            regime_label = regime_state.get("label")

        # Allow entry only in TREND_UP, TREND_DOWN (for short), or normal vol conditions
        # Block entry in VOL_EXPANSION, VOL_COMPRESSION, PANIC
        if regime_label in ("VOL_EXPANSION", "VOL_COMPRESSION", "PANIC"):
            regime_entry_ok = False

    # ADX evaluation
    # "off" mode: bypass ADX checks entirely
    if adx_mode == "off":
        adx_ok = True
        adx_abs_ok = True
        adx_rel_ok = True
    else:
        adx_abs_ok = (adx_min <= 0.0) or (adx == adx and adx >= adx_min)
        adx_rel_ok = (adx_rel_min <= 0.0) or (adx_rel == adx_rel and adx_rel >= adx_rel_min)
        adx_ok = adx_abs_ok and adx_rel_ok

    conf = _trend_bucket_conf(trend_s if trend_s == trend_s else 0.0)

    # Optional p_long confirmation (doesn't veto)
    if p_long is not None:
        if p_long < p_long_min:
            conf -= 0.05
        elif p_long >= 0.70:
            conf += boost_70
        elif p_long >= 0.60:
            conf += boost_60

    # ADX handling (soft/hard)
    adx_note = "ok"
    if not adx_ok:
        if adx_mode == "hard":
            return {
                "action": "FLAT",
                "confidence": _clamp(conf, 0.0, 0.99),
                "desired_exposure_frac": expo_frac,
                "horizon_hours": horizon_h,
                "reason": "no_entry_adx_gate",
                "meta": {**base_meta, "adx_ok": False, "adx_mode": adx_mode},
            }
        conf -= abs(adx_soft_penalty)
        adx_note = "soft_penalty"

    conf = _clamp(conf, 0.0, 0.99)

    if trend_ok and vol_ok and spike_ok and regime_entry_ok:
        return {
            "action": "ENTER_LONG",
            "confidence": conf,
            "desired_exposure_frac": expo_frac,
            "horizon_hours": horizon_h,
            "reason": f"enter_trend_vol_{adx_note}",
            "meta": {
                **base_meta,
                "trend_ok": bool(trend_ok),
                "vol_ok": bool(vol_ok),
                "spike_ok": bool(spike_ok),
                "adx_ok": bool(adx_ok),
                "adx_mode": adx_mode,
                "regime_entry_ok": bool(regime_entry_ok),
            },
        }

    return {
        "action": "FLAT",
        "confidence": conf,
        "desired_exposure_frac": expo_frac,
        "horizon_hours": horizon_h,
        "reason": "no_entry_filters",
        "meta": {
            **base_meta,
            "trend_ok": bool(trend_ok),
            "vol_ok": bool(vol_ok),
            "spike_ok": bool(spike_ok),
            "adx_ok": bool(adx_ok),
            "adx_mode": adx_mode,
            "regime_entry_ok": bool(regime_entry_ok),
        },
    }


def generate_intent_from_values(*args, ctx: Any = None, **kwargs) -> Dict[str, Any]:
    """
    Backtest harness compatibility.

    Supported call styles:
      A) generate_intent_from_values(values_dict, ctx=ctx)
      B) generate_intent_from_values(ema_fast=..., ema_slow=..., ..., cfg=cfg, ctx=ctx)
      C) generate_intent_from_values(candidates={...}, ctx=ctx)   <-- hardened
    """
    # Positional dict path
    if args:
        if len(args) != 1 or not isinstance(args[0], dict):
            raise TypeError("generate_intent_from_values expected either a single dict positional arg or pure kwargs")
        values = dict(args[0])
        if "cfg" not in values:
            values["cfg"] = _get_env_cfg()
        return _intent_from_values(values, ctx=ctx)

    # Harden: accept nested candidates dict
    if "candidates" in kwargs and isinstance(kwargs["candidates"], dict):
        merged = dict(kwargs["candidates"])
        # allow explicit kwargs to override candidates
        for k, v in kwargs.items():
            if k == "candidates":
                continue
            merged[k] = v
        values = merged
    else:
        values = dict(kwargs)

    if "cfg" not in values or values["cfg"] is None:
        values["cfg"] = _get_env_cfg()

    return _intent_from_values(values, ctx=ctx)


def generate_intent(df: pd.DataFrame, ctx: Any, *, closed_only: bool = True) -> Dict[str, Any]:
    """
    Generate strategy intent from OHLCV DataFrame.

    This is the primary entry point for Signal Generator integration.

    Args:
        df: DataFrame with OHLCV data
        ctx: StrategyContext from SG (dict-like, may contain p_long)
        closed_only: If True (default), drop the last row before computing.
                     This ensures decisions based on CLOSED candles only.

    Returns:
        Dict with keys: action, confidence, desired_exposure_frac, horizon_hours, reason, meta

    Timeline Safety:
        When closed_only=True, the final row is ALWAYS dropped regardless of
        whether it represents a closed or forming candle.
    """
    # Get regime classification from Layer 1
    regime_state = None
    if _REGIME_ENGINE_AVAILABLE:
        try:
            regime_state = classify_regime(df, closed_only=closed_only)
        except Exception as e:
            # Don't fail strategy if regime engine fails
            regime_state = None

    # Compute strategy indicators
    computed, err, meta = _compute_indicators(df, closed_only=closed_only)

    if err is not None or computed is None:
        # Handle NaN/insufficient history case
        reason = err or "compute_failed"
        nan_fields = []
        if "NAN" in reason.upper() or "insufficient" in reason.lower():
            nan_fields = ["indicators"]

        return {
            "action": "HOLD",
            "confidence": 0.0,
            "desired_exposure_frac": _get_env_float("SGRT_EXPO_FRAC", 0.25),
            "horizon_hours": _get_env_int("SGRT_HORIZON_H", 36),
            "reason": "INSUFFICIENT_HISTORY_OR_NAN" if nan_fields else reason,
            "meta": {
                **meta,
                "nan_fields": nan_fields,
                "regime_state": regime_state.to_dict() if regime_state and hasattr(regime_state, "to_dict") else None,
            },
        }

    values = {
        "price": computed.price,
        "ema_fast": computed.ema_fast,
        "ema_slow": computed.ema_slow,
        "atr": computed.atr,
        "atr_pct_raw": computed.atr_pct_raw,
        "atr_pct_s": computed.atr_pct_s,
        "trend_strength_raw": computed.trend_strength_raw,
        "trend_strength_s": computed.trend_strength_s,
        "adx": computed.adx,
        "adx_rel": computed.adx_rel,
        "spike_ratio": computed.spike_ratio,
        "cfg": meta.get("cfg", _get_env_cfg()),
        "closed_only": closed_only,
        "dropped_last_row": meta.get("dropped_last_row", False),
    }

    return _intent_from_values(values, ctx=ctx, regime_state=regime_state)


# ─────────────────────────────────────────────────────────────────────────────
# Module info for verification
# ─────────────────────────────────────────────────────────────────────────────

__file_info__ = {
    "module": "research.strategies.sg_regime_trend_v1",
    "layer": 2,
    "description": "Strategy v1 - pluggable, deterministic, per-regime alpha module",
    "depends_on": ["research.regime.regime_engine"],
}
