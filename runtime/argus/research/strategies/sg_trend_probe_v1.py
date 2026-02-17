"""
sg_trend_probe_v1.py

Strategy A — Regime-Constrained Trend Follower (Probe)

Structural requirements:
- Regime obtained via Layer 1:
    reg = classify_regime(df0, closed_only=closed_only)
    regime_state = reg.label
    dropped_last_row = bool(reg.meta.get("dropped_last_row", False))
- No double-dropping: strategy mirrors Layer 1 drop decision:
    df_used = df0.iloc[:-1] if dropped_last_row else df0
- Return contract always includes:
  action, confidence, desired_exposure_frac, horizon_hours, reason, meta
- Deterministic indicators, no side effects, no external deps.

Schema robustness:
- Dataframes may have capitalized OHLCV (Open/High/Low/Close/Volume).
- Strategy normalizes to lowercase aliases (open/high/low/close/volume) deterministically.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd

from research.regime import classify_regime


ENV_EMA_FAST = "SG_TREND_PROBE_EMA_FAST"   # default 20
ENV_EMA_SLOW = "SG_TREND_PROBE_EMA_SLOW"   # default 50
ENV_ATR_LEN = "SG_TREND_PROBE_ATR_LEN"     # default 14
ENV_HORIZON_HOURS = "SG_TREND_PROBE_HORIZON_HOURS"  # default 36
ENV_EXPO_FRAC = "SG_TREND_PROBE_EXPOSURE_FRAC"      # default 0.65


def _get_env_int(name: str, default: int) -> int:
    try:
        v = int(os.getenv(name, str(default)).strip())
        return v if v > 0 else default
    except Exception:
        return default


def _get_env_float(name: str, default: float) -> float:
    try:
        v = float(os.getenv(name, str(default)).strip())
        if "FRAC" in name.upper():
            return float(min(1.0, max(0.0, v)))
        return float(v)
    except Exception:
        return default


def _safe_copy_df(df: pd.DataFrame) -> pd.DataFrame:
    return df.copy(deep=False)


def _normalize_ohlcv_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Deterministic, side-effect-free OHLCV normalization.
    Adds lowercase aliases: open, high, low, close, volume when found.
    Leaves original columns intact. Returns shallow copy.
    """
    if df is None or not hasattr(df, "columns"):
        return df

    cols = list(df.columns)
    lower_map = {str(c).strip().lower(): c for c in cols}

    def pick(*names: str):
        for n in names:
            if n in lower_map:
                return lower_map[n]
        return None

    src_open = pick("open", "o")
    src_high = pick("high", "h")
    src_low = pick("low", "l")
    src_close = pick("close", "c", "price", "last", "mid")
    src_vol = pick("volume", "vol", "v")

    if src_close is None:
        return df

    out = df.copy(deep=False)

    out["close"] = out[src_close].astype(float)
    if src_open is not None:
        out["open"] = out[src_open].astype(float)
    if src_high is not None:
        out["high"] = out[src_high].astype(float)
    if src_low is not None:
        out["low"] = out[src_low].astype(float)
    if src_vol is not None:
        out["volume"] = out[src_vol].astype(float)

    return out


def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False, min_periods=span).mean()


def _atr(df: pd.DataFrame, length: int) -> pd.Series:
    high = df.get("high")
    low = df.get("low")
    close = df.get("close")
    if high is None or low is None or close is None:
        return pd.Series(np.nan, index=df.index)

    prev_close = close.shift(1)
    tr = pd.concat(
        [
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    return tr.rolling(window=length, min_periods=length).mean()


def _trend_strength_bucket(ema_fast: float, ema_slow: float, atr: float, price: float) -> Tuple[float, str]:
    spread = float(ema_fast - ema_slow)
    denom = float(atr) if (atr is not None and np.isfinite(atr) and atr > 0) else float(price) if (price > 0) else 1.0
    norm = spread / denom

    if norm <= 0:
        return 0.0, "no_trend"
    if norm < 0.25:
        return 0.45, "weak"
    if norm < 0.60:
        return 0.65, "moderate"
    return 0.85, "strong"


def _action_dict(
    *,
    action: str,
    confidence: float,
    desired_exposure_frac: float,
    horizon_hours: int,
    reason: str,
    meta: Dict[str, Any],
) -> Dict[str, Any]:
    # IMPORTANT: do not add extra top-level keys; Prime constructs StrategyIntent(**dict)
    return {
        "action": str(action),
        "confidence": float(confidence),
        "desired_exposure_frac": float(desired_exposure_frac),
        "horizon_hours": int(horizon_hours),
        "reason": str(reason),
        "meta": meta,
    }


def generate_intent(df: pd.DataFrame, ctx: Dict[str, Any], *, closed_only: bool = True) -> Dict[str, Any]:
    df0 = _safe_copy_df(df)
    df0 = _normalize_ohlcv_columns(df0)

    ema_fast_n = _get_env_int(ENV_EMA_FAST, 20)
    ema_slow_n = _get_env_int(ENV_EMA_SLOW, 50)
    atr_n = _get_env_int(ENV_ATR_LEN, 14)
    horizon_hours = _get_env_int(ENV_HORIZON_HOURS, 36)
    desired_exposure_frac = _get_env_float(ENV_EXPO_FRAC, 0.65)

    regime_state = None
    dropped_last_row = False
    regime_meta = None
    regime_err = None
    try:
        reg = classify_regime(df0, closed_only=closed_only)
        regime_state = getattr(reg, "label", None)
        regime_meta = getattr(reg, "meta", None)
        dropped_last_row = bool((regime_meta or {}).get("dropped_last_row", False))
        if isinstance(regime_state, str):
            regime_state = regime_state.strip()
        else:
            regime_state = None
    except Exception as e:
        regime_err = f"{type(e).__name__}: {e}"

    df_used = df0.iloc[:-1] if dropped_last_row else df0

    meta: Dict[str, Any] = {
        "strategy": "sg_trend_probe_v1",
        "regime_state": regime_state,
        "closed_only": bool(closed_only),
        "dropped_last_row": bool(dropped_last_row),
        "regime_meta": regime_meta,
        "regime_error": regime_err,
        "params": {
            "ema_fast": ema_fast_n,
            "ema_slow": ema_slow_n,
            "atr_len": atr_n,
            "horizon_hours": horizon_hours,
            "desired_exposure_frac": desired_exposure_frac,
        },
    }

    if regime_state is None:
        return _action_dict(
            action="HOLD",
            confidence=0.0,
            desired_exposure_frac=0.0,
            horizon_hours=horizon_hours,
            reason="missing_regime_state_fail_closed",
            meta=meta,
        )

    if df_used is None or "close" not in df_used.columns:
        return _action_dict(
            action="HOLD",
            confidence=0.0,
            desired_exposure_frac=0.0,
            horizon_hours=horizon_hours,
            reason="missing_close_column",
            meta=meta,
        )

    need = max(ema_fast_n, ema_slow_n, atr_n) + 2
    if len(df_used) < need:
        return _action_dict(
            action="HOLD",
            confidence=0.0,
            desired_exposure_frac=0.0,
            horizon_hours=horizon_hours,
            reason=f"insufficient_history(len={len(df_used)}, need={need})",
            meta=meta,
        )

    close = df_used["close"].astype(float)
    ema_fast = _ema(close, ema_fast_n)
    ema_slow = _ema(close, ema_slow_n)
    atr = _atr(df_used, atr_n)

    last_i = df_used.index[-1]
    px = float(close.loc[last_i]) if np.isfinite(close.loc[last_i]) else np.nan
    ef = float(ema_fast.loc[last_i]) if np.isfinite(ema_fast.loc[last_i]) else np.nan
    es = float(ema_slow.loc[last_i]) if np.isfinite(ema_slow.loc[last_i]) else np.nan
    a = float(atr.loc[last_i]) if np.isfinite(atr.loc[last_i]) else np.nan

    meta_signal = {"ema_fast": ef, "ema_slow": es, "atr": a, "price": px}

    if not (np.isfinite(px) and np.isfinite(ef) and np.isfinite(es)):
        return _action_dict(
            action="HOLD",
            confidence=0.0,
            desired_exposure_frac=0.0,
            horizon_hours=horizon_hours,
            reason="nan_guard(latest_indicators_invalid)",
            meta={**meta, "signal": meta_signal},
        )

    if regime_state == "PANIC":
        return _action_dict(
            action="EXIT_LONG",
            confidence=1.0,
            desired_exposure_frac=0.0,
            horizon_hours=horizon_hours,
            reason="regime_panic_force_exit",
            meta={**meta, "signal": meta_signal},
        )

    if regime_state != "TREND_UP":
        return _action_dict(
            action="EXIT_LONG",
            confidence=0.9,
            desired_exposure_frac=0.0,
            horizon_hours=horizon_hours,
            reason=f"regime_not_trend_up({regime_state})",
            meta={**meta, "signal": meta_signal},
        )

    if ef > es:
        conf, bucket = _trend_strength_bucket(ef, es, a, px)
        return _action_dict(
            action="ENTER_LONG",
            confidence=conf,
            desired_exposure_frac=desired_exposure_frac,
            horizon_hours=horizon_hours,
            reason=f"trend_up_ema_state(bucket={bucket})",
            meta={**meta, "signal": {**meta_signal, "bucket": bucket}},
        )

    return _action_dict(
        action="HOLD",
        confidence=0.15,
        desired_exposure_frac=0.0,
        horizon_hours=horizon_hours,
        reason="trend_up_but_no_ema_confirmation",
        meta={**meta, "signal": meta_signal},
    )
