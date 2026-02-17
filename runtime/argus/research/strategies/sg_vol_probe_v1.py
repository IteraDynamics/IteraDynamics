"""
sg_vol_probe_v1.py

Strategy B — Volatility Compression Breakout (Probe)

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


ENV_BB_LEN = "SG_VOL_PROBE_BB_LEN"                 # default 20
ENV_BB_STD = "SG_VOL_PROBE_BB_STD"                 # default 2.0
ENV_WIDTH_EXPAND_FACTOR = "SG_VOL_PROBE_WIDTH_EXPAND_FACTOR"  # default 1.20
ENV_HORIZON_HOURS = "SG_VOL_PROBE_HORIZON_HOURS"   # default 18
ENV_EXPO_FRAC = "SG_VOL_PROBE_EXPOSURE_FRAC"       # default 0.25


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


def _bollinger(close: pd.Series, length: int, n_std: float) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    mid = close.rolling(window=length, min_periods=length).mean()
    sd = close.rolling(window=length, min_periods=length).std(ddof=0)
    upper = mid + (n_std * sd)
    lower = mid - (n_std * sd)
    width = (upper - lower) / mid.replace(0.0, np.nan)
    return mid, upper, lower, width


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

    bb_len = _get_env_int(ENV_BB_LEN, 20)
    bb_std = _get_env_float(ENV_BB_STD, 2.0)
    width_expand_factor = _get_env_float(ENV_WIDTH_EXPAND_FACTOR, 1.20)
    horizon_hours = _get_env_int(ENV_HORIZON_HOURS, 18)
    desired_exposure_frac = _get_env_float(ENV_EXPO_FRAC, 0.25)

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
        "strategy": "sg_vol_probe_v1",
        "regime_state": regime_state,
        "closed_only": bool(closed_only),
        "dropped_last_row": bool(dropped_last_row),
        "regime_meta": regime_meta,
        "regime_error": regime_err,
        "params": {
            "bb_len": bb_len,
            "bb_std": bb_std,
            "width_expand_factor": width_expand_factor,
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

    need = bb_len + 3
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
    mid, upper, lower, width = _bollinger(close, bb_len, bb_std)

    last_i = df_used.index[-1]
    prev_i = df_used.index[-2]

    px = float(close.loc[last_i]) if np.isfinite(close.loc[last_i]) else np.nan
    up = float(upper.loc[last_i]) if np.isfinite(upper.loc[last_i]) else np.nan
    md = float(mid.loc[last_i]) if np.isfinite(mid.loc[last_i]) else np.nan
    w = float(width.loc[last_i]) if np.isfinite(width.loc[last_i]) else np.nan
    w_prev = float(width.loc[prev_i]) if np.isfinite(width.loc[prev_i]) else np.nan

    signal = {"price": px, "bb_mid": md, "bb_upper": up, "bb_width": w, "bb_width_prev": w_prev}

    if not (np.isfinite(px) and np.isfinite(up) and np.isfinite(md) and np.isfinite(w) and np.isfinite(w_prev)):
        return _action_dict(
            action="HOLD",
            confidence=0.0,
            desired_exposure_frac=0.0,
            horizon_hours=horizon_hours,
            reason="nan_guard(latest_indicators_invalid)",
            meta={**meta, "signal": signal},
        )

    if regime_state == "PANIC":
        return _action_dict(
            action="EXIT_LONG",
            confidence=1.0,
            desired_exposure_frac=0.0,
            horizon_hours=horizon_hours,
            reason="regime_panic_force_exit",
            meta={**meta, "signal": signal},
        )

    if regime_state in ("TREND_UP", "TREND_DOWN"):
        return _action_dict(
            action="EXIT_LONG",
            confidence=0.9,
            desired_exposure_frac=0.0,
            horizon_hours=horizon_hours,
            reason=f"avoid_trend_regime({regime_state})",
            meta={**meta, "signal": signal},
        )

    if regime_state != "VOL_COMPRESSION":
        if regime_state == "VOL_EXPANSION" and (w < w_prev):
            return _action_dict(
                action="EXIT_LONG",
                confidence=0.75,
                desired_exposure_frac=0.0,
                horizon_hours=horizon_hours,
                reason="vol_expansion_exhaustion(width_contracting)",
                meta={**meta, "signal": signal},
            )
        return _action_dict(
            action="EXIT_LONG",
            confidence=0.85,
            desired_exposure_frac=0.0,
            horizon_hours=horizon_hours,
            reason=f"regime_flip({regime_state})",
            meta={**meta, "signal": signal},
        )

    width_expanding = (w_prev > 0) and (w / w_prev >= width_expand_factor)
    breakout = px > up

    if width_expanding and breakout:
        ratio = (w / w_prev) if w_prev > 0 else 0.0
        if ratio < (width_expand_factor * 1.10):
            conf, bucket = 0.55, "modest_expand"
        elif ratio < (width_expand_factor * 1.30):
            conf, bucket = 0.70, "clean_expand"
        else:
            conf, bucket = 0.82, "strong_expand"

        return _action_dict(
            action="ENTER_LONG",
            confidence=conf,
            desired_exposure_frac=desired_exposure_frac,
            horizon_hours=horizon_hours,
            reason=f"compression_breakout(bucket={bucket})",
            meta={**meta, "signal": {**signal, "ratio": ratio, "bucket": bucket}},
        )

    return _action_dict(
        action="HOLD",
        confidence=0.20,
        desired_exposure_frac=0.0,
        horizon_hours=horizon_hours,
        reason=f"no_breakout(width_expanding={bool(width_expanding)}, breakout={bool(breakout)})",
        meta={**meta, "signal": signal},
    )
