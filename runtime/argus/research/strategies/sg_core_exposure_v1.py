"""
sg_core_exposure_v1.py

Itera Dynamics — Official Research Phase
Core Strategy v1: Macro Regime + Volatility-Scaled Exposure (Long-Only BTC)

Mission alignment:
- Participate in BTC upside while materially reducing catastrophic drawdowns.
- Deterministic, closed-only, no side effects, no ML, no new architecture.

Rules honored:
- Strategy calls Layer 1 directly:
    reg = classify_regime(df0, closed_only=closed_only)
- Strategy does NOT touch Prime, wallet, broker, files, state.
- Strategy returns strict Layer 2 contract:
    action, confidence, desired_exposure_frac, horizon_hours, reason, meta
- closed_only=True default and required.
- No forming candle usage: we mirror Layer 1’s dropped_last_row.

High-level design:
1) Long bias only in TREND_UP (core exposure)
2) Flat in TREND_DOWN (explicit exit)
3) Force exit in PANIC
4) Reduced or zero new exposure in VOL_EXPANSION (do not add)
5) Limited anticipatory exposure in VOL_COMPRESSION (small optional long)
6) Volatility-scaled exposure: scale inversely to ATR% (ATR / price)
7) Confidence from trend strength + regime

Notes:
- This strategy is not optimized for frequency or breakout timing.
- This strategy tests whether regime + volatility scaling improves risk-adjusted exposure.

Data schema robustness:
- Accepts capitalized OHLCV from flight recorder (Open/High/Low/Close/Volume).
- Adds lowercase aliases open/high/low/close/volume deterministically.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd

from research.regime import classify_regime


# ---------------------------
# Environment knobs (sane defaults)
# ---------------------------

ENV_EMA_FAST = "SG_CORE_EMA_FAST"          # default 20
ENV_EMA_SLOW = "SG_CORE_EMA_SLOW"          # default 50
ENV_ATR_LEN = "SG_CORE_ATR_LEN"            # default 14

ENV_HORIZON_HOURS = "SG_CORE_HORIZON_HOURS"  # default 48

# Exposure caps per regime
ENV_MAX_EXPO_TREND_UP = "SG_CORE_MAX_EXPO_TREND_UP"          # default 0.75
ENV_BASE_EXPO_COMPRESSION = "SG_CORE_BASE_EXPO_COMPRESSION"  # default 0.12
ENV_MAX_EXPO_COMPRESSION = "SG_CORE_MAX_EXPO_COMPRESSION"    # default 0.20

# Volatility scaling target
ENV_TARGET_ATR_PCT = "SG_CORE_TARGET_ATR_PCT"  # default 0.010 (1.0%)
ENV_ATR_PCT_FLOOR = "SG_CORE_ATR_PCT_FLOOR"    # default 0.003 (0.3%) prevents huge leverage-like scaling
ENV_SCALE_CAP = "SG_CORE_SCALE_CAP"            # default 1.50 cap on scaling multiplier

# Micro-trend and confidence
ENV_TREND_STRENGTH_WEAK = "SG_CORE_TREND_STRENGTH_WEAK"       # default 0.20
ENV_TREND_STRENGTH_STRONG = "SG_CORE_TREND_STRENGTH_STRONG"   # default 0.60


# ---------------------------
# Helpers
# ---------------------------

def _get_env_int(name: str, default: int) -> int:
    try:
        v = int(str(os.getenv(name, default)).strip())
        return v if v > 0 else default
    except Exception:
        return default


def _get_env_float(name: str, default: float) -> float:
    try:
        v = float(str(os.getenv(name, default)).strip())
        # Clamp obvious fraction knobs if named like one
        if "EXPO" in name.upper() or "FRAC" in name.upper() or "CAP" in name.upper():
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


def _clamp(x: float, lo: float, hi: float) -> float:
    return float(min(hi, max(lo, x)))


def _trend_strength_norm(ema_fast: float, ema_slow: float, atr: float, price: float) -> float:
    """
    Normalized trend strength:
        (ema_fast - ema_slow) / denom
    denom uses ATR if available, else price as fallback.
    """
    spread = float(ema_fast - ema_slow)
    if np.isfinite(atr) and atr > 0:
        denom = float(atr)
    else:
        denom = float(price) if (np.isfinite(price) and price > 0) else 1.0
    return spread / denom


def _bucket_trend_strength(ts: float, weak: float, strong: float) -> Tuple[float, str]:
    """
    Map trend strength to a confidence component (0..1) and a label bucket.
    """
    if not np.isfinite(ts):
        return 0.0, "nan"

    if ts <= 0:
        return 0.10, "negative"
    if ts < weak:
        return 0.35, "weak"
    if ts < strong:
        return 0.60, "moderate"
    return 0.80, "strong"


def _vol_scale_multiplier(atr_pct: float, target_atr_pct: float, atr_floor: float, scale_cap: float) -> Tuple[float, float]:
    """
    Volatility scaling:
        atr_pct = ATR / price
        mult = target_atr_pct / max(atr_pct, atr_floor)
    Then clamp mult to [0, scale_cap].

    Returns:
        (mult_clamped, atr_pct_eff)
    """
    atr_pct_eff = float(max(atr_floor, atr_pct)) if np.isfinite(atr_pct) else float(atr_floor)
    raw = float(target_atr_pct) / atr_pct_eff if atr_pct_eff > 0 else 0.0
    mult = _clamp(raw, 0.0, float(scale_cap))
    return mult, atr_pct_eff


def _action_dict(
    *,
    action: str,
    confidence: float,
    desired_exposure_frac: float,
    horizon_hours: int,
    reason: str,
    meta: Dict[str, Any],
) -> Dict[str, Any]:
    # IMPORTANT: Prime constructs StrategyIntent(**dict) — do not add extra top-level keys.
    return {
        "action": str(action),
        "confidence": float(confidence),
        "desired_exposure_frac": float(desired_exposure_frac),
        "horizon_hours": int(horizon_hours),
        "reason": str(reason),
        "meta": meta,
    }


# ---------------------------
# Strategy entrypoint
# ---------------------------

def generate_intent(df: pd.DataFrame, ctx: Dict[str, Any], *, closed_only: bool = True) -> Dict[str, Any]:
    """
    Layer 2 contract entrypoint.

    ctx is accepted by signature but unused (no state dependence).
    """
    df0 = _safe_copy_df(df)
    df0 = _normalize_ohlcv_columns(df0)

    # Params
    ema_fast_n = _get_env_int(ENV_EMA_FAST, 20)
    ema_slow_n = _get_env_int(ENV_EMA_SLOW, 50)
    atr_n = _get_env_int(ENV_ATR_LEN, 14)

    horizon_hours = _get_env_int(ENV_HORIZON_HOURS, 48)

    max_expo_trend_up = _get_env_float(ENV_MAX_EXPO_TREND_UP, 0.75)
    base_expo_compression = _get_env_float(ENV_BASE_EXPO_COMPRESSION, 0.12)
    max_expo_compression = _get_env_float(ENV_MAX_EXPO_COMPRESSION, 0.20)

    target_atr_pct = _get_env_float(ENV_TARGET_ATR_PCT, 0.010)
    atr_floor = _get_env_float(ENV_ATR_PCT_FLOOR, 0.003)
    scale_cap = _get_env_float(ENV_SCALE_CAP, 1.50)

    ts_weak = _get_env_float(ENV_TREND_STRENGTH_WEAK, 0.20)
    ts_strong = _get_env_float(ENV_TREND_STRENGTH_STRONG, 0.60)

    # Layer 1 regime classification (authoritative)
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

    # Mirror Layer 1 closed-only timeline (no double-dropping)
    df_used = df0.iloc[:-1] if dropped_last_row else df0

    meta: Dict[str, Any] = {
        "strategy": "sg_core_exposure_v1",
        "closed_only": bool(closed_only),
        "regime_state": regime_state,
        "dropped_last_row": bool(dropped_last_row),
        "regime_meta": regime_meta,
        "regime_error": regime_err,
        "params": {
            "ema_fast": ema_fast_n,
            "ema_slow": ema_slow_n,
            "atr_len": atr_n,
            "horizon_hours": horizon_hours,
            "max_expo_trend_up": max_expo_trend_up,
            "base_expo_compression": base_expo_compression,
            "max_expo_compression": max_expo_compression,
            "target_atr_pct": target_atr_pct,
            "atr_floor": atr_floor,
            "scale_cap": scale_cap,
            "trend_strength_weak": ts_weak,
            "trend_strength_strong": ts_strong,
        },
    }

    # Fail-closed if regime unavailable
    if regime_state is None:
        return _action_dict(
            action="HOLD",
            confidence=0.0,
            desired_exposure_frac=0.0,
            horizon_hours=horizon_hours,
            reason="missing_regime_state_fail_closed",
            meta=meta,
        )

    # Schema guard
    if df_used is None or "close" not in df_used.columns:
        return _action_dict(
            action="HOLD",
            confidence=0.0,
            desired_exposure_frac=0.0,
            horizon_hours=horizon_hours,
            reason="missing_close_column",
            meta=meta,
        )

    # History guard
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

    # Indicators (deterministic)
    close = df_used["close"].astype(float)
    ema_fast = _ema(close, ema_fast_n)
    ema_slow = _ema(close, ema_slow_n)
    atr = _atr(df_used, atr_n)

    last_i = df_used.index[-1]
    px = float(close.loc[last_i]) if np.isfinite(close.loc[last_i]) else np.nan
    ef = float(ema_fast.loc[last_i]) if np.isfinite(ema_fast.loc[last_i]) else np.nan
    es = float(ema_slow.loc[last_i]) if np.isfinite(ema_slow.loc[last_i]) else np.nan
    a = float(atr.loc[last_i]) if np.isfinite(atr.loc[last_i]) else np.nan

    atr_pct = (a / px) if (np.isfinite(a) and np.isfinite(px) and px > 0) else np.nan

    # Trend strength + bucket for confidence
    ts = _trend_strength_norm(ef, es, a, px) if (np.isfinite(ef) and np.isfinite(es) and np.isfinite(px)) else np.nan
    conf_trend, ts_bucket = _bucket_trend_strength(ts, ts_weak, ts_strong)

    # Regime confidence component (simple deterministic mapping)
    # (We intentionally don't assume regime engine exposes a numeric confidence.)
    if regime_state == "TREND_UP":
        conf_regime = 0.75
    elif regime_state == "VOL_COMPRESSION":
        conf_regime = 0.45
    elif regime_state == "VOL_EXPANSION":
        conf_regime = 0.25
    elif regime_state == "CHOP":
        conf_regime = 0.20
    elif regime_state == "TREND_DOWN":
        conf_regime = 0.10
    elif regime_state == "PANIC":
        conf_regime = 1.00
    else:
        conf_regime = 0.20

    # Volatility scaling multiplier
    mult, atr_pct_eff = _vol_scale_multiplier(float(atr_pct) if np.isfinite(atr_pct) else np.nan, target_atr_pct, atr_floor, scale_cap)

    signal = {
        "price": px,
        "ema_fast": ef,
        "ema_slow": es,
        "atr": a,
        "atr_pct": float(atr_pct) if np.isfinite(atr_pct) else None,
        "atr_pct_eff": float(atr_pct_eff),
        "vol_mult": float(mult),
        "trend_strength": float(ts) if np.isfinite(ts) else None,
        "trend_bucket": ts_bucket,
    }

    meta = {**meta, "signal": signal}

    # NaN guard (fail-closed)
    if not (np.isfinite(px) and np.isfinite(ef) and np.isfinite(es)):
        return _action_dict(
            action="HOLD",
            confidence=0.0,
            desired_exposure_frac=0.0,
            horizon_hours=horizon_hours,
            reason="nan_guard(latest_indicators_invalid)",
            meta=meta,
        )

    # ---------------------------
    # Regime policy
    # ---------------------------

    # 3) PANIC => force exit
    if regime_state == "PANIC":
        return _action_dict(
            action="EXIT_LONG",
            confidence=1.0,
            desired_exposure_frac=0.0,
            horizon_hours=horizon_hours,
            reason="regime_panic_force_exit",
            meta=meta,
        )

    # 2) TREND_DOWN => flat
    if regime_state == "TREND_DOWN":
        return _action_dict(
            action="EXIT_LONG",
            confidence=0.95,
            desired_exposure_frac=0.0,
            horizon_hours=horizon_hours,
            reason="regime_trend_down_flat",
            meta=meta,
        )

    # 4) VOL_EXPANSION => reduce or zero new exposure (do not add)
    # With no position state in strategy, we express "do not add" as HOLD + desired_exposure_frac=0.
    # Prime governors can manage exits independently if needed.
    if regime_state == "VOL_EXPANSION":
        # If micro-trend is sharply negative, we can optionally recommend exit (still deterministic).
        if ef < es and conf_trend <= 0.35:
            return _action_dict(
                action="EXIT_LONG",
                confidence=0.75,
                desired_exposure_frac=0.0,
                horizon_hours=horizon_hours,
                reason="vol_expansion_and_microtrend_negative_exit",
                meta=meta,
            )
        return _action_dict(
            action="HOLD",
            confidence=0.25,
            desired_exposure_frac=0.0,
            horizon_hours=horizon_hours,
            reason="regime_vol_expansion_no_new_exposure",
            meta=meta,
        )

    # 5) VOL_COMPRESSION => limited anticipatory exposure (small)
    if regime_state == "VOL_COMPRESSION":
        # Only allow anticipatory long if micro-trend is not negative
        if ef < es:
            return _action_dict(
                action="HOLD",
                confidence=0.20,
                desired_exposure_frac=0.0,
                horizon_hours=horizon_hours,
                reason="vol_compression_but_microtrend_negative",
                meta=meta,
            )

        # Base small exposure scaled by vol
        expo_raw = float(base_expo_compression) * float(mult)
        expo = _clamp(expo_raw, 0.0, float(max_expo_compression))

        # Confidence blends regime + trend (kept conservative in compression)
        confidence = _clamp(0.55 * conf_regime + 0.45 * conf_trend, 0.0, 1.0)

        return _action_dict(
            action="ENTER_LONG" if expo > 0 else "HOLD",
            confidence=confidence,
            desired_exposure_frac=expo,
            horizon_hours=horizon_hours,
            reason=f"vol_compression_anticipatory_long(bucket={ts_bucket})",
            meta={**meta, "exposure": {"expo_raw": expo_raw, "expo_clamped": expo}},
        )

    # 6) Core: TREND_UP => volatility-scaled exposure (long)
    if regime_state == "TREND_UP":
        # Require basic micro-trend confirmation (EMA fast >= EMA slow)
        if ef < es:
            return _action_dict(
                action="HOLD",
                confidence=0.30,
                desired_exposure_frac=0.0,
                horizon_hours=horizon_hours,
                reason="trend_up_but_microtrend_not_confirmed",
                meta=meta,
            )

        expo_raw = float(max_expo_trend_up) * float(mult)
        expo = _clamp(expo_raw, 0.0, float(max_expo_trend_up))

        # Confidence: blend regime + trend strength
        confidence = _clamp(0.55 * conf_regime + 0.45 * conf_trend, 0.0, 1.0)

        return _action_dict(
            action="ENTER_LONG" if expo > 0 else "HOLD",
            confidence=confidence,
            desired_exposure_frac=expo,
            horizon_hours=horizon_hours,
            reason=f"trend_up_vol_scaled_exposure(bucket={ts_bucket})",
            meta={**meta, "exposure": {"expo_raw": expo_raw, "expo_clamped": expo}},
        )

    # CHOP or unknown => stay conservative (flat / no new exposure)
    if regime_state == "CHOP":
        return _action_dict(
            action="HOLD",
            confidence=0.20,
            desired_exposure_frac=0.0,
            horizon_hours=horizon_hours,
            reason="regime_chop_no_new_exposure",
            meta=meta,
        )

    return _action_dict(
        action="HOLD",
        confidence=0.15,
        desired_exposure_frac=0.0,
        horizon_hours=horizon_hours,
        reason=f"regime_unhandled({regime_state})",
        meta=meta,
    )
