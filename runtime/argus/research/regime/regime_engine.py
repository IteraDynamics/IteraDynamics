"""
Layer 1: Regime Engine
======================

Authoritative, stable, shared regime classification.

TIMELINE SAFETY (non-negotiable):
- closed_only=True (default): ALWAYS drop the last row before computing indicators
- All decisions based on CLOSED candles only
- RegimeState.meta includes: "closed_only", "dropped_last_row"

REGIME LABELS (priority order):
    PANIC > VOL_EXPANSION > VOL_COMPRESSION > TREND_UP/TREND_DOWN > CHOP

NaN SAFETY:
- If indicators produce NaN due to insufficient history, return CHOP with confidence=0.0

ENVIRONMENT VARIABLES:
    REGIME_EMA_FAST=20          Fast EMA period
    REGIME_EMA_SLOW=50          Slow EMA period
    REGIME_ATR_LEN=14           ATR calculation period
    REGIME_TREND_THRESH=0.25    Min trend strength (|EMA diff|/ATR) for trend classification
    REGIME_VOL_LO=0.003         Below this ATR%: vol_compression
    REGIME_VOL_HI=0.025         Above this ATR%: vol_expansion
    REGIME_PANIC_HI=0.040       Above this ATR%: panic
    REGIME_VOLUME_Z_PANIC=2.5   Volume zscore threshold for panic

Imports:
    This module is designed to be imported when sys.path includes runtime/argus:
    from research.regime.regime_engine import classify_regime, RegimeState
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

import pandas as pd

try:
    import pandas_ta as ta
except ImportError:
    ta = None


# ─────────────────────────────────────────────────────────────────────────────
# Regime Labels
# ─────────────────────────────────────────────────────────────────────────────

class RegimeLabel(str, Enum):
    """Regime labels in priority order (highest priority first)."""
    PANIC = "PANIC"
    VOL_EXPANSION = "VOL_EXPANSION"
    VOL_COMPRESSION = "VOL_COMPRESSION"
    TREND_UP = "TREND_UP"
    TREND_DOWN = "TREND_DOWN"
    CHOP = "CHOP"


# Priority ordering (lower index = higher priority)
_REGIME_PRIORITY = [
    RegimeLabel.PANIC,
    RegimeLabel.VOL_EXPANSION,
    RegimeLabel.VOL_COMPRESSION,
    RegimeLabel.TREND_UP,
    RegimeLabel.TREND_DOWN,
    RegimeLabel.CHOP,
]


# ─────────────────────────────────────────────────────────────────────────────
# RegimeState dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RegimeState:
    """
    Immutable regime classification result.

    Attributes:
        asof_ts: ISO8601 UTC timestamp derived from the last included (closed) row.
        label: One of the RegimeLabel values.
        confidence: Float in [0, 1] representing classification confidence.
        features: Dict of key metrics used for classification.
        meta: Optional extra metadata (closed_only, dropped_last_row, nan_fields, etc.)
    """
    asof_ts: str
    label: str
    confidence: float
    features: Dict[str, Any] = field(default_factory=dict)
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return asdict(self)


# ─────────────────────────────────────────────────────────────────────────────
# Environment helpers
# ─────────────────────────────────────────────────────────────────────────────

def _get_env_float(name: str, default: float) -> float:
    v = os.getenv(name, "")
    if v == "":
        return float(default)
    try:
        return float(v)
    except (ValueError, TypeError):
        return float(default)


def _get_env_int(name: str, default: int) -> int:
    v = os.getenv(name, "")
    if v == "":
        return int(default)
    try:
        return int(float(v))
    except (ValueError, TypeError):
        return int(default)


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

def _get_regime_config() -> Dict[str, Any]:
    """Load regime engine configuration from environment with defaults."""
    return {
        "ema_fast": _get_env_int("REGIME_EMA_FAST", 20),
        "ema_slow": _get_env_int("REGIME_EMA_SLOW", 50),
        "atr_len": _get_env_int("REGIME_ATR_LEN", 14),
        "trend_thresh": _get_env_float("REGIME_TREND_THRESH", 0.25),
        "vol_lo": _get_env_float("REGIME_VOL_LO", 0.003),
        "vol_hi": _get_env_float("REGIME_VOL_HI", 0.025),
        "panic_hi": _get_env_float("REGIME_PANIC_HI", 0.040),
        "volume_z_panic": _get_env_float("REGIME_VOLUME_Z_PANIC", 2.5),
    }


# ─────────────────────────────────────────────────────────────────────────────
# DataFrame normalization
# ─────────────────────────────────────────────────────────────────────────────

def _normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize DataFrame to standard OHLCV format with Timestamp column.

    Accepts:
        - Timestamp/Open/High/Low/Close/Volume columns (case-insensitive)
        - datetime index as timestamp source

    Returns DataFrame with columns: Timestamp, Open, High, Low, Close, Volume
    Timestamp is UTC-aware pandas datetime.
    """
    if df is None or df.empty:
        return df

    df = df.copy()

    # Column name mapping (case-insensitive)
    col_map: Dict[str, Optional[str]] = {}
    lower_to_orig = {c.lower(): c for c in df.columns}

    def pick(*names: str) -> Optional[str]:
        for n in names:
            if n in df.columns:
                return n
            if n.lower() in lower_to_orig:
                return lower_to_orig[n.lower()]
        return None

    col_map["Timestamp"] = pick("Timestamp", "timestamp", "time", "datetime", "date")
    col_map["Open"] = pick("Open", "open", "o")
    col_map["High"] = pick("High", "high", "h")
    col_map["Low"] = pick("Low", "low", "l")
    col_map["Close"] = pick("Close", "close", "c")
    col_map["Volume"] = pick("Volume", "volume", "v")

    # If no timestamp column but datetime index, use index
    if col_map["Timestamp"] is None and isinstance(df.index, pd.DatetimeIndex):
        df = df.reset_index()
        if "index" in df.columns:
            df = df.rename(columns={"index": "Timestamp"})
        col_map["Timestamp"] = "Timestamp" if "Timestamp" in df.columns else None

    # Check for required columns
    missing = [k for k, v in col_map.items() if v is None]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Build normalized DataFrame
    out = pd.DataFrame({
        "Timestamp": df[col_map["Timestamp"]],
        "Open": df[col_map["Open"]].astype(float),
        "High": df[col_map["High"]].astype(float),
        "Low": df[col_map["Low"]].astype(float),
        "Close": df[col_map["Close"]].astype(float),
        "Volume": df[col_map["Volume"]].astype(float),
    })

    # Normalize timestamps to UTC-aware datetime
    out["Timestamp"] = pd.to_datetime(out["Timestamp"], utc=True, errors="coerce")

    return out


def _get_asof_ts(df: pd.DataFrame) -> str:
    """
    Get ISO8601 UTC timestamp from the last row of a normalized DataFrame.
    """
    if df is None or df.empty:
        return datetime.now(timezone.utc).isoformat()

    ts = df["Timestamp"].iloc[-1]
    if pd.isna(ts):
        return datetime.now(timezone.utc).isoformat()

    # Ensure it's UTC-aware
    if hasattr(ts, "tzinfo") and ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)

    return ts.isoformat()


# ─────────────────────────────────────────────────────────────────────────────
# NaN safety helpers
# ─────────────────────────────────────────────────────────────────────────────

def _is_nan(x: Any) -> bool:
    """Check if value is NaN."""
    try:
        return pd.isna(x) or (isinstance(x, float) and x != x)
    except (TypeError, ValueError):
        return False


def _safe_last(s: pd.Series, default: float = float("nan")) -> float:
    """Safely get last value from series."""
    if s is None or len(s) == 0:
        return default
    try:
        v = float(s.iloc[-1])
        return v if v == v else default  # NaN check
    except (IndexError, TypeError, ValueError):
        return default


def _check_nan_fields(features: Dict[str, Any]) -> List[str]:
    """Return list of feature keys that are NaN."""
    return [k for k, v in features.items() if _is_nan(v)]


# ─────────────────────────────────────────────────────────────────────────────
# Regime classification
# ─────────────────────────────────────────────────────────────────────────────

def _compute_features(df: pd.DataFrame, cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compute indicators for regime classification.

    Returns dict with:
        ema_fast, ema_slow, atr, atr_pct, trend_strength, volume_z
    """
    if ta is None:
        raise ImportError("pandas_ta is required for regime classification")

    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    volume = df["Volume"]

    # EMAs
    ema_fast = ta.ema(close, length=int(cfg["ema_fast"]))
    ema_slow = ta.ema(close, length=int(cfg["ema_slow"]))

    # ATR
    atr = ta.atr(high, low, close, length=int(cfg["atr_len"]))

    # Derived metrics
    price = _safe_last(close)
    ema_fast_v = _safe_last(ema_fast)
    ema_slow_v = _safe_last(ema_slow)
    atr_v = _safe_last(atr)

    # ATR as percentage of price
    atr_pct = atr_v / price if (price > 0 and atr_v == atr_v) else float("nan")

    # Trend strength: |EMA diff| / ATR
    ema_diff = abs(ema_fast_v - ema_slow_v) if (ema_fast_v == ema_fast_v and ema_slow_v == ema_slow_v) else float("nan")
    trend_strength = ema_diff / atr_v if (atr_v > 0 and atr_v == atr_v) else float("nan")

    # Volume z-score (rolling 50 periods)
    vol_mean = volume.rolling(50, min_periods=10).mean()
    vol_std = volume.rolling(50, min_periods=10).std()
    vol_z = (volume - vol_mean) / vol_std.replace(0, pd.NA)
    volume_z = _safe_last(vol_z)

    # Range as percentage of price (for panic detection)
    range_pct = (high - low) / close
    range_pct_v = _safe_last(range_pct)

    return {
        "price": price,
        "ema_fast": ema_fast_v,
        "ema_slow": ema_slow_v,
        "atr": atr_v,
        "atr_pct": atr_pct,
        "trend_strength": trend_strength,
        "volume_z": volume_z,
        "range_pct": range_pct_v,
    }


def _classify_from_features(features: Dict[str, Any], cfg: Dict[str, Any]) -> tuple[str, float, str]:
    """
    Apply regime classification rules to features.

    Returns (label, confidence, reason).
    Priority: PANIC > VOL_EXPANSION > VOL_COMPRESSION > TREND_UP/TREND_DOWN > CHOP
    """
    atr_pct = features.get("atr_pct", float("nan"))
    trend_strength = features.get("trend_strength", float("nan"))
    volume_z = features.get("volume_z", float("nan"))
    ema_fast = features.get("ema_fast", float("nan"))
    ema_slow = features.get("ema_slow", float("nan"))
    range_pct = features.get("range_pct", float("nan"))

    # NaN check
    if _is_nan(atr_pct) or _is_nan(trend_strength):
        return RegimeLabel.CHOP.value, 0.0, "insufficient_indicators"

    # Thresholds
    vol_lo = cfg["vol_lo"]
    vol_hi = cfg["vol_hi"]
    panic_hi = cfg["panic_hi"]
    trend_thresh = cfg["trend_thresh"]
    volume_z_panic = cfg["volume_z_panic"]

    # PANIC: extreme volatility + high volume
    if atr_pct > panic_hi or (range_pct == range_pct and range_pct > panic_hi):
        vol_z_high = volume_z == volume_z and volume_z > volume_z_panic
        if vol_z_high or atr_pct > panic_hi * 1.5:
            conf = min(1.0, 0.7 + (atr_pct - panic_hi) * 10)
            return RegimeLabel.PANIC.value, conf, "extreme_volatility_and_volume"

    # VOL_EXPANSION: high ATR% but not panic
    if atr_pct > vol_hi:
        conf = min(0.9, 0.6 + (atr_pct - vol_hi) * 5)
        return RegimeLabel.VOL_EXPANSION.value, conf, "high_volatility"

    # VOL_COMPRESSION: very low ATR%
    if atr_pct < vol_lo:
        conf = min(0.9, 0.6 + (vol_lo - atr_pct) * 50)
        return RegimeLabel.VOL_COMPRESSION.value, conf, "low_volatility"

    # TREND_UP / TREND_DOWN: directional EMA alignment + strength
    if ema_fast == ema_fast and ema_slow == ema_slow and trend_strength >= trend_thresh:
        if ema_fast > ema_slow:
            conf = min(0.9, 0.5 + trend_strength * 0.4)
            return RegimeLabel.TREND_UP.value, conf, "bullish_trend"
        else:
            conf = min(0.9, 0.5 + trend_strength * 0.4)
            return RegimeLabel.TREND_DOWN.value, conf, "bearish_trend"

    # CHOP: default when no clear regime
    conf = max(0.3, 0.6 - trend_strength * 0.5) if trend_strength == trend_strength else 0.4
    return RegimeLabel.CHOP.value, conf, "no_clear_trend_or_volatility_signal"


def classify_regime(df: pd.DataFrame, *, closed_only: bool = True) -> RegimeState:
    """
    Classify market regime from OHLCV DataFrame.

    Args:
        df: DataFrame with OHLCV data. Must have Timestamp/Open/High/Low/Close/Volume
            columns (case-insensitive) OR a datetime index.
        closed_only: If True (default), drop the last row before computing indicators.
                     This ensures decisions are based on CLOSED candles only.

    Returns:
        RegimeState with classification result.

    Timeline Safety:
        When closed_only=True, the final row is ALWAYS dropped regardless of
        whether it represents a closed or forming candle. This is the safest
        approach for v1 and ensures no lookahead bias.

    NaN Safety:
        If required indicators produce NaN due to insufficient history,
        returns label="CHOP", confidence=0.0, with meta={"nan_fields":[...]}.
    """
    meta: Dict[str, Any] = {
        "closed_only": closed_only,
        "dropped_last_row": False,
    }

    # Handle empty or very small dataframes
    if df is None or len(df) < 2:
        return RegimeState(
            asof_ts=datetime.now(timezone.utc).isoformat(),
            label=RegimeLabel.CHOP.value,
            confidence=0.0,
            features={"reason": "insufficient_history"},
            meta=meta,
        )

    try:
        # Normalize OHLCV
        df_norm = _normalize_ohlcv(df)
    except ValueError as e:
        return RegimeState(
            asof_ts=datetime.now(timezone.utc).isoformat(),
            label=RegimeLabel.CHOP.value,
            confidence=0.0,
            features={"reason": f"normalization_error: {e}"},
            meta=meta,
        )

    # Apply closed_only: drop last row
    if closed_only and len(df_norm) > 1:
        df_norm = df_norm.iloc[:-1].copy()
        meta["dropped_last_row"] = True

    # Check minimum history after dropping
    cfg = _get_regime_config()
    min_rows = max(cfg["ema_slow"], cfg["atr_len"]) + 10  # Need buffer for indicators

    if len(df_norm) < min_rows:
        return RegimeState(
            asof_ts=_get_asof_ts(df_norm),
            label=RegimeLabel.CHOP.value,
            confidence=0.0,
            features={"reason": f"insufficient_history (need {min_rows}, have {len(df_norm)})"},
            meta=meta,
        )

    # Compute features
    try:
        features = _compute_features(df_norm, cfg)
    except Exception as e:
        return RegimeState(
            asof_ts=_get_asof_ts(df_norm),
            label=RegimeLabel.CHOP.value,
            confidence=0.0,
            features={"reason": f"feature_computation_error: {e}"},
            meta=meta,
        )

    # Check for NaN in required features
    nan_fields = _check_nan_fields(features)
    if nan_fields:
        meta["nan_fields"] = nan_fields
        return RegimeState(
            asof_ts=_get_asof_ts(df_norm),
            label=RegimeLabel.CHOP.value,
            confidence=0.0,
            features=features,
            meta={**meta, "reason": "INSUFFICIENT_HISTORY_OR_NAN"},
        )

    # Classify regime
    label, confidence, reason = _classify_from_features(features, cfg)

    meta["classification_reason"] = reason
    meta["config"] = cfg

    return RegimeState(
        asof_ts=_get_asof_ts(df_norm),
        label=label,
        confidence=round(confidence, 4),
        features=features,
        meta=meta,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Module info for verification
# ─────────────────────────────────────────────────────────────────────────────

__file_info__ = {
    "module": "research.regime.regime_engine",
    "layer": 1,
    "description": "Regime Engine - authoritative, stable, shared regime classification",
}

