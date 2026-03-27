"""
Volatility Breakout Strategy v3 (Volatility Quality Filter Research Variant)

ETH-focused additive variant of sg_volatility_breakout_v1:
  - keeps v1 breakout/exit logic intact
  - adds one entry-quality gate for new LONG entries only:
      ATR(14) / SMA(ATR(14), 50) > VOL_EXPANSION_MIN

Rationale:
  Reduce false breakouts in noisy / low-quality regimes without the broad
  suppression of a macro trend filter.

Important:
  - Long-only (no shorts)
  - Exits remain v1-driven
  - No Layer 3 / harness changes
  - No lookahead (all calculations use bars available in current df slice)
"""

from __future__ import annotations

import copy
from typing import Any, Dict

import numpy as np
import pandas as pd

from research.strategies import sg_volatility_breakout_v1 as v1

# Volatility-quality filter constants for this research variant.
ATR_LEN = 14
ATR_MA_LEN = 50
VOL_EXPANSION_MIN = 1.10


def _resolve_hlc(df: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.Series]:
    cols = {str(c).strip().lower(): c for c in df.columns}

    def _pick(name_lower: str) -> pd.Series:
        if name_lower in df.columns:
            return pd.to_numeric(df[name_lower], errors="coerce")
        name_upper = name_lower.capitalize()
        if name_upper in df.columns:
            return pd.to_numeric(df[name_upper], errors="coerce")
        src = cols.get(name_lower)
        if src is not None:
            return pd.to_numeric(df[src], errors="coerce")
        return pd.Series(dtype=float)

    high = _pick("high")
    low = _pick("low")
    close = _pick("close")
    return high, low, close


def _atr_rolling_mean(high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
    prev_close = close.shift(1)
    tr = np.maximum(
        high - low,
        np.maximum((high - prev_close).abs(), (low - prev_close).abs()),
    )
    return pd.Series(tr, index=close.index).rolling(period, min_periods=1).mean()


def _vol_filter_ok(df: pd.DataFrame) -> tuple[bool, float | None, float | None, float | None]:
    high, low, close = _resolve_hlc(df)
    if high.empty or low.empty or close.empty:
        return False, None, None, None

    atr = _atr_rolling_mean(high, low, close, ATR_LEN)
    atr_ma = atr.rolling(ATR_MA_LEN, min_periods=1).mean()

    atr_last = float(atr.iloc[-1]) if len(atr) else None
    atr_ma_last = float(atr_ma.iloc[-1]) if len(atr_ma) else None
    if atr_last is None or atr_ma_last is None or not np.isfinite(atr_last) or not np.isfinite(atr_ma_last) or atr_ma_last <= 0:
        return False, atr_last, atr_ma_last, None

    ratio = atr_last / atr_ma_last
    return bool(ratio > VOL_EXPANSION_MIN), atr_last, atr_ma_last, float(ratio)


def _is_entry_action(action: str) -> bool:
    a = str(action or "").strip().upper()
    return a in {"BUY", "ENTER", "ENTER_LONG", "LONG"}


def generate_intent(
    df: pd.DataFrame,
    state: Any = None,
    *,
    closed_only: bool = True,
    **kwargs: Any,
) -> Dict[str, Any]:
    """
    Wrapper around v1:
      1) evaluate v1 on a shadow state
      2) if action is entry and vol-quality filter fails -> force HOLD/flat
      3) otherwise run v1 on real state and return unchanged behavior

    Shadow-state probing prevents mutating real state when an entry is blocked.
    """
    shadow_state = copy.deepcopy(state) if isinstance(state, dict) else state
    probe_intent = v1.generate_intent(df, shadow_state, closed_only=closed_only, **kwargs)

    vol_ok, atr_last, atr_ma_last, ratio = _vol_filter_ok(df)
    if _is_entry_action(probe_intent.get("action")) and not vol_ok:
        base_meta = probe_intent.get("meta") if isinstance(probe_intent.get("meta"), dict) else {}
        meta = dict(base_meta)
        meta.update(
            {
                "strategy": "sg_volatility_breakout_v3_volfilter",
                "base_strategy": "sg_volatility_breakout_v1",
                "vol_filter": f"ATR({ATR_LEN})/SMA(ATR,{ATR_MA_LEN})>{VOL_EXPANSION_MIN}",
                "vol_filter_pass": False,
                "atr_last": atr_last,
                "atr_ma_last": atr_ma_last,
                "atr_to_atr_ma_ratio": ratio,
            }
        )
        return {
            "action": "HOLD",
            "confidence": float(probe_intent.get("confidence", 0.0)),
            "desired_exposure_frac": 0.0,
            "horizon_hours": int(probe_intent.get("horizon_hours", v1.BREAKOUT_LOOKBACK)),
            "reason": (
                "VolBreakout v3: volatility-quality filter blocked entry "
                f"(ATR ratio <= {VOL_EXPANSION_MIN})"
            ),
            "meta": meta,
        }

    # If filter passes (or action is not entry), preserve v1 behavior exactly on real state.
    real_intent = v1.generate_intent(df, state, closed_only=closed_only, **kwargs)
    if isinstance(real_intent.get("meta"), dict):
        real_intent["meta"] = dict(real_intent["meta"])
        real_intent["meta"]["strategy"] = "sg_volatility_breakout_v3_volfilter"
        real_intent["meta"]["base_strategy"] = "sg_volatility_breakout_v1"
        real_intent["meta"]["vol_filter"] = f"ATR({ATR_LEN})/SMA(ATR,{ATR_MA_LEN})>{VOL_EXPANSION_MIN}"
        real_intent["meta"]["vol_filter_pass"] = bool(vol_ok)
        real_intent["meta"]["atr_last"] = atr_last
        real_intent["meta"]["atr_ma_last"] = atr_ma_last
        real_intent["meta"]["atr_to_atr_ma_ratio"] = ratio
    return real_intent

