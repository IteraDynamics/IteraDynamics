"""
Volatility Breakout Continuation Strategy (Layer 2) — Sleeve 2 candidate

Long-only momentum continuation after volatility expansion and breakout.
Targets short-horizon, lower-correlation behavior vs Core trend (sg_core_exposure_v2).

Contract: generate_intent(df, state=None, closed_only=True, **kwargs) -> dict
- action: BUY | EXIT | HOLD
- desired_exposure_frac: 0 or max_exposure when long
- Closed-bar deterministic, no lookahead.

Uses trailing window + state in ctx for O(1) per bar in the harness.

Author: Itera Dynamics
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd


# ---------------------------
# Constants
# ---------------------------

ATR_LEN = 14
ATR_MEAN_WINDOW = 24
RETURN_6H_BARS = 6
ROLLING_HL_BARS = 24
ENTRY_ATR_SPIKE = 1.5
ENTRY_RETURN_6H_MIN = 0.03
MAX_HOLD_BARS = 24
ENTRY_CONFIDENCE = 0.7
HOLD_CONFIDENCE = 0.5
MAX_EXPOSURE = 0.985

MIN_BARS = ATR_LEN + ATR_MEAN_WINDOW - 1  # 37
TRAIL_WINDOW = 40
STATE_KEY = "_sg_vol_breakout_continuation_v1"


# ---------------------------
# Helpers
# ---------------------------

def _normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure lowercase close, high, low. Leaves original cols."""
    if df is None or df.empty:
        return df
    out = df.copy(deep=False)
    cols = {str(c).strip().lower(): c for c in df.columns}
    for key in ("close", "high", "low"):
        if key not in out.columns:
            src = cols.get(key) or (cols.get("c") if key == "close" else None)
            if src is not None:
                out[key] = df[src].astype(float)
        if key.upper() in df.columns and key not in out.columns:
            out[key] = df[key.upper()].astype(float)
    return out


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
    """ATR(period) Wilder-style (RMA of True Range). No lookahead."""
    prev_close = close.shift(1)
    tr = np.maximum(
        high - low,
        np.maximum(
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ),
    )
    return tr.ewm(alpha=1.0 / period, adjust=False).mean()


def _compute_current_indicators(
    close: pd.Series,
    high: pd.Series,
    low: pd.Series,
) -> Tuple[float, float, float, float, float, float, float]:
    """
    Compute ATR, atr_spike, return_6h, prior 24-bar rolling high/low for last bar.
    Returns (atr, atr_spike, return_6h, breakout_high_prior, rolling_low_prior, close_last, prev_bar_low).
    """
    atr = _atr(high, low, close, ATR_LEN)
    atr_24_mean = atr.rolling(ATR_MEAN_WINDOW, min_periods=ATR_MEAN_WINDOW).mean()
    atr_spike = atr / atr_24_mean.replace(0, np.nan)
    return_6h = close / close.shift(RETURN_6H_BARS) - 1.0
    rolling_high = high.rolling(ROLLING_HL_BARS, min_periods=ROLLING_HL_BARS).max()
    rolling_low = low.rolling(ROLLING_HL_BARS, min_periods=ROLLING_HL_BARS).min()

    atr_last = float(atr.iloc[-1]) if np.isfinite(atr.iloc[-1]) else float("nan")
    spike_last = float(atr_spike.iloc[-1]) if np.isfinite(atr_spike.iloc[-1]) else float("nan")
    ret6_last = float(return_6h.iloc[-1]) if np.isfinite(return_6h.iloc[-1]) else float("nan")
    # Prior bar's 24-bar rolling high (no lookahead)
    breakout_high_prior = (
        float(rolling_high.iloc[-2]) if len(rolling_high) >= 2 and np.isfinite(rolling_high.iloc[-2]) else float("nan")
    )
    rolling_low_prior = (
        float(rolling_low.iloc[-2]) if len(rolling_low) >= 2 and np.isfinite(rolling_low.iloc[-2]) else float("nan")
    )
    close_last = float(close.iloc[-1])
    prev_bar_low = float(low.iloc[-2]) if len(low) >= 2 else float("nan")
    return atr_last, spike_last, ret6_last, breakout_high_prior, rolling_low_prior, close_last, prev_bar_low


def _intent_dict(
    action: str,
    confidence: float,
    desired_exposure_frac: float,
    reason: str,
    meta: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "action": action,
        "confidence": float(confidence),
        "desired_exposure_frac": float(desired_exposure_frac),
        "horizon_hours": MAX_HOLD_BARS,
        "reason": reason,
        "meta": meta,
    }


# ---------------------------
# Public API
# ---------------------------

def generate_intent(
    df: pd.DataFrame,
    state: Any = None,
    *,
    closed_only: bool = True,
    **kwargs: Any,
) -> Dict[str, Any]:
    """
    Volatility breakout continuation intent: long after expansion + breakout.

    Entry: atr_spike > 1.5, 6h return > 3%, close > prior 24-bar rolling high, not in trade.
    Exit: hold >= 24 bars, or close < previous bar low, or 6h return < 0.
    """
    if df is None or df.empty:
        return _intent_dict(
            action="HOLD",
            confidence=0.0,
            desired_exposure_frac=0.0,
            reason="VolBreakout: no data",
            meta={"strategy": "sg_vol_breakout_continuation_v1"},
        )

    n_total = len(df)
    if n_total < MIN_BARS:
        return _intent_dict(
            action="HOLD",
            confidence=0.0,
            desired_exposure_frac=0.0,
            reason="VolBreakout: insufficient bars",
            meta={"strategy": "sg_vol_breakout_continuation_v1"},
        )

    window = df.tail(TRAIL_WINDOW)
    df0 = _normalize_ohlcv(window)
    if "close" not in df0.columns or "high" not in df0.columns or "low" not in df0.columns:
        return _intent_dict(
            action="HOLD",
            confidence=0.0,
            desired_exposure_frac=0.0,
            reason="VolBreakout: missing OHLC",
            meta={"strategy": "sg_vol_breakout_continuation_v1"},
        )

    close = df0["close"].astype(float)
    high = df0["high"].astype(float)
    low = df0["low"].astype(float)

    (
        atr_val,
        spike_val,
        ret6_val,
        breakout_high_prior,
        _rolling_low_prior,
        close_last,
        prev_bar_low,
    ) = _compute_current_indicators(close, high, low)
    current_bar_index = n_total - 1

    ctx = state if isinstance(state, dict) else None
    if ctx is not None:
        st = ctx.setdefault(STATE_KEY, {"in_trade": False, "entry_bar_index": None, "entry_ts": None, "entry_price": None})
    else:
        st = {"in_trade": False, "entry_bar_index": None, "entry_ts": None, "entry_price": None}

    in_trade = st.get("in_trade", False)
    entry_bar_index = st.get("entry_bar_index")
    entry_ts = st.get("entry_ts")
    entry_price = st.get("entry_price")

    # Exit checks when in trade
    if in_trade and entry_bar_index is not None:
        bars_held = current_bar_index - entry_bar_index
        exit_hold = bars_held >= MAX_HOLD_BARS
        exit_below_prev_low = np.isfinite(prev_bar_low) and close_last < prev_bar_low
        exit_ret6_neg = np.isfinite(ret6_val) and ret6_val < 0
        if exit_hold or exit_below_prev_low or exit_ret6_neg:
            if ctx is not None and STATE_KEY in ctx:
                ctx[STATE_KEY] = {"in_trade": False, "entry_bar_index": None, "entry_ts": None, "entry_price": None}
            return _intent_dict(
                action="EXIT",
                confidence=HOLD_CONFIDENCE,
                desired_exposure_frac=0.0,
                reason=f"VolBreakout: exit (hold>={MAX_HOLD_BARS}h or close<prev_low or 6h ret<0)",
                meta={
                    "strategy": "sg_vol_breakout_continuation_v1",
                    "atr": atr_val,
                    "atr_spike": spike_val,
                    "return_6h": ret6_val,
                    "breakout_high": breakout_high_prior,
                    "in_trade": False,
                },
            )

    # Entry check when not in trade
    if not in_trade:
        base_ok = (
            np.isfinite(spike_val)
            and np.isfinite(ret6_val)
            and np.isfinite(breakout_high_prior)
            and spike_val > ENTRY_ATR_SPIKE
            and ret6_val > ENTRY_RETURN_6H_MIN
            and close_last > breakout_high_prior
        )
        if base_ok:
            ts = df.iloc[-1].get("Timestamp") if "Timestamp" in df.columns else None
            if ctx is not None:
                ctx[STATE_KEY] = {
                    "in_trade": True,
                    "entry_bar_index": current_bar_index,
                    "entry_ts": ts,
                    "entry_price": close_last,
                }
            return _intent_dict(
                action="BUY",
                confidence=ENTRY_CONFIDENCE,
                desired_exposure_frac=MAX_EXPOSURE,
                reason=(
                    f"VolBreakout: long (atr_spike={spike_val:.2f}, 6h={ret6_val:.2%}, "
                    f"close>{breakout_high_prior:.0f}); exit hold>={MAX_HOLD_BARS}h or close<prev_low or 6h<0"
                ),
                meta={
                    "strategy": "sg_vol_breakout_continuation_v1",
                    "atr": atr_val,
                    "atr_spike": spike_val,
                    "return_6h": ret6_val,
                    "breakout_high": breakout_high_prior,
                    "in_trade": True,
                },
            )

    # Maintain long
    if in_trade:
        return _intent_dict(
            action="BUY",
            confidence=HOLD_CONFIDENCE,
            desired_exposure_frac=MAX_EXPOSURE,
            reason=(
                f"VolBreakout: hold long (atr_spike={spike_val:.2f}, 6h={ret6_val:.2%}); "
                f"exit hold>={MAX_HOLD_BARS}h or close<prev_low or 6h<0"
            ),
            meta={
                "strategy": "sg_vol_breakout_continuation_v1",
                "atr": atr_val,
                "atr_spike": spike_val,
                "return_6h": ret6_val,
                "breakout_high": breakout_high_prior,
                "in_trade": True,
            },
        )

    return _intent_dict(
        action="HOLD",
        confidence=HOLD_CONFIDENCE,
        desired_exposure_frac=0.0,
        reason=f"VolBreakout: flat (atr_spike={spike_val:.2f}, 6h={ret6_val:.2%}; no breakout)",
        meta={
            "strategy": "sg_vol_breakout_continuation_v1",
            "atr": atr_val,
            "atr_spike": spike_val,
            "return_6h": ret6_val,
            "breakout_high": breakout_high_prior,
            "in_trade": False,
        },
    )
