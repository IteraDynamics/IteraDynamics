"""
ETH Trend Pullback Continuation v1 (Layer 2) — research sleeve candidate.

Screener-aligned fixed config:
  fast_ema_bars=72
  slow_ema_bars=200
  pullback_bars=5
  reentry_breakout_bars=20

Long-only intent:
  - Entry in bullish regime after a short pullback and re-breakout.
  - Exit on trend breakdown (close below fast or slow EMA).
"""

from __future__ import annotations

from typing import Any, Dict

import pandas as pd

FAST_EMA_BARS = 72
SLOW_EMA_BARS = 200
PULLBACK_BARS = 5
REENTRY_BREAKOUT_BARS = 20

ENTRY_CONFIDENCE = 0.65
HOLD_CONFIDENCE = 0.45
MAX_EXPOSURE = 0.985
STATE_KEY = "_sg_eth_trend_pullback_continuation_v1"
MIN_BARS = max(SLOW_EMA_BARS + 2, REENTRY_BREAKOUT_BARS + PULLBACK_BARS + 2, 256)
TRAIL_WINDOW = 900


def _normalize_ohlc(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy(deep=False)
    cols = {str(c).strip().lower(): c for c in out.columns}
    for key in ("close", "high", "low"):
        if key in out.columns:
            continue
        src = cols.get(key)
        if src is not None:
            out[key] = pd.to_numeric(out[src], errors="coerce")
    return out


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
        "horizon_hours": int(REENTRY_BREAKOUT_BARS),
        "reason": reason,
        "meta": meta,
    }


def generate_intent(
    df: pd.DataFrame,
    state: Any = None,
    *,
    closed_only: bool = True,
    **kwargs: Any,
) -> Dict[str, Any]:
    if df is None or df.empty:
        return _intent_dict("HOLD", 0.0, 0.0, "TrendPullbackContinuation: no data", {"strategy": STATE_KEY})

    if len(df) < MIN_BARS:
        return _intent_dict(
            "HOLD",
            0.0,
            0.0,
            "TrendPullbackContinuation: insufficient bars",
            {"strategy": STATE_KEY},
        )

    w = _normalize_ohlc(df.tail(TRAIL_WINDOW))
    if any(k not in w.columns for k in ("close", "high", "low")):
        return _intent_dict("HOLD", 0.0, 0.0, "TrendPullbackContinuation: missing OHLC", {"strategy": STATE_KEY})

    close = w["close"].astype(float)
    fast = close.ewm(span=FAST_EMA_BARS, adjust=False).mean()
    slow = close.ewm(span=SLOW_EMA_BARS, adjust=False).mean()

    trend_ok = bool((fast.iloc[-2] > slow.iloc[-2]) and (close.iloc[-1] >= slow.iloc[-2]))
    pullback_recent = bool((close.shift(1) < fast.shift(1)).rolling(PULLBACK_BARS, min_periods=1).max().iloc[-1] > 0)
    breakout_level = float(close.rolling(REENTRY_BREAKOUT_BARS, min_periods=1).max().shift(1).iloc[-1])
    reclaims_fast = bool(close.iloc[-1] >= fast.iloc[-2])
    breakout_ok = bool(close.iloc[-1] >= breakout_level)
    entry_triggered = trend_ok and pullback_recent and reclaims_fast and breakout_ok

    exit_triggered = bool((close.iloc[-1] < fast.iloc[-2]) or (close.iloc[-1] < slow.iloc[-2]))

    ctx = state if isinstance(state, dict) else None
    if ctx is not None:
        st = ctx.setdefault(STATE_KEY, {"in_trade": False})
    else:
        st = {"in_trade": False}
    in_trade = bool(st.get("in_trade", False))

    if in_trade and exit_triggered:
        if ctx is not None:
            ctx[STATE_KEY] = {"in_trade": False}
        return _intent_dict(
            "EXIT",
            HOLD_CONFIDENCE,
            0.0,
            "TrendPullbackContinuation: exit (close below EMA trend support)",
            {"strategy": STATE_KEY, "in_trade": False},
        )

    if (not in_trade) and entry_triggered:
        if ctx is not None:
            ctx[STATE_KEY] = {"in_trade": True}
        return _intent_dict(
            "BUY",
            ENTRY_CONFIDENCE,
            MAX_EXPOSURE,
            "TrendPullbackContinuation: long (trend + pullback + re-breakout)",
            {"strategy": STATE_KEY, "in_trade": True},
        )

    if in_trade:
        return _intent_dict(
            "BUY",
            HOLD_CONFIDENCE,
            MAX_EXPOSURE,
            "TrendPullbackContinuation: hold long",
            {"strategy": STATE_KEY, "in_trade": True},
        )

    return _intent_dict(
        "HOLD",
        HOLD_CONFIDENCE,
        0.0,
        "TrendPullbackContinuation: flat (no qualified setup)",
        {"strategy": STATE_KEY, "in_trade": False},
    )

