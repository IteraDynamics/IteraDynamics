"""
Volatility Breakout Strategy v1 (Layer 2) — promoted from fast_strategy_screener

Long-only: enter after ATR expansion + breakout above recent high; exit when close
below recent low. Matches the top surviving fast-screener candidate:
  atr_period=14, breakout_lookback=48, atr_expansion_min=1.1

Contract: generate_intent(df, state=None, closed_only=True, **kwargs) -> dict
- action: BUY | EXIT | HOLD
- desired_exposure_frac: 0 or max_exposure when long
- Closed-bar deterministic, no lookahead.

State: in_trade, entry_bar_idx, entry_ts, entry_price (persisted in state dict when provided).
Costs applied by harness; strategy does not apply fees/slippage.

Author: Itera Dynamics
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd


# ---------------------------
# Constants (top screener candidate: volatility_breakout, 48/1.1)
# ---------------------------

ATR_PERIOD = 14
BREAKOUT_LOOKBACK = 48
ATR_EXPANSION_MIN = 1.1
ENTRY_CONFIDENCE = 0.7
HOLD_CONFIDENCE = 0.5
MAX_EXPOSURE = 0.985

# Need enough bars for ATR(14), ATR_avg over 28, and 48-bar rolling high/low
MIN_BARS = max(ATR_PERIOD * 2 + BREAKOUT_LOOKBACK, 48 + 28)  # 96
TRAIL_WINDOW = 100
STATE_KEY = "_sg_volatility_breakout_v1"


# ---------------------------
# Helpers (match screener: rolling mean ATR, not Wilder)
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


def _atr_rolling_mean(high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
    """ATR as rolling mean of True Range (matches fast_strategy_screener). No lookahead."""
    prev_close = close.shift(1)
    tr = np.maximum(
        high - low,
        np.maximum(
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ),
    )
    return pd.Series(tr, index=close.index).rolling(period, min_periods=1).mean()


def _compute_indicators_last_bar(
    close: pd.Series,
    high: pd.Series,
    low: pd.Series,
) -> Tuple[float, float, float, float, float]:
    """
    For last bar only: atr, atr_avg_prior, roll_high_prior, roll_low_current, close_last.
    Matches screener: atr_avg = atr.rolling(period*2).mean().shift(1); roll_high = high.rolling(lookback).max().shift(1).
    """
    atr = _atr_rolling_mean(high, low, close, ATR_PERIOD)
    atr_avg = atr.rolling(ATR_PERIOD * 2, min_periods=1).mean().shift(1)
    roll_high = high.rolling(BREAKOUT_LOOKBACK, min_periods=1).max().shift(1)
    roll_low = low.rolling(BREAKOUT_LOOKBACK, min_periods=1).min()

    atr_last = float(atr.iloc[-1]) if np.isfinite(atr.iloc[-1]) else float("nan")
    atr_avg_last = float(atr_avg.iloc[-1]) if len(atr_avg) and np.isfinite(atr_avg.iloc[-1]) else float("nan")
    roll_high_prior = float(roll_high.iloc[-1]) if len(roll_high) and np.isfinite(roll_high.iloc[-1]) else float("nan")
    roll_low_last = float(roll_low.iloc[-1]) if len(roll_low) and np.isfinite(roll_low.iloc[-1]) else float("nan")
    close_last = float(close.iloc[-1])
    return atr_last, atr_avg_last, roll_high_prior, roll_low_last, close_last


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
        "horizon_hours": BREAKOUT_LOOKBACK,
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
    Volatility breakout intent: long after ATR expansion + close >= prior 48-bar high;
    exit when close < prior 48-bar low. Matches fast_strategy_screener volatility_breakout
    (atr_period=14, breakout_lookback=48, atr_expansion_min=1.1).
    """
    if df is None or df.empty:
        return _intent_dict(
            action="HOLD",
            confidence=0.0,
            desired_exposure_frac=0.0,
            reason="VolBreakout: no data",
            meta={"strategy": "sg_volatility_breakout_v1"},
        )

    n_total = len(df)
    if n_total < MIN_BARS:
        return _intent_dict(
            action="HOLD",
            confidence=0.0,
            desired_exposure_frac=0.0,
            reason="VolBreakout: insufficient bars",
            meta={"strategy": "sg_volatility_breakout_v1"},
        )

    window = df.tail(TRAIL_WINDOW)
    df0 = _normalize_ohlcv(window)
    if "close" not in df0.columns or "high" not in df0.columns or "low" not in df0.columns:
        return _intent_dict(
            action="HOLD",
            confidence=0.0,
            desired_exposure_frac=0.0,
            reason="VolBreakout: missing OHLC",
            meta={"strategy": "sg_volatility_breakout_v1"},
        )

    close = df0["close"].astype(float)
    high = df0["high"].astype(float)
    low = df0["low"].astype(float)

    atr_val, atr_avg_val, roll_high_prior, roll_low_last, close_last = _compute_indicators_last_bar(
        close, high, low
    )
    current_bar_idx = n_total - 1

    # Exit: close < roll_low at previous bar. For "at bar close" we use roll_low through current bar;
    # screener exit_sig = close < roll_low.shift(1) means at current bar we compare close to roll_low of previous bar.
    # roll_low.shift(1) at current index = rolling low of bars [current-48, current-1]. So we need roll_low
    # computed excluding current bar. Here roll_low_last is low.rolling(48).min() at last index of window,
    # which is the min over the last 48 bars of the window. That includes current bar's low. So for exit we need
    # "close_last < rolling_low_prior" where rolling_low_prior = min(low) over bars [current-48, current-1].
    # So we need one more value: roll_low at -2 (prior bar's 48-bar low). Let me add that.
    roll_low_prior = float(
        low.rolling(BREAKOUT_LOOKBACK, min_periods=1).min().shift(1).iloc[-1]
    ) if len(low) >= 2 else float("nan")
    exit_triggered = np.isfinite(roll_low_prior) and close_last < roll_low_prior

    ctx = state if isinstance(state, dict) else None
    if ctx is not None:
        st = ctx.setdefault(STATE_KEY, {
            "in_trade": False,
            "entry_bar_idx": None,
            "entry_ts": None,
            "entry_price": None,
        })
    else:
        st = {"in_trade": False, "entry_bar_idx": None, "entry_ts": None, "entry_price": None}

    in_trade = st.get("in_trade", False)
    entry_bar_idx = st.get("entry_bar_idx")
    entry_ts = st.get("entry_ts")
    entry_price = st.get("entry_price")

    # Exit when in trade and close < prior 48-bar low
    if in_trade and exit_triggered:
        if ctx is not None and STATE_KEY in ctx:
            ctx[STATE_KEY] = {
                "in_trade": False,
                "entry_bar_idx": None,
                "entry_ts": None,
                "entry_price": None,
            }
        return _intent_dict(
            action="EXIT",
            confidence=HOLD_CONFIDENCE,
            desired_exposure_frac=0.0,
            reason=f"VolBreakout: exit (close < {BREAKOUT_LOOKBACK}-bar low)",
            meta={
                "strategy": "sg_volatility_breakout_v1",
                "atr": atr_val,
                "atr_avg": atr_avg_val,
                "roll_high_prior": roll_high_prior,
                "roll_low_prior": roll_low_prior,
                "in_trade": False,
                "entry_bar_idx": None,
                "entry_ts": None,
                "entry_price": None,
            },
        )

    # Entry when not in trade: ATR expanded and close >= prior 48-bar high
    atr_expanded = (
        np.isfinite(atr_val)
        and np.isfinite(atr_avg_val)
        and atr_avg_val > 0
        and atr_val > (atr_avg_val * ATR_EXPANSION_MIN)
    )
    entry_triggered = (
        atr_expanded
        and np.isfinite(roll_high_prior)
        and close_last >= roll_high_prior
    )

    if not in_trade and entry_triggered:
        ts = df.index[-1] if hasattr(df.index, "__getitem__") else None
        if hasattr(ts, "to_pydatetime"):
            ts = ts.to_pydatetime()
        if ctx is not None:
            ctx[STATE_KEY] = {
                "in_trade": True,
                "entry_bar_idx": current_bar_idx,
                "entry_ts": ts,
                "entry_price": close_last,
            }
        return _intent_dict(
            action="BUY",
            confidence=ENTRY_CONFIDENCE,
            desired_exposure_frac=MAX_EXPOSURE,
            reason=(
                f"VolBreakout: long (close>={roll_high_prior:.0f}, ATR>{ATR_EXPANSION_MIN}x avg); "
                f"exit when close < {BREAKOUT_LOOKBACK}-bar low"
            ),
            meta={
                "strategy": "sg_volatility_breakout_v1",
                "atr": atr_val,
                "atr_avg": atr_avg_val,
                "roll_high_prior": roll_high_prior,
                "in_trade": True,
                "entry_bar_idx": current_bar_idx,
                "entry_ts": ts,
                "entry_price": close_last,
            },
        )

    if in_trade:
        return _intent_dict(
            action="BUY",
            confidence=HOLD_CONFIDENCE,
            desired_exposure_frac=MAX_EXPOSURE,
            reason=f"VolBreakout: hold long; exit when close < {BREAKOUT_LOOKBACK}-bar low",
            meta={
                "strategy": "sg_volatility_breakout_v1",
                "atr": atr_val,
                "atr_avg": atr_avg_val,
                "roll_high_prior": roll_high_prior,
                "in_trade": True,
                "entry_bar_idx": entry_bar_idx,
                "entry_ts": entry_ts,
                "entry_price": entry_price,
            },
        )

    return _intent_dict(
        action="HOLD",
        confidence=HOLD_CONFIDENCE,
        desired_exposure_frac=0.0,
        reason=f"VolBreakout: flat (no breakout or ATR expansion)",
        meta={
            "strategy": "sg_volatility_breakout_v1",
            "atr": atr_val,
            "atr_avg": atr_avg_val,
            "roll_high_prior": roll_high_prior,
            "in_trade": False,
        },
    )
