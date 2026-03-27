"""
Volatility Spike Reversion Strategy (Layer 2) — Sleeve 2 candidate

Long-only mean reversion on volatility dislocations: enter after sharp,
liquidation-style selloffs (large 6h drop + elevated ATR + oversold RSI).
Exit on RSI recovery or 24h max hold.

Designed to complement Core trend (sg_core_exposure_v2); trades rare
volatility spikes, not gradual oversold.

Contract: generate_intent(df, ctx, *, closed_only=True) -> dict
- action: BUY | EXIT | HOLD
- desired_exposure_frac: 0 or 1.0 when long
- Closed-bar deterministic, no lookahead.

Performance: Uses a fixed trailing window for indicators and persists
trade state in ctx to avoid O(n) work per bar (harness passes same ctx).

Author: Itera Dynamics
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd


# ---------------------------
# Constants (research baseline)
# ---------------------------

RSI_LEN = 14
ATR_LEN = 14
ATR_MEAN_WINDOW = 24
RETURN_6H_BARS = 6
ENTRY_RETURN_THRESH = -0.04   # return_6h < -4%
ENTRY_RSI_THRESH = 35
ENTRY_ATR_SPIKE = 1.5
EXIT_RSI = 50
MAX_HOLD_BARS = 24
ENTRY_CONFIDENCE = 0.7
MAX_EXPOSURE = 0.985  # slightly below 1.0 so fee/slip-adjusted cost fits in equity

# Minimum bars to have valid atr_spike at current bar: ATR(14) then 24-bar mean
MIN_BARS = ATR_LEN + ATR_MEAN_WINDOW - 1  # 37
# Trailing window for indicator computation only (avoids full-series work)
TRAIL_WINDOW = 40

# Key in ctx for persisting trade state (avoids re-walking history every bar)
STATE_KEY = "_sg_vol_spike_reversion_v1"


# ---------------------------
# Helpers
# ---------------------------

def _normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure lowercase close, high, low for calculations. Leaves original cols."""
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


def _rsi(close: pd.Series, period: int) -> pd.Series:
    """RSI (Wilder-style). Uses only past and current bar; no lookahead."""
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100.0 - (100.0 / (1.0 + rs))


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
) -> Tuple[float, float, float, float]:
    """
    Compute RSI(14), 6h return, atr_spike for the *last* bar only,
    plus previous-bar RSI for turning-point confirmation.
    Assumes series are already a trailing window of length >= MIN_BARS.
    Returns (rsi_last, rsi_prev, return_6h_last, atr_spike_last).
    """
    rsi = _rsi(close, RSI_LEN)
    atr = _atr(high, low, close, ATR_LEN)
    atr_24h_mean = atr.rolling(ATR_MEAN_WINDOW, min_periods=ATR_MEAN_WINDOW).mean()
    return_6h = close / close.shift(RETURN_6H_BARS) - 1.0
    atr_spike = atr / atr_24h_mean.replace(0, np.nan)

    rsi_last = float(rsi.iloc[-1]) if np.isfinite(rsi.iloc[-1]) else float("nan")
    rsi_prev = float(rsi.iloc[-2]) if len(rsi) >= 2 and np.isfinite(rsi.iloc[-2]) else float("nan")
    ret6_last = float(return_6h.iloc[-1]) if np.isfinite(return_6h.iloc[-1]) else float("nan")
    spike_last = float(atr_spike.iloc[-1]) if np.isfinite(atr_spike.iloc[-1]) else float("nan")
    return rsi_last, rsi_prev, ret6_last, spike_last


def _intent_dict(
    action: str,
    confidence: float,
    desired_exposure_frac: float,
    reason: str,
    meta: Dict[str, Any],
) -> Dict[str, Any]:
    """Build intent dict per harness contract."""
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
    state: Any,
    *,
    closed_only: bool = True,
    **kwargs: Any,
) -> Dict[str, Any]:
    """
    Volatility Spike Reversion intent from OHLCV history.

    Entry: return_6h < -4%, RSI(14) < 35, atr_spike > 1.5 (sharp vol selloff).
    Exit: RSI > 50 or hold >= 24 bars.
    Uses only closed bars; no lookahead. Persists trade state in state dict
    when provided so each bar is O(1) instead of re-walking full history.

    Args:
        df: OHLCV DataFrame (tail = most recent bar). Must have Close, High, Low.
        state: Strategy context (e.g. from harness); if dict, used to persist in_trade/entry_bar_index.
        closed_only: Ignored; behavior is always closed-bar.
        **kwargs: Ignored for compatibility.

    Returns:
        Dict: action, confidence, desired_exposure_frac, horizon_hours, reason, meta.
    """
    if df is None or df.empty:
        return _intent_dict(
            action="HOLD",
            confidence=0.0,
            desired_exposure_frac=0.0,
            reason="VolSpikeReversion: no data",
            meta={"strategy": "sg_vol_spike_reversion_v1"},
        )

    # Use only trailing window to avoid O(n) indicator work per bar
    n_total = len(df)
    if n_total < MIN_BARS:
        return _intent_dict(
            action="HOLD",
            confidence=0.0,
            desired_exposure_frac=0.0,
            reason="VolSpikeReversion: insufficient bars",
            meta={"strategy": "sg_vol_spike_reversion_v1"},
        )

    window = df.tail(TRAIL_WINDOW)
    df0 = _normalize_ohlcv(window)
    if "close" not in df0.columns or "high" not in df0.columns or "low" not in df0.columns:
        return _intent_dict(
            action="HOLD",
            confidence=0.0,
            desired_exposure_frac=0.0,
            reason="VolSpikeReversion: missing close/high/low",
            meta={"strategy": "sg_vol_spike_reversion_v1"},
        )

    close = df0["close"].astype(float)
    high = df0["high"].astype(float)
    low = df0["low"].astype(float)

    rsi_val, rsi_prev, ret6_val, spike_val = _compute_current_indicators(close, high, low)
    current_bar_index = n_total - 1

    # Persist trade state in ctx when it's a mutable dict (harness passes same ctx every bar)
    ctx = state if isinstance(state, dict) else None
    if ctx is not None:
        st = ctx.setdefault(STATE_KEY, {"in_trade": False, "entry_bar_index": None})
    else:
        st = {"in_trade": False, "entry_bar_index": None}

    in_trade = st.get("in_trade", False)
    entry_bar_index = st.get("entry_bar_index")

    # Exit check when in trade
    if in_trade and entry_bar_index is not None:
        bars_held = current_bar_index - entry_bar_index
        if rsi_val > EXIT_RSI or bars_held >= MAX_HOLD_BARS:
            if ctx is not None and STATE_KEY in ctx:
                ctx[STATE_KEY] = {"in_trade": False, "entry_bar_index": None}
            return _intent_dict(
                action="EXIT",
                confidence=ENTRY_CONFIDENCE,
                desired_exposure_frac=0.0,
                reason=f"VolSpikeReversion: exit (RSI>{EXIT_RSI} or hold>={MAX_HOLD_BARS}h)",
                meta={
                    "strategy": "sg_vol_spike_reversion_v1",
                    "rsi": rsi_val,
                    "return_6h": ret6_val,
                    "atr_spike": spike_val,
                    "in_trade": False,
                    "entry_bar_idx": None,
                },
            )

    # Entry check when not in trade (and indicators are finite)
    if not in_trade and np.isfinite(rsi_val) and np.isfinite(ret6_val) and np.isfinite(spike_val):
        # Base crash + volatility spike condition (unchanged)
        base_entry = (
            ret6_val < ENTRY_RETURN_THRESH
            and rsi_val < ENTRY_RSI_THRESH
            and spike_val > ENTRY_ATR_SPIKE
        )

        # Confirmation gating:
        # 1) Positive bar: close[t] > close[t-1]
        # 2) RSI turning up out of oversold: rsi[t-1] < oversold_thresh and rsi[t] > rsi[t-1]
        if len(close) >= 2 and np.isfinite(rsi_prev):
            positive_bar = float(close.iloc[-1]) > float(close.iloc[-2])
            rsi_turning_up = (rsi_prev < ENTRY_RSI_THRESH) and (rsi_val > rsi_prev)
            confirmations_ok = positive_bar and rsi_turning_up
        else:
            # Not enough history for confirmation checks -> allow base behavior
            confirmations_ok = True

        if base_entry and confirmations_ok:
            if ctx is not None:
                ctx[STATE_KEY] = {"in_trade": True, "entry_bar_index": current_bar_index}
            return _intent_dict(
                action="BUY",
                confidence=ENTRY_CONFIDENCE,
                desired_exposure_frac=MAX_EXPOSURE,
                reason=(
                    f"VolSpikeReversion: long (return_6h={ret6_val:.2%}, RSI={rsi_val:.1f}, "
                    f"atr_spike={spike_val:.2f}); exit RSI>{EXIT_RSI} or {MAX_HOLD_BARS}h"
                ),
                meta={
                    "strategy": "sg_vol_spike_reversion_v1",
                    "rsi": rsi_val,
                    "return_6h": ret6_val,
                    "atr_spike": spike_val,
                    "in_trade": True,
                    "entry_bar_idx": current_bar_index,
                },
            )

    # Maintain long if already in trade
    if in_trade:
        return _intent_dict(
            action="BUY",
            confidence=ENTRY_CONFIDENCE,
            desired_exposure_frac=MAX_EXPOSURE,
            reason=(
                f"VolSpikeReversion: long (return_6h={ret6_val:.2%}, RSI={rsi_val:.1f}, "
                f"atr_spike={spike_val:.2f}); exit RSI>{EXIT_RSI} or {MAX_HOLD_BARS}h"
            ),
            meta={
                "strategy": "sg_vol_spike_reversion_v1",
                "rsi": rsi_val,
                "return_6h": ret6_val,
                "atr_spike": spike_val,
                "in_trade": True,
                "entry_bar_idx": entry_bar_index,
            },
        )

    # Flat
    return _intent_dict(
        action="HOLD",
        confidence=0.5,
        desired_exposure_frac=0.0,
        reason=(
            f"VolSpikeReversion: flat (return_6h={ret6_val:.2%}, RSI={rsi_val:.1f}, "
            f"atr_spike={spike_val:.2f}; no entry signal)"
        ),
        meta={
            "strategy": "sg_vol_spike_reversion_v1",
            "rsi": rsi_val,
            "return_6h": ret6_val,
            "atr_spike": spike_val,
            "in_trade": False,
            "entry_bar_idx": None,
        },
    )
