"""
ETH Mean Reversion Sleeve (v1)
==============================

Research sleeve purpose
-----------------------
This is a NEW long-only ETH mean-reversion research sleeve designed to be more
orthogonal to the existing BTC volatility-breakout (VB) sleeve:

- BTC VB is convex / trend + breakout oriented.
- This sleeve aims to be concave / mean-reversion oriented by buying oversold
  ETH stretches vs a rolling fair-value anchor.

Safety / regime intent
----------------------
Mean-reversion strategies can fail badly by repeatedly buying into structural
downtrends ("catching falling knives"). For v1, we add a simple ENTRY-ONLY
regime gate to reduce participation in breakdown regimes:

- Only allow new entries when price is above EMA(200).
- Exits are always honored regardless of regime.

Implementation notes
--------------------
- Deterministic, closed-candle decisions (no lookahead).
- Side-effect free outside of using `ctx` to track in-trade state (entry price,
  bars held). No global mutation, no I/O.
- Fits the existing harness interface: generate_intent(df, ctx=None, closed_only=True)
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

# -----------------------------
# Config (defaults for v1)
# -----------------------------

VWAP_WINDOW = 48
ENTRY_BAND = 0.02

RSI_PERIOD = 14
RSI_ENTRY_MAX = 35

STOP_LOSS_PCT = 0.04
MAX_HOLD_BARS = 48

EMA_FILTER_LEN = 200

# Exposure sizing for the sleeve (simple, deterministic)
DESIRED_EXPOSURE_FRAC = 1.0

HORIZON_HOURS = 48

_STATE_KEY = "sg_eth_mean_reversion_v1_state"


def _get_cols(df: pd.DataFrame) -> Tuple[str, str, str, str, str]:
    """
    Accept either canonical (Open/High/Low/Close/Volume) or lowercase variants.
    """
    def pick(*names: str) -> Optional[str]:
        for n in names:
            if n in df.columns:
                return n
        return None

    o = pick("Open", "open")
    h = pick("High", "high")
    l = pick("Low", "low")
    c = pick("Close", "close")
    v = pick("Volume", "volume")
    if any(x is None for x in (o, h, l, c, v)):
        raise ValueError("OHLCV columns missing; need Open/High/Low/Close/Volume (or lowercase variants).")
    return o or "Open", h or "High", l or "Low", c or "Close", v or "Volume"


def _rolling_vwap(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, window: int) -> pd.Series:
    """
    Rolling VWAP using typical price * volume.
    VWAP_t = sum_{t-window+1..t}(typical_price * vol) / sum(vol)
    """
    tp = (high + low + close) / 3.0
    pv = tp * volume
    pv_sum = pv.rolling(window=window, min_periods=window).sum()
    v_sum = volume.rolling(window=window, min_periods=window).sum()
    vwap = pv_sum / v_sum.replace(0.0, np.nan)
    return vwap


def _rsi_wilder(close: pd.Series, period: int) -> pd.Series:
    """
    Wilder RSI (deterministic, no lookahead).
    Uses exponentially smoothed average gains/losses with alpha=1/period.
    """
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi


def _get_state(ctx: Any) -> Dict[str, Any]:
    if ctx is None:
        return {}
    if isinstance(ctx, dict):
        st = ctx.get(_STATE_KEY)
        if isinstance(st, dict):
            return st
    return {}


def _set_state(ctx: Any, st: Dict[str, Any]) -> None:
    if ctx is None:
        return
    if isinstance(ctx, dict):
        ctx[_STATE_KEY] = dict(st)


def generate_intent(df: pd.DataFrame, ctx: Any = None, *, closed_only: bool = True) -> Dict[str, Any]:
    """
    ETH long-only mean reversion:
      Entry: close <= VWAP*(1-ENTRY_BAND) AND RSI <= RSI_ENTRY_MAX AND (close >= EMA(200))
      Exit: close >= VWAP OR close <= entry*(1-STOP_LOSS_PCT) OR bars_held >= MAX_HOLD_BARS
    """
    if df is None or len(df) == 0:
        return {
            "action": "HOLD",
            "confidence": 0.0,
            "desired_exposure_frac": 0.0,
            "horizon_hours": HORIZON_HOURS,
            "reason": "no_data",
            "meta": {"source": "sg_eth_mean_reversion_v1", "closed_only": closed_only},
        }

    o_col, h_col, l_col, c_col, v_col = _get_cols(df)
    high = pd.to_numeric(df[h_col], errors="coerce")
    low = pd.to_numeric(df[l_col], errors="coerce")
    close = pd.to_numeric(df[c_col], errors="coerce")
    volume = pd.to_numeric(df[v_col], errors="coerce").fillna(0.0)

    # Indicators (computed on df slice only)
    vwap = _rolling_vwap(high, low, close, volume, VWAP_WINDOW)
    rsi = _rsi_wilder(close, RSI_PERIOD)
    ema200 = close.ewm(span=EMA_FILTER_LEN, adjust=False, min_periods=EMA_FILTER_LEN).mean()

    i = len(df) - 1
    px = float(close.iloc[i]) if pd.notna(close.iloc[i]) else np.nan
    vwap_last = float(vwap.iloc[i]) if pd.notna(vwap.iloc[i]) else np.nan
    rsi_last = float(rsi.iloc[i]) if pd.notna(rsi.iloc[i]) else np.nan
    ema_last = float(ema200.iloc[i]) if pd.notna(ema200.iloc[i]) else np.nan

    st = _get_state(ctx)
    in_pos = bool(st.get("in_position", False))
    entry_price = st.get("entry_price")
    entry_bar_index = st.get("entry_bar_index")

    # If we don't have enough data to compute the core anchor, do nothing.
    if not np.isfinite(px) or not np.isfinite(vwap_last) or not np.isfinite(rsi_last) or not np.isfinite(ema_last):
        return {
            "action": "HOLD",
            "confidence": 0.0,
            "desired_exposure_frac": 0.0 if not in_pos else DESIRED_EXPOSURE_FRAC,
            "horizon_hours": HORIZON_HOURS,
            "reason": "insufficient_indicator_history",
            "meta": {
                "source": "sg_eth_mean_reversion_v1",
                "in_position": in_pos,
                "closed_only": closed_only,
                "px": px,
                "vwap": vwap_last,
                "rsi": rsi_last,
                "ema_filter_len": EMA_FILTER_LEN,
                "ema": ema_last,
            },
        }

    # Exit logic (always honored)
    if in_pos:
        bars_held = 0
        if isinstance(entry_bar_index, int):
            bars_held = max(0, i - entry_bar_index)

        stop_level = None
        if isinstance(entry_price, (int, float)) and float(entry_price) > 0:
            stop_level = float(entry_price) * (1.0 - STOP_LOSS_PCT)

        exit_reasons = []
        if px >= vwap_last:
            exit_reasons.append("revert_to_vwap")
        if stop_level is not None and px <= stop_level:
            exit_reasons.append("stop_loss")
        if bars_held >= MAX_HOLD_BARS:
            exit_reasons.append("time_stop")

        if exit_reasons:
            _set_state(ctx, {"in_position": False, "entry_price": None, "entry_bar_index": None})
            return {
                "action": "EXIT_LONG",
                "confidence": 0.8,
                "desired_exposure_frac": 0.0,
                "horizon_hours": HORIZON_HOURS,
                "reason": "|".join(exit_reasons),
                "meta": {
                    "source": "sg_eth_mean_reversion_v1",
                    "in_position": True,
                    "entry_price": float(entry_price) if isinstance(entry_price, (int, float)) else None,
                    "bars_held": bars_held,
                    "px": px,
                    "vwap": vwap_last,
                    "rsi": rsi_last,
                    "stop_level": stop_level,
                    "ema": ema_last,
                    "ema_filter_len": EMA_FILTER_LEN,
                },
            }

        # Otherwise hold exposure while in position
        return {
            "action": "HOLD",
            "confidence": 0.4,
            "desired_exposure_frac": DESIRED_EXPOSURE_FRAC,
            "horizon_hours": HORIZON_HOURS,
            "reason": "in_position_hold",
            "meta": {
                "source": "sg_eth_mean_reversion_v1",
                "in_position": True,
                "entry_price": float(entry_price) if isinstance(entry_price, (int, float)) else None,
                "px": px,
                "vwap": vwap_last,
                "rsi": rsi_last,
                "ema": ema_last,
                "ema_filter_len": EMA_FILTER_LEN,
            },
        }

    # Entry gate (ENTRY ONLY): avoid structural breakdown regimes
    regime_ok = px >= ema_last

    stretched = px <= vwap_last * (1.0 - ENTRY_BAND)
    oversold = rsi_last <= RSI_ENTRY_MAX
    entry_ok = regime_ok and stretched and oversold

    if entry_ok:
        _set_state(ctx, {"in_position": True, "entry_price": px, "entry_bar_index": i})
        return {
            "action": "ENTER_LONG",
            "confidence": 0.7,
            "desired_exposure_frac": DESIRED_EXPOSURE_FRAC,
            "horizon_hours": HORIZON_HOURS,
            "reason": "vwap_stretch_and_rsi_oversold_with_ema_gate",
            "meta": {
                "source": "sg_eth_mean_reversion_v1",
                "in_position": False,
                "px": px,
                "vwap": vwap_last,
                "entry_band": ENTRY_BAND,
                "rsi": rsi_last,
                "rsi_entry_max": RSI_ENTRY_MAX,
                "ema": ema_last,
                "ema_filter_len": EMA_FILTER_LEN,
                "regime_ok": True,
                "stretched": True,
                "oversold": True,
            },
        }

    return {
        "action": "HOLD",
        "confidence": 0.0,
        "desired_exposure_frac": 0.0,
        "horizon_hours": HORIZON_HOURS,
        "reason": "no_entry",
        "meta": {
            "source": "sg_eth_mean_reversion_v1",
            "px": px,
            "vwap": vwap_last,
            "stretched": bool(stretched),
            "entry_band": ENTRY_BAND,
            "rsi": rsi_last,
            "oversold": bool(oversold),
            "rsi_entry_max": RSI_ENTRY_MAX,
            "ema": ema_last,
            "ema_filter_len": EMA_FILTER_LEN,
            "regime_ok": bool(regime_ok),
            "closed_only": closed_only,
        },
    }


__file_info__ = {
    "module": "research.strategies.sg_eth_mean_reversion_v1",
    "layer": 2,
    "description": "ETH long-only mean reversion sleeve (VWAP+RSI entry, EMA regime gate, simple exits)",
}

