"""
ETH Volume-Confirmed Breakout v1 (Layer 2) — research sleeve candidate

Long-only breakout sleeve promoted from fast_strategy_screener top-ranked non-VB candidate:
  template: volume_confirmed_breakout
  params:   lookback=48, volume_ma_bars=48, vol_mult=1.5

Intent:
  Reduce false breakouts by requiring volume confirmation on entry.

Contract:
  generate_intent(df, state=None, closed_only=True, **kwargs) -> dict
  - action: BUY | EXIT | HOLD
  - desired_exposure_frac: 0 or MAX_EXPOSURE when long
  - Closed-bar deterministic, no lookahead (all indicator comparisons use prior-bar bands).

Notes:
  - This is a NEW ETH-focused research sleeve (does not modify BTC VB).
  - Costs are applied by the harness; strategy does not apply fees/slippage.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd

# ---------------------------
# Constants (screener-aligned defaults)
# ---------------------------

BREAKOUT_LOOKBACK = 48
VOLUME_MA_BARS = 48
VOL_MULT = 1.5

ENTRY_CONFIDENCE = 0.7
HOLD_CONFIDENCE = 0.5
MAX_EXPOSURE = 0.985

STATE_KEY = "_sg_eth_volume_confirmed_breakout_v1"
MIN_BARS = max(BREAKOUT_LOOKBACK + 2, VOLUME_MA_BARS + 2, 64)
TRAIL_WINDOW = 400  # keep computations bounded but comfortably > lookbacks


def _normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure lowercase columns exist: close, high, low, volume.
    Leaves original columns intact.
    """
    if df is None or df.empty:
        return df
    out = df.copy(deep=False)
    cols = {str(c).strip().lower(): c for c in df.columns}

    def _ensure(key: str) -> None:
        if key in out.columns:
            return
        src = cols.get(key)
        if src is not None:
            out[key] = pd.to_numeric(df[src], errors="coerce")
        elif key.upper() in df.columns:
            out[key] = pd.to_numeric(df[key.upper()], errors="coerce")

    for k in ("close", "high", "low", "volume"):
        _ensure(k)
    if "volume" not in out.columns:
        out["volume"] = 0.0

    return out


def _compute_indicators_last_bar(
    close: pd.Series,
    high: pd.Series,
    low: pd.Series,
    volume: pd.Series,
) -> Tuple[float, float, float, float]:
    """
    For last bar only:
      - roll_high_prior: prior N-bar high (shift(1))
      - roll_low_prior:  prior N-bar low  (shift(1))
      - vol_ma_prior:    prior volume MA  (shift(1))
      - close_last
    """
    roll_high_prior = high.rolling(BREAKOUT_LOOKBACK, min_periods=1).max().shift(1)
    roll_low_prior = low.rolling(BREAKOUT_LOOKBACK, min_periods=1).min().shift(1)
    vol_ma_prior = volume.rolling(VOLUME_MA_BARS, min_periods=1).mean().shift(1)

    return (
        float(roll_high_prior.iloc[-1]) if len(roll_high_prior) else float("nan"),
        float(roll_low_prior.iloc[-1]) if len(roll_low_prior) else float("nan"),
        float(vol_ma_prior.iloc[-1]) if len(vol_ma_prior) else float("nan"),
        float(close.iloc[-1]),
    )


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
        "horizon_hours": int(BREAKOUT_LOOKBACK),
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
        return _intent_dict(
            action="HOLD",
            confidence=0.0,
            desired_exposure_frac=0.0,
            reason="VolConfirmedBreakout: no data",
            meta={"strategy": "sg_eth_volume_confirmed_breakout_v1"},
        )

    n_total = len(df)
    if n_total < MIN_BARS:
        return _intent_dict(
            action="HOLD",
            confidence=0.0,
            desired_exposure_frac=0.0,
            reason="VolConfirmedBreakout: insufficient bars",
            meta={"strategy": "sg_eth_volume_confirmed_breakout_v1"},
        )

    window = df.tail(TRAIL_WINDOW)
    df0 = _normalize_ohlcv(window)
    if any(k not in df0.columns for k in ("close", "high", "low", "volume")):
        return _intent_dict(
            action="HOLD",
            confidence=0.0,
            desired_exposure_frac=0.0,
            reason="VolConfirmedBreakout: missing OHLCV",
            meta={"strategy": "sg_eth_volume_confirmed_breakout_v1"},
        )

    close = df0["close"].astype(float)
    high = df0["high"].astype(float)
    low = df0["low"].astype(float)
    volume = pd.to_numeric(df0["volume"], errors="coerce").fillna(0.0).astype(float)

    roll_high_prior, roll_low_prior, vol_ma_prior, close_last = _compute_indicators_last_bar(
        close, high, low, volume
    )
    vol_last = float(volume.iloc[-1]) if len(volume) else float("nan")

    ctx = state if isinstance(state, dict) else None
    if ctx is not None:
        st = ctx.setdefault(
            STATE_KEY,
            {"in_trade": False, "entry_bar_idx": None, "entry_price": None},
        )
    else:
        st = {"in_trade": False, "entry_bar_idx": None, "entry_price": None}

    in_trade = bool(st.get("in_trade", False))

    exit_triggered = np.isfinite(roll_low_prior) and close_last < roll_low_prior
    if in_trade and exit_triggered:
        if ctx is not None:
            ctx[STATE_KEY] = {"in_trade": False, "entry_bar_idx": None, "entry_price": None}
        return _intent_dict(
            action="EXIT",
            confidence=HOLD_CONFIDENCE,
            desired_exposure_frac=0.0,
            reason=f"VolConfirmedBreakout: exit (close < {BREAKOUT_LOOKBACK}-bar low)",
            meta={
                "strategy": "sg_eth_volume_confirmed_breakout_v1",
                "roll_high_prior": roll_high_prior,
                "roll_low_prior": roll_low_prior,
                "vol_last": vol_last,
                "vol_ma_prior": vol_ma_prior,
                "vol_mult": VOL_MULT,
                "in_trade": False,
            },
        )

    entry_breakout = np.isfinite(roll_high_prior) and close_last >= roll_high_prior
    entry_vol_ok = np.isfinite(vol_ma_prior) and vol_ma_prior > 0 and (vol_last >= (vol_ma_prior * VOL_MULT))
    entry_triggered = entry_breakout and entry_vol_ok

    if (not in_trade) and entry_triggered:
        if ctx is not None:
            ctx[STATE_KEY] = {
                "in_trade": True,
                "entry_bar_idx": n_total - 1,
                "entry_price": close_last,
            }
        return _intent_dict(
            action="BUY",
            confidence=ENTRY_CONFIDENCE,
            desired_exposure_frac=MAX_EXPOSURE,
            reason=(
                f"VolConfirmedBreakout: long (close>=prior high, vol>={VOL_MULT}x MA); "
                f"exit when close < {BREAKOUT_LOOKBACK}-bar low"
            ),
            meta={
                "strategy": "sg_eth_volume_confirmed_breakout_v1",
                "roll_high_prior": roll_high_prior,
                "roll_low_prior": roll_low_prior,
                "vol_last": vol_last,
                "vol_ma_prior": vol_ma_prior,
                "vol_mult": VOL_MULT,
                "in_trade": True,
            },
        )

    if in_trade:
        return _intent_dict(
            action="BUY",
            confidence=HOLD_CONFIDENCE,
            desired_exposure_frac=MAX_EXPOSURE,
            reason=f"VolConfirmedBreakout: hold long; exit when close < {BREAKOUT_LOOKBACK}-bar low",
            meta={
                "strategy": "sg_eth_volume_confirmed_breakout_v1",
                "roll_high_prior": roll_high_prior,
                "roll_low_prior": roll_low_prior,
                "vol_last": vol_last,
                "vol_ma_prior": vol_ma_prior,
                "vol_mult": VOL_MULT,
                "in_trade": True,
            },
        )

    return _intent_dict(
        action="HOLD",
        confidence=HOLD_CONFIDENCE,
        desired_exposure_frac=0.0,
        reason="VolConfirmedBreakout: flat (no breakout or no volume confirmation)",
        meta={
            "strategy": "sg_eth_volume_confirmed_breakout_v1",
            "roll_high_prior": roll_high_prior,
            "roll_low_prior": roll_low_prior,
            "vol_last": vol_last,
            "vol_ma_prior": vol_ma_prior,
            "vol_mult": VOL_MULT,
            "in_trade": False,
        },
    )

