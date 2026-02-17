from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any

import numpy as np
import pandas as pd


# ============================
# Parameters
# ============================

@dataclass
class RegimeTrendV2Params:
    # Regime
    sma_slow: int = 200
    sma_mid: int = 50
    sma_slow_slope_lookback: int = 24

    # Entry
    sma_fast: int = 20
    breakout_lookback: int = 168
    use_breakout: bool = True
    use_golden_cross: bool = True

    # Risk
    hard_stop_pct: float = 0.10
    trail_stop_pct: float = 0.12
    atr_len: int = 14
    atr_stop_mult: float = 0.0   # 0 disables
    exit_on_close_below_mid: bool = True

    # Sizing / safety
    target_fraction: float = 1.0
    min_bars: int = 300


# ============================
# Helpers
# ============================

def _sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n, min_periods=n).mean()


def _atr(df: pd.DataFrame, n: int) -> pd.Series:
    high = df["High"].astype(float)
    low = df["Low"].astype(float)
    close = df["Close"].astype(float)

    prev_close = close.shift(1)
    tr = pd.concat(
        [
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    return tr.rolling(n, min_periods=n).mean()


# ============================
# Core Signal Builder
# ============================

def build_signals_regime_trend_v2(
    df: pd.DataFrame,
    params: Optional[RegimeTrendV2Params] = None,
) -> pd.DataFrame:

    if params is None:
        params = RegimeTrendV2Params()

    out = pd.DataFrame(index=df.index)
    out["enter_long"] = False
    out["exit_long"] = False

    close = df["Close"].astype(float)

    # Indicators
    sma_slow = _sma(close, params.sma_slow)
    sma_mid = _sma(close, params.sma_mid)
    sma_fast = _sma(close, params.sma_fast)

    atr = (
        _atr(df, params.atr_len)
        if params.atr_stop_mult and params.atr_stop_mult > 0
        else None
    )

    # ============================
    # Regime Filter
    # ============================

    slow_slope = sma_slow - sma_slow.shift(params.sma_slow_slope_lookback)

    regime_on = (
        (close > sma_slow) &
        (sma_mid > sma_slow) &
        (slow_slope > 0)
    )

    # ============================
    # Entry Triggers
    # ============================

    fast_above = sma_fast > sma_mid
    prev_fast_above = fast_above.shift(1).fillna(False)
    golden_cross = fast_above & (~prev_fast_above)

    breakout = pd.Series(False, index=df.index)
    if params.use_breakout:
        hh = close.shift(1).rolling(
            params.breakout_lookback,
            min_periods=params.breakout_lookback,
        ).max()
        breakout = (close > hh).fillna(False)

    # ============================
    # Exit Baseline
    # ============================

    exit_mid_break = (
        (close < sma_mid).fillna(False)
        if params.exit_on_close_below_mid
        else pd.Series(False, index=df.index)
    )

    # ============================
    # Stateful Loop
    # ============================

    in_pos = False
    entry_px = np.nan
    peak_px = np.nan

    for i in range(len(df)):

        if i < params.min_bars:
            continue

        px = float(close.iloc[i])
        reg = bool(regime_on.iloc[i])

        # -------- Entry --------
        entry_signal = False

        regime_ok = False
        if "regime_label" in df.columns:
            regime_ok = (df["regime_label"].iloc[i] == "TREND_UP")

        if reg and regime_ok:
            if params.use_golden_cross and bool(golden_cross.iloc[i]):
                entry_signal = True
            if params.use_breakout and bool(breakout.iloc[i]):
                entry_signal = True

        if not in_pos:
            if entry_signal:
                out.iat[i, 0] = True
                in_pos = True
                entry_px = px
                peak_px = px
            continue

        # -------- In Position --------
        peak_px = max(peak_px, px)

        hard_stop = (
            px <= entry_px * (1 - params.hard_stop_pct)
            if params.hard_stop_pct > 0
            else False
        )

        trail_stop = (
            px <= peak_px * (1 - params.trail_stop_pct)
            if params.trail_stop_pct > 0
            else False
        )

        atr_stop = False
        if atr is not None and not pd.isna(atr.iloc[i]):
            level = peak_px - atr.iloc[i] * params.atr_stop_mult
            atr_stop = px <= level

        trend_exit = bool(exit_mid_break.iloc[i])

        if hard_stop or trail_stop or atr_stop or trend_exit:
            out.iat[i, 1] = True
            in_pos = False
            entry_px = np.nan
            peak_px = np.nan

    return out


# ============================
# Params Export
# ============================

def params_dict(params: RegimeTrendV2Params) -> Dict[str, Any]:
    return {k: getattr(params, k) for k in params.__dataclass_fields__}
