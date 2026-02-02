from __future__ import annotations

from dataclasses import dataclass
import pandas as pd
import numpy as np


@dataclass
class RTR1Params:
    sma_fast: int = 50
    sma_regime: int = 200
    atr_len: int = 14

    use_atr_filter: bool = True
    atr_filter_lookback: int = 200  # median window for ATR filter

    max_hold_bars: int | None = None  # e.g. 48 for 48 hours if hourly bars; None disables
    target_frac_equity: float = 1.0   # 1.0 = deploy all cash on entry


def _sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n).mean()


def _atr(df: pd.DataFrame, n: int) -> pd.Series:
    high = df["High"]
    low = df["Low"]
    close = df["Close"]
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.rolling(n).mean()


def build_signals_rtr1(df: pd.DataFrame, params: RTR1Params) -> pd.DataFrame:
    df = df.copy()

    close = df["Close"].astype(float)

    sma_fast = _sma(close, params.sma_fast)
    sma_regime = _sma(close, params.sma_regime)

    bull = close > sma_regime

    # Cross logic: "cross above" and "cross below"
    fast_above = close > sma_fast
    cross_up = fast_above & (~fast_above.shift(1).fillna(False))
    cross_dn = (~fast_above) & (fast_above.shift(1).fillna(False))

    # ATR filter (avoid dead chop)
    atr = _atr(df, params.atr_len)
    if params.use_atr_filter:
        atr_med = atr.rolling(params.atr_filter_lookback).median()
        vol_ok = atr > atr_med
    else:
        vol_ok = pd.Series(True, index=df.index)

    enter_long = cross_up & bull & vol_ok

    # Exit: cross below fast MA OR (optional) time exit
    exit_long = cross_dn

    # Time-exit needs position tracking. We implement it as a secondary pass.
    if params.max_hold_bars is not None and params.max_hold_bars > 0:
        in_pos = False
        hold = 0
        for i in range(len(df)):
            if (not in_pos) and bool(enter_long.iloc[i]):
                in_pos = True
                hold = 0
            elif in_pos:
                hold += 1
                if hold >= params.max_hold_bars:
                    exit_long.iloc[i] = True
                    in_pos = False
                    hold = 0

            if in_pos and bool(exit_long.iloc[i]):
                in_pos = False
                hold = 0

    # target_usd will be set by the engine using cash at runtime, but we can provide a “fraction” hint
    # We'll leave target_usd blank here (engine defaults to all cash), but keep a column for future sizing.
    sig = pd.DataFrame(
        {
            "enter_long": enter_long.fillna(False).astype(bool),
            "exit_long": exit_long.fillna(False).astype(bool),
            "target_frac_equity": float(params.target_frac_equity),
        }
    )
    return sig
