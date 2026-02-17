from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import numpy as np
import pandas as pd


# ============================
# Parameters
# ============================

@dataclass
class RegimeClassifierV1Params:
    sma_fast: int = 20
    sma_mid: int = 50
    sma_slow: int = 200

    slope_lookback: int = 24

    atr_len: int = 14
    bb_len: int = 20
    bb_std: float = 2.0

    donchian_len: int = 168
    chop_lookback: int = 72

    high_vol_pct: float = 0.75
    low_vol_pct: float = 0.25


# ============================
# Helpers
# ============================

def sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n, min_periods=n).mean()


def atr(df: pd.DataFrame, n: int) -> pd.Series:
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


def rolling_percentile_rank(s: pd.Series, window: int) -> pd.Series:
    return s.rolling(window).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1],
        raw=False,
    )


def chop_score(close: pd.Series, lookback: int) -> pd.Series:
    abs_ret_sum = close.diff().abs().rolling(lookback).sum()
    net_move = (close - close.shift(lookback)).abs()
    return abs_ret_sum / (net_move + 1e-9)


# ============================
# Core Builder
# ============================

def build_regime_classifier_v1(
    df: pd.DataFrame,
    params: Optional[RegimeClassifierV1Params] = None,
) -> pd.DataFrame:

    if params is None:
        params = RegimeClassifierV1Params()

    out = pd.DataFrame(index=df.index)

    close = df["Close"].astype(float)

    # ---- Trend MAs ----
    out["sma_fast"] = sma(close, params.sma_fast)
    out["sma_mid"] = sma(close, params.sma_mid)
    out["sma_slow"] = sma(close, params.sma_slow)

    out["sma_slow_slope"] = (
        out["sma_slow"] - out["sma_slow"].shift(params.slope_lookback)
    )

    out["trend_align_up"] = (
        (out["sma_fast"] > out["sma_mid"]) &
        (out["sma_mid"] > out["sma_slow"])
    )

    out["trend_align_down"] = (
        (out["sma_fast"] < out["sma_mid"]) &
        (out["sma_mid"] < out["sma_slow"])
    )

    out["above_sma_slow"] = close > out["sma_slow"]

    # ---- Volatility ----
    out["atr"] = atr(df, params.atr_len)
    out["atr_pct"] = out["atr"] / close
    out["atr_pct_rank"] = rolling_percentile_rank(out["atr_pct"], 500)

    # ---- Bollinger Bandwidth ----
    bb_mid = sma(close, params.bb_len)
    bb_std = close.rolling(params.bb_len).std()
    bb_upper = bb_mid + params.bb_std * bb_std
    bb_lower = bb_mid - params.bb_std * bb_std
    out["bb_width"] = (bb_upper - bb_lower) / bb_mid
    out["bb_width_rank"] = rolling_percentile_rank(out["bb_width"], 500)

    # ---- Donchian ----
    hh = close.shift(1).rolling(params.donchian_len).max()
    ll = close.shift(1).rolling(params.donchian_len).min()
    out["donch_pos"] = (close - ll) / (hh - ll)

    out["break_high"] = close > hh
    out["break_low"] = close < ll

    # ---- Chop ----
    out["chop"] = chop_score(close, params.chop_lookback)

    # ============================
    # Classification
    # ============================

    labels = []
    confidences = []

    for i in range(len(out)):
        row = out.iloc[i]

        score = {
            "TREND_UP": 0,
            "TREND_DOWN": 0,
            "VOL_EXPANSION": 0,
            "VOL_CONTRACTION": 0,
            "RANGE": 0,
        }

        # --- Trend Up ---
        if row["above_sma_slow"]:
            score["TREND_UP"] += 1
        if row["sma_slow_slope"] > 0:
            score["TREND_UP"] += 1
        if row["trend_align_up"]:
            score["TREND_UP"] += 1
        if row["chop"] < 2.0:
            score["TREND_UP"] += 1

        # --- Trend Down ---
        if not row["above_sma_slow"]:
            score["TREND_DOWN"] += 1
        if row["sma_slow_slope"] < 0:
            score["TREND_DOWN"] += 1
        if row["trend_align_down"]:
            score["TREND_DOWN"] += 1
        if row["chop"] < 2.0:
            score["TREND_DOWN"] += 1

        # --- Vol Expansion ---
        if row["atr_pct_rank"] > params.high_vol_pct:
            score["VOL_EXPANSION"] += 1
        if row["bb_width_rank"] > params.high_vol_pct:
            score["VOL_EXPANSION"] += 1
        if row["break_high"] or row["break_low"]:
            score["VOL_EXPANSION"] += 1

        # --- Vol Contraction ---
        if row["atr_pct_rank"] < params.low_vol_pct:
            score["VOL_CONTRACTION"] += 1
        if row["bb_width_rank"] < params.low_vol_pct:
            score["VOL_CONTRACTION"] += 1

        # --- Range ---
        if row["chop"] > 3.0:
            score["RANGE"] += 2
        if abs(row["donch_pos"] - 0.5) < 0.2:
            score["RANGE"] += 1

        best_label = max(score, key=score.get)
        best_score = score[best_label]
        confidence = best_score / max(1, sum(score.values()))

        labels.append(best_label)
        confidences.append(confidence)

    out["regime_label"] = labels
    out["regime_confidence"] = confidences

    return out
