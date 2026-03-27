# research/experiments/fast_strategy_screener.py
"""
Fast Deterministic Strategy Screener (Stage 1 of hybrid quant research pipeline)

Purpose:
  Quickly reject obviously bad strategy ideas before promoting them into the full
  Argus research harness and portfolio geometry. Uses only pandas/numpy.
  Does NOT replace the harness; it is a cheap first-pass filter.

Architecture:
  - Data: single hourly OHLCV CSV → DataFrame with DatetimeIndex.
  - Backtest engine: compute_position_series(entry, exit) → position (0/1);
    compute_equity_curve(df, position, fee_bps, slippage_bps) → equity;
    get_trade_list(df, position) → list of {entry_idx, exit_idx, entry_price, exit_price, hold_bars};
    summary_metrics(df, equity, trades) → dict of total_return_pct, cagr, max_drawdown, win_rate,
    profit_factor, n_trades, avg_hold_hours. run_backtest() ties them together.
  - Templates: each is a function (df, **params) -> (entry_signal, exit_signal) (boolean Series).
  - Runner: for each (strategy_name, param_set) in TEMPLATES × PARAM_GRIDS, run_backtest → one row;
    add survives_filter (cagr>0, profit_factor>1, max_drawdown<60%%, n_trades>=30); write full CSV and
    filtered CSV; print top N from filtered, sorted by cagr desc, profit_factor desc, max_drawdown asc.

Example command:
  python research/experiments/fast_strategy_screener.py
  python research/experiments/fast_strategy_screener.py --asset eth --top 10

Example command (explicit data path):
  python research/experiments/fast_strategy_screener.py --data data/ethusd_3600s_2019-01-01_to_2025-12-30.csv --fee_bps 25 --slippage_bps 10 --top 10

Output CSV locations:
  research/experiments/output/fast_strategy_screener_results_<asset>.csv (all candidates + survives_filter)
  research/experiments/output/fast_strategy_screener_filtered_results_<asset>.csv (survivors only)

Adding new templates later:
  1. Implement a function (df: pd.DataFrame, **kwargs) -> Tuple[pd.Series, pd.Series]
     returning (entry_signal, exit_signal) as boolean Series aligned to df.index.
  2. Register it in TEMPLATES: TEMPLATES["my_template"] = my_template_signals.
  3. Add a small parameter grid in PARAM_GRIDS["my_template"] = [{"param": value}, ...].
  4. Re-run the screener; the new template will be evaluated with the rest.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

import numpy as np
import pandas as pd

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = REPO_ROOT / "research" / "experiments" / "output"

# Default data inference expects your current repo naming convention:
#   data/{asset}usd_3600s_2019-01-01_to_2025-12-30.csv  (e.g. btcusd..., ethusd...)
DEFAULT_DATA_TEMPLATE = REPO_ROOT / "data" / "{asset}usd_3600s_2019-01-01_to_2025-12-30.csv"
DEFAULT_ASSET = "btc"

# Pre-filter thresholds for "top candidates" (only survivors are printed and in filtered CSV)
# You can override these via CLI flags for asset-specific tuning.
FILTER_CAGR_MIN = 0.0
FILTER_PROFIT_FACTOR_MIN = 1.0
FILTER_MAX_DRAWDOWN_MAX = 0.60   # reject if drawdown >= 60%
FILTER_N_TRADES_MIN = 30


# -----------------------------------------------------------------------------
# Backtest engine: equity curve, trade list, summary metrics
# -----------------------------------------------------------------------------

def compute_position_series(
    entry_signal: pd.Series,
    exit_signal: pd.Series,
    max_hold_bars: int | None = None,
) -> pd.Series:
    """
    Long-only position (0 or 1) from entry/exit signals.
    Entry at close of bar t => position 1 from bar t+1.
    Exit at close of bar t => position 0 from bar t+1.
    """
    n = len(entry_signal)
    position = np.zeros(n, dtype=np.float64)
    pos = 0.0
    entry_idx: int | None = None

    for i in range(n):
        if pos == 0.0 and entry_signal.iloc[i]:
            pos = 1.0
            entry_idx = i
        elif pos == 1.0:
            exit_due_to_signal = bool(exit_signal.iloc[i])
            exit_due_to_hold = (
                max_hold_bars is not None
                and entry_idx is not None
                and (i - entry_idx) >= max_hold_bars
            )
            if exit_due_to_signal or exit_due_to_hold:
                pos = 0.0
                entry_idx = None

        position[i] = pos

    return pd.Series(position, index=entry_signal.index)


def compute_equity_curve(
    df: pd.DataFrame,
    position: pd.Series,
    fee_bps: float,
    slippage_bps: float,
) -> pd.Series:
    """
    Equity curve starting at 1.0. Assumes position is known at start of each bar.
    Bar return = position_prev * (close[t]/close[t-1] - 1). Costs applied on entry/exit.
    """
    close = df["Close"].values
    n = len(close)
    equity = np.ones(n)
    pos_prev = 0.0
    cost_bps = (fee_bps + slippage_bps) / 1e4

    for i in range(1, n):
        pos_curr = position.iloc[i]
        bar_ret = (close[i] / close[i - 1]) - 1.0
        cost = 0.0
        if pos_prev == 0 and pos_curr == 1:
            cost = cost_bps
        elif pos_prev == 1 and pos_curr == 0:
            cost = cost_bps
        equity[i] = equity[i - 1] * (1.0 + pos_prev * bar_ret - cost)
        pos_prev = pos_curr

    return pd.Series(equity, index=df.index)


def get_trade_list(
    df: pd.DataFrame,
    position: pd.Series,
) -> List[Dict[str, Any]]:
    """List of trades: entry_bar, exit_bar, entry_price, exit_price, hold_bars."""
    pos = position.values
    n = len(pos)
    trades: List[Dict[str, Any]] = []
    in_trade = False
    entry_idx = 0

    for i in range(n):
        if not in_trade and pos[i] == 1:
            in_trade = True
            entry_idx = i
        elif in_trade and pos[i] == 0:
            in_trade = False
            exit_idx = i - 1  # exit at close of previous bar
            if exit_idx >= entry_idx:
                trades.append({
                    "entry_idx": entry_idx,
                    "exit_idx": exit_idx,
                    "entry_price": df["Close"].iloc[entry_idx],
                    "exit_price": df["Close"].iloc[exit_idx],
                    "hold_bars": exit_idx - entry_idx,
                })
    if in_trade:
        trades.append({
            "entry_idx": entry_idx,
            "exit_idx": n - 1,
            "entry_price": df["Close"].iloc[entry_idx],
            "exit_price": df["Close"].iloc[n - 1],
            "hold_bars": (n - 1) - entry_idx,
        })
    return trades


def summary_metrics(
    df: pd.DataFrame,
    equity: pd.Series,
    trades: List[Dict[str, Any]],
) -> Dict[str, float]:
    """Compute total_return_pct, cagr, max_drawdown, win_rate, profit_factor, n_trades, avg_hold_hours."""
    n = len(equity)
    if n == 0:
        return _empty_metrics()

    total_return = (equity.iloc[-1] / equity.iloc[0]) - 1.0
    total_return_pct = total_return * 100.0

    index = df.index
    if isinstance(index, pd.DatetimeIndex):
        years = (index[-1] - index[0]).total_seconds() / (365.25 * 24 * 3600)
    else:
        years = (n - 1) / (24 * 365.25)  # assume hourly
    years = max(years, 1e-6)
    cagr = (equity.iloc[-1] / equity.iloc[0]) ** (1.0 / years) - 1.0

    cummax = np.maximum.accumulate(equity.values)
    drawdowns = (cummax - equity.values) / np.where(cummax > 0, cummax, 1.0)
    max_drawdown = float(np.max(drawdowns))

    n_trades = len(trades)
    if n_trades == 0:
        win_rate = 0.0
        profit_factor = 0.0
        avg_hold_hours = 0.0
    else:
        pnls = [t["exit_price"] / t["entry_price"] - 1.0 for t in trades]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]
        win_rate = len(wins) / n_trades
        gross_profit = sum(wins)
        gross_loss = abs(sum(losses))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else (gross_profit if gross_profit > 0 else 0.0)
        avg_hold_hours = float(np.mean([t["hold_bars"] for t in trades]))

    calmar = 0.0
    if max_drawdown > 1e-12:
        calmar = cagr / max_drawdown
    elif cagr > 0:
        calmar = float("inf")

    return {
        "total_return_pct": total_return_pct,
        "cagr": cagr,
        "max_drawdown": max_drawdown,
        "calmar": calmar,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "n_trades": n_trades,
        "avg_hold_hours": avg_hold_hours,
    }


def _empty_metrics() -> Dict[str, float]:
    return {
        "total_return_pct": 0.0,
        "cagr": 0.0,
        "max_drawdown": 0.0,
        "calmar": 0.0,
        "win_rate": 0.0,
        "profit_factor": 0.0,
        "n_trades": 0,
        "avg_hold_hours": 0.0,
    }


def _compute_ranked_score(results: pd.DataFrame) -> pd.DataFrame:
    """
    Add percentile-rank based composite score.

    Composite (higher is better):
      score =
        cagr_rank * w_cagr +
        calmar_rank * w_calmar +
        profit_factor_rank * w_profit_factor +
        drawdown_good_rank * w_drawdown_good

    where drawdown_good_rank = 1 - drawdown_rank (smaller DD -> better).
    """
    ranked = results.copy()
    if ranked.empty:
        ranked["score"] = []
        return ranked

    cagr_rank = ranked["cagr"].rank(pct=True, method="average", ascending=True)
    calmar_rank = ranked["calmar"].rank(pct=True, method="average", ascending=True)
    profit_factor_rank = ranked["profit_factor"].rank(pct=True, method="average", ascending=True)
    drawdown_rank = ranked["max_drawdown"].rank(pct=True, method="average", ascending=True)
    drawdown_good_rank = 1.0 - drawdown_rank

    ranked["cagr_rank"] = cagr_rank
    ranked["calmar_rank"] = calmar_rank
    ranked["profit_factor_rank"] = profit_factor_rank
    ranked["drawdown_rank"] = drawdown_rank
    ranked["drawdown_good_rank"] = drawdown_good_rank
    w_cagr = float(globals().get("SCORE_W_CAGR", 0.4))
    w_calmar = float(globals().get("SCORE_W_CALMAR", 0.3))
    w_profit_factor = float(globals().get("SCORE_W_PROFIT_FACTOR", 0.2))
    w_drawdown_good = float(globals().get("SCORE_W_DRAWDOWN_GOOD", 0.1))

    w_sum = w_cagr + w_calmar + w_profit_factor + w_drawdown_good
    if w_sum <= 0:
        w_cagr, w_calmar, w_profit_factor, w_drawdown_good = 0.4, 0.3, 0.2, 0.1
        w_sum = 1.0

    # Normalize so score remains in a predictable 0..~1-ish range.
    w_cagr /= w_sum
    w_calmar /= w_sum
    w_profit_factor /= w_sum
    w_drawdown_good /= w_sum

    ranked["score"] = (
        (cagr_rank * w_cagr)
        + (calmar_rank * w_calmar)
        + (profit_factor_rank * w_profit_factor)
        + (drawdown_good_rank * w_drawdown_good)
    )
    return ranked


def run_backtest(
    df: pd.DataFrame,
    entry_signal: pd.Series,
    exit_signal: pd.Series,
    fee_bps: float = 25.0,
    slippage_bps: float = 10.0,
    max_hold_bars: int | None = None,
) -> Tuple[pd.Series, List[Dict[str, Any]], Dict[str, float]]:
    """
    Run long-only backtest from entry/exit signals.
    Returns (equity_curve, trade_list, summary_metrics).
    """
    position = compute_position_series(entry_signal, exit_signal, max_hold_bars=max_hold_bars)
    equity = compute_equity_curve(df, position, fee_bps, slippage_bps)
    trades = get_trade_list(df, position)
    metrics = summary_metrics(df, equity, trades)
    return equity, trades, metrics


# -----------------------------------------------------------------------------
# Strategy template interface
# Each template is a function: (df, **params) -> (entry_signal, exit_signal)
# -----------------------------------------------------------------------------

def breakout_continuation_signals(
    df: pd.DataFrame,
    lookback: int = 24,
    recent_return_bars: int = 6,
    min_recent_return_pct: float = 0.5,
) -> Tuple[pd.Series, pd.Series]:
    """
    Enter on breakout above rolling high with positive recent return.
    Exit on close below rolling low (or simple trailing).
    """
    high = df["High"]
    low = df["Low"]
    close = df["Close"]
    roll_high = high.rolling(lookback, min_periods=1).max()
    roll_low = low.rolling(lookback, min_periods=1).min()
    recent_ret = close.pct_change(recent_return_bars)
    entry = (
        (close >= roll_high.shift(1)) &
        (roll_high.shift(1).notna()) &
        (recent_ret >= min_recent_return_pct / 100.0)
    )
    # Exit: close below recent low or below entry-period low
    exit_roll_low = low.rolling(max(lookback // 2, 4), min_periods=1).min()
    exit_sig = close < exit_roll_low.shift(1)
    return entry.fillna(False), exit_sig.fillna(False)


def trend_pullback_signals(
    df: pd.DataFrame,
    trend_ma_bars: int = 48,
    pullback_lookback: int = 12,
    pullback_pct: float = 1.0,
) -> Tuple[pd.Series, pd.Series]:
    """
    Enter when price above trend filter but short-term pullback occurred.
    Exit when price crosses below trend MA.
    """
    close = df["Close"]
    trend_ma = close.rolling(trend_ma_bars, min_periods=1).mean()
    above_trend = close > trend_ma.shift(1)
    roll_high = close.rolling(pullback_lookback, min_periods=1).max()
    pullback = (roll_high.shift(1) - close) / roll_high.shift(1).replace(0, np.nan) >= (pullback_pct / 100.0)
    entry = above_trend & pullback
    exit_sig = close < trend_ma.shift(1)
    return entry.fillna(False), exit_sig.fillna(False)


def rsi_mean_reversion_signals(
    df: pd.DataFrame,
    rsi_period: int = 14,
    oversold: float = 30.0,
    exit_rsi: float = 50.0,
) -> Tuple[pd.Series, pd.Series]:
    """
    Enter when RSI oversold, exit on RSI recovery.
    """
    close = df["Close"]
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.rolling(rsi_period, min_periods=1).mean()
    avg_loss = loss.rolling(rsi_period, min_periods=1).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    rsi = rsi.fillna(50.0)
    entry = rsi < oversold
    exit_sig = rsi >= exit_rsi
    return entry.fillna(False), exit_sig.fillna(False)


def volatility_spike_reversion_signals(
    df: pd.DataFrame,
    atr_period: int = 14,
    drop_pct: float = 3.0,
    atr_mult: float = 1.5,
    exit_bars: int = 24,
) -> Tuple[pd.Series, pd.Series]:
    """
    Enter after large drop + volatility spike (ATR expansion).
    Exit after fixed bars or recovery (simplified: exit after exit_bars or when price > entry-level).
    """
    close = df["Close"]
    high, low = df["High"], df["Low"]
    tr = np.maximum(high - low, np.maximum(abs(high - close.shift(1)), abs(low - close.shift(1))))
    atr = pd.Series(tr, index=df.index).rolling(atr_period, min_periods=1).mean()
    ret = close.pct_change()
    # Large drop: close below close of N bars ago by at least drop_pct
    lookback = max(atr_period, 6)
    prev_close = close.shift(lookback)
    drop = (prev_close - close) / prev_close.replace(0, np.nan) >= (drop_pct / 100.0)
    atr_above_avg = atr > atr.rolling(atr_period * 2, min_periods=1).mean().shift(1)
    entry = drop & atr_above_avg
    # Exit: time-based would require state; use simple exit when close > rolling mean
    roll_mean = close.rolling(exit_bars, min_periods=1).mean()
    exit_sig = close >= roll_mean.shift(1)
    return entry.fillna(False), exit_sig.fillna(False)


def volatility_breakout_signals(
    df: pd.DataFrame,
    atr_period: int = 14,
    breakout_lookback: int = 24,
    atr_expansion_min: float = 1.2,
) -> Tuple[pd.Series, pd.Series]:
    """
    Enter after ATR expansion + breakout above recent high.
    Exit when close below recent low.
    """
    close = df["Close"]
    high, low = df["High"], df["Low"]
    tr = np.maximum(high - low, np.maximum(abs(high - close.shift(1)), abs(low - close.shift(1))))
    atr = pd.Series(tr, index=df.index).rolling(atr_period, min_periods=1).mean()
    atr_avg = atr.rolling(atr_period * 2, min_periods=1).mean().shift(1)
    roll_high = high.rolling(breakout_lookback, min_periods=1).max().shift(1)
    roll_low = low.rolling(breakout_lookback, min_periods=1).min()
    atr_expanded = atr > (atr_avg * atr_expansion_min)
    entry = (close >= roll_high) & atr_expanded
    exit_sig = close < roll_low.shift(1)
    return entry.fillna(False), exit_sig.fillna(False)


def ema_crossover_signals(
    df: pd.DataFrame,
    fast_ema_bars: int = 12,
    slow_ema_bars: int = 48,
) -> Tuple[pd.Series, pd.Series]:
    """
    Long-only EMA crossover:
      - enter when fast EMA crosses ABOVE slow EMA
      - exit when fast EMA crosses BELOW slow EMA
    """
    close = df["Close"]
    fast = close.ewm(span=fast_ema_bars, adjust=False).mean()
    slow = close.ewm(span=slow_ema_bars, adjust=False).mean()

    # Cross conditions at bar close (t):
    #   fast[t] > slow[t] and fast[t-1] <= slow[t-1]
    entry = (fast > slow) & (fast.shift(1) <= slow.shift(1))
    exit_sig = (fast < slow) & (fast.shift(1) >= slow.shift(1))
    return entry.fillna(False), exit_sig.fillna(False)


def bollinger_mean_reversion_signals(
    df: pd.DataFrame,
    window: int = 20,
    num_std: float = 2.0,
) -> Tuple[pd.Series, pd.Series]:
    """
    Bollinger mean reversion:
      - enter when Close < lower band (using prior-band to reduce lookahead)
      - exit when Close >= SMA(mid-band)
    """
    close = df["Close"]
    sma = close.rolling(window, min_periods=1).mean()
    std = close.rolling(window, min_periods=1).std(ddof=0)
    lower = sma - (num_std * std)

    entry = (close <= lower.shift(1)) & sma.shift(1).notna()
    exit_sig = close >= sma.shift(1)
    return entry.fillna(False), exit_sig.fillna(False)


def volume_confirmed_breakout_signals(
    df: pd.DataFrame,
    lookback: int = 24,
    volume_ma_bars: int = 24,
    vol_mult: float = 1.2,
) -> Tuple[pd.Series, pd.Series]:
    """
    Volume-confirmed breakout (long-only):
      - enter on breakout above rolling high with volume confirmation
      - exit when Close falls below rolling low

    If `Volume` is missing, returns all-False signals.
    """
    if "Volume" not in df.columns:
        false_series = pd.Series(False, index=df.index)
        return false_series, false_series

    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    vol = df["Volume"]

    roll_high = high.rolling(lookback, min_periods=1).max().shift(1)
    roll_low = low.rolling(lookback, min_periods=1).min()
    vol_ma = vol.rolling(volume_ma_bars, min_periods=1).mean()

    vol_ok = vol >= (vol_ma.shift(1) * vol_mult)
    entry = (close >= roll_high) & vol_ok
    exit_sig = close < roll_low.shift(1)
    return entry.fillna(False), exit_sig.fillna(False)


def macd_crossover_signals(
    df: pd.DataFrame,
    macd_fast: int = 12,
    macd_slow: int = 26,
    macd_signal: int = 9,
) -> Tuple[pd.Series, pd.Series]:
    """
    MACD crossover (long-only):
      - entry when MACD crosses above signal line
      - exit when MACD crosses below signal line

    No lookahead: uses shifted cross detection at bar close.
    """
    close = df["Close"]
    ema_fast = close.ewm(span=macd_fast, adjust=False).mean()
    ema_slow = close.ewm(span=macd_slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal = macd.ewm(span=macd_signal, adjust=False).mean()

    entry = (macd > signal) & (macd.shift(1) <= signal.shift(1))
    exit_sig = (macd < signal) & (macd.shift(1) >= signal.shift(1))
    return entry.fillna(False), exit_sig.fillna(False)


def donchian_breakout_signals(
    df: pd.DataFrame,
    lookback: int = 48,
) -> Tuple[pd.Series, pd.Series]:
    """
    Donchian channel breakout (long-only):
      - entry when close >= prior-N-bar high
      - exit when close < prior-N-bar low

    No lookahead: prior high/low uses shift(1).
    """
    close = df["Close"]
    high = df["High"]
    low = df["Low"]

    roll_high_prior = high.rolling(lookback, min_periods=1).max().shift(1)
    roll_low_prior = low.rolling(lookback, min_periods=1).min().shift(1)

    entry = close >= roll_high_prior
    exit_sig = close < roll_low_prior
    return entry.fillna(False), exit_sig.fillna(False)


def atr_trend_trailing_stop_signals(
    df: pd.DataFrame,
    trend_ema_bars: int = 100,
    atr_period: int = 14,
    atr_mult: float = 2.0,
) -> Tuple[pd.Series, pd.Series]:
    """
    ATR-trailing-stop approximation (long-only):
      - entry: close > EMA(trend_ema_bars)
      - exit: close < (rolling_max_close_prev - atr_mult * ATR_prev)

    This is a deterministic, state-free approximation of a trailing stop.
    It uses prior-bar rolling max / ATR to avoid lookahead.
    """
    close = df["Close"]
    high = df["High"]
    low = df["Low"]

    ema_trend = close.ewm(span=trend_ema_bars, adjust=False).mean()
    entry = close > ema_trend.shift(1)

    prev_close = close.shift(1)
    tr = np.maximum(
        high - low,
        np.maximum((high - prev_close).abs(), (low - prev_close).abs()),
    )
    atr = pd.Series(tr, index=df.index).rolling(atr_period, min_periods=1).mean()
    atr_prev = atr.shift(1)

    roll_max_close_prev = close.rolling(trend_ema_bars, min_periods=1).max().shift(1)
    stop_level = roll_max_close_prev - (atr_mult * atr_prev)
    exit_sig = close < stop_level

    return entry.fillna(False), exit_sig.fillna(False)


def vwap_mean_reversion_signals(
    df: pd.DataFrame,
    window: int = 48,
) -> Tuple[pd.Series, pd.Series]:
    """
    VWAP mean reversion (long-only):
      - entry when close <= rolling VWAP
      - exit when close >= rolling VWAP

    Uses prior VWAP (shift(1)) for cross-like behavior to avoid lookahead.
    If Volume column is missing, returns all-False signals.
    """
    if "Volume" not in df.columns:
        false_series = pd.Series(False, index=df.index)
        return false_series, false_series

    close = df["Close"]
    vol = df["Volume"]

    pv = close * vol
    vwap = pv.rolling(window, min_periods=1).sum() / vol.rolling(window, min_periods=1).sum()

    vwap_prev = vwap.shift(1)
    entry = close <= vwap_prev
    exit_sig = close >= vwap_prev
    return entry.fillna(False), exit_sig.fillna(False)


def zscore_mean_reversion_signals(
    df: pd.DataFrame,
    window: int = 48,
    z_entry: float = -2.0,
    z_exit: float = -0.5,
) -> Tuple[pd.Series, pd.Series]:
    """
    Z-score mean reversion on Close:
      - compute z = (close - SMA(window)) / STD(window)
      - entry when z <= z_entry (stretched down)
      - exit when z >= z_exit (mean reversion)

    Uses prior z (shift(1)) for deterministic bar-close logic.
    """
    close = df["Close"]
    mu = close.rolling(window, min_periods=1).mean()
    sd = close.rolling(window, min_periods=1).std(ddof=0).replace(0.0, np.nan)
    z = (close - mu) / sd
    z_prev = z.shift(1)
    entry = z_prev <= z_entry
    exit_sig = z_prev >= z_exit
    return entry.fillna(False), exit_sig.fillna(False)


def keltner_mean_reversion_signals(
    df: pd.DataFrame,
    ema_window: int = 48,
    atr_period: int = 14,
    atr_mult: float = 1.5,
) -> Tuple[pd.Series, pd.Series]:
    """
    Keltner-channel style mean reversion:
      - center = EMA(close, ema_window)
      - ATR via rolling mean TR (same ATR style as other templates)
      - lower = center - atr_mult * ATR
      - entry when close <= prior lower band
      - exit when close >= prior center
    """
    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    center = close.ewm(span=ema_window, adjust=False).mean()
    prev_close = close.shift(1)
    tr = np.maximum(
        high - low,
        np.maximum((high - prev_close).abs(), (low - prev_close).abs()),
    )
    atr = pd.Series(tr, index=df.index).rolling(atr_period, min_periods=1).mean()
    lower = center - (atr_mult * atr)
    entry = close <= lower.shift(1)
    exit_sig = close >= center.shift(1)
    return entry.fillna(False), exit_sig.fillna(False)


def cci_mean_reversion_signals(
    df: pd.DataFrame,
    period: int = 20,
    cci_entry: float = -150.0,
    cci_exit: float = -50.0,
) -> Tuple[pd.Series, pd.Series]:
    """
    CCI mean reversion (long-only):
      - typical price TP = (H+L+C)/3
      - CCI = (TP - SMA(TP)) / (0.015 * mean_dev)
      - entry when CCI <= cci_entry
      - exit when CCI >= cci_exit
    """
    high = df["High"]
    low = df["Low"]
    close = df["Close"]
    tp = (high + low + close) / 3.0
    sma = tp.rolling(period, min_periods=1).mean()
    mean_dev = (tp - sma).abs().rolling(period, min_periods=1).mean().replace(0.0, np.nan)
    cci = (tp - sma) / (0.015 * mean_dev)
    cci_prev = cci.shift(1)
    entry = cci_prev <= cci_entry
    exit_sig = cci_prev >= cci_exit
    return entry.fillna(False), exit_sig.fillna(False)


def return_reversal_signals(
    df: pd.DataFrame,
    lookback_bars: int = 1,
    drop_pct: float = 2.0,
) -> Tuple[pd.Series, pd.Series]:
    """
    Simple return-reversal mean reversion:
      - entry after a sharp drop over N bars
      - exit when price recovers above prior bar close (very simple)
    """
    close = df["Close"]
    ret = close.pct_change(lookback_bars)
    entry = ret.shift(1) <= (-abs(drop_pct) / 100.0)
    exit_sig = close >= close.shift(1)
    return entry.fillna(False), exit_sig.fillna(False)


def trend_conditioned_rsi_mr_signals(
    df: pd.DataFrame,
    ema_len: int = 200,
    rsi_period: int = 14,
    oversold: float = 30.0,
    exit_rsi: float = 50.0,
) -> Tuple[pd.Series, pd.Series]:
    """
    Trend-conditioned mean reversion:
      - only allow entries when close >= EMA(ema_len)
      - entry when RSI is oversold
      - exit when RSI recovers
    """
    close = df["Close"]
    ema = close.ewm(span=ema_len, adjust=False).mean()
    trend_ok = close >= ema.shift(1)

    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.rolling(rsi_period, min_periods=1).mean()
    avg_loss = loss.rolling(rsi_period, min_periods=1).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    rsi = rsi.fillna(50.0)

    entry = trend_ok & (rsi.shift(1) < oversold)
    exit_sig = rsi.shift(1) >= exit_rsi
    return entry.fillna(False), exit_sig.fillna(False)


def trend_pullback_continuation_signals(
    df: pd.DataFrame,
    fast_ema_bars: int = 50,
    slow_ema_bars: int = 200,
    pullback_bars: int = 5,
    reentry_breakout_bars: int = 20,
) -> Tuple[pd.Series, pd.Series]:
    """
    Trend pullback continuation (long-only):
      - regime/trend: fast EMA > slow EMA and close >= slow EMA
      - pullback: prior close briefly below fast EMA during recent pullback window
      - trigger: close reclaims fast EMA and breaks prior short-term high
      - exit: close below slow EMA or momentum roll (close < fast EMA)
    """
    close = df["Close"]
    fast = close.ewm(span=fast_ema_bars, adjust=False).mean()
    slow = close.ewm(span=slow_ema_bars, adjust=False).mean()

    trend_ok = (fast.shift(1) > slow.shift(1)) & (close >= slow.shift(1))
    pullback_recent = (close.shift(1) < fast.shift(1)).rolling(pullback_bars, min_periods=1).max() > 0
    breakout_level = close.rolling(reentry_breakout_bars, min_periods=1).max().shift(1)
    reclaims_fast = close >= fast.shift(1)
    breakout_ok = close >= breakout_level

    entry = trend_ok & pullback_recent & reclaims_fast & breakout_ok
    exit_sig = (close < slow.shift(1)) | (close < fast.shift(1))
    return entry.fillna(False), exit_sig.fillna(False)


def regime_gated_mean_reversion_signals(
    df: pd.DataFrame,
    regime_ema_len: int = 200,
    rsi_period: int = 14,
    oversold: float = 30.0,
    exit_rsi: float = 55.0,
    z_window: int = 48,
    z_entry: float = -1.5,
    z_exit: float = -0.2,
) -> Tuple[pd.Series, pd.Series]:
    """
    Regime-gated mean reversion (long-only):
      - entries only when close >= EMA(regime_ema_len)
      - require BOTH RSI oversold and negative z-score stretch
      - exit on RSI recovery, z-score normalization, or regime break
    """
    close = df["Close"]
    ema_regime = close.ewm(span=regime_ema_len, adjust=False).mean()
    regime_ok = close >= ema_regime.shift(1)

    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.rolling(rsi_period, min_periods=1).mean()
    avg_loss = loss.rolling(rsi_period, min_periods=1).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = (100.0 - (100.0 / (1.0 + rs))).fillna(50.0)

    mu = close.rolling(z_window, min_periods=1).mean()
    sd = close.rolling(z_window, min_periods=1).std(ddof=0).replace(0.0, np.nan)
    z = (close - mu) / sd

    entry = regime_ok & (rsi.shift(1) <= oversold) & (z.shift(1) <= z_entry)
    exit_sig = (rsi.shift(1) >= exit_rsi) | (z.shift(1) >= z_exit) | (close < ema_regime.shift(1))
    return entry.fillna(False), exit_sig.fillna(False)


# -----------------------------------------------------------------------------
# Template registry and parameter grids
# -----------------------------------------------------------------------------

TEMPLATES: Dict[str, Callable[..., Tuple[pd.Series, pd.Series]]] = {
    "breakout_continuation": breakout_continuation_signals,
    "trend_pullback": trend_pullback_signals,
    "rsi_mean_reversion": rsi_mean_reversion_signals,
    "volatility_spike_reversion": volatility_spike_reversion_signals,
    "volatility_breakout": volatility_breakout_signals,
    "ema_crossover": ema_crossover_signals,
    "bollinger_mean_reversion": bollinger_mean_reversion_signals,
    "volume_confirmed_breakout": volume_confirmed_breakout_signals,
    "macd_crossover": macd_crossover_signals,
    "donchian_breakout": donchian_breakout_signals,
    "atr_trend_trailing_stop": atr_trend_trailing_stop_signals,
    "vwap_mean_reversion": vwap_mean_reversion_signals,
    "zscore_mean_reversion": zscore_mean_reversion_signals,
    "keltner_mean_reversion": keltner_mean_reversion_signals,
    "cci_mean_reversion": cci_mean_reversion_signals,
    "return_reversal": return_reversal_signals,
    "trend_conditioned_rsi_mr": trend_conditioned_rsi_mr_signals,
    "trend_pullback_continuation": trend_pullback_continuation_signals,
    "regime_gated_mean_reversion": regime_gated_mean_reversion_signals,
}

PARAM_GRIDS: Dict[str, List[Dict[str, Any]]] = {
    "breakout_continuation": [
        {"lookback": 24, "recent_return_bars": 6, "min_recent_return_pct": 0.5},
        {"lookback": 48, "recent_return_bars": 12, "min_recent_return_pct": 0.3},
        {"lookback": 12, "recent_return_bars": 4, "min_recent_return_pct": 1.0},
    ],
    "trend_pullback": [
        {"trend_ma_bars": 48, "pullback_lookback": 12, "pullback_pct": 1.0},
        {"trend_ma_bars": 24, "pullback_lookback": 6, "pullback_pct": 1.5},
        {"trend_ma_bars": 72, "pullback_lookback": 18, "pullback_pct": 0.8},
    ],
    "rsi_mean_reversion": [
        {"rsi_period": 14, "oversold": 30.0, "exit_rsi": 50.0},
        {"rsi_period": 7, "oversold": 25.0, "exit_rsi": 45.0},
        {"rsi_period": 21, "oversold": 35.0, "exit_rsi": 55.0},
    ],
    "volatility_spike_reversion": [
        {"atr_period": 14, "drop_pct": 3.0, "atr_mult": 1.5, "exit_bars": 24},
        {"atr_period": 14, "drop_pct": 5.0, "atr_mult": 1.2, "exit_bars": 48},
        {"atr_period": 7, "drop_pct": 2.0, "atr_mult": 1.8, "exit_bars": 12},
    ],
    "volatility_breakout": [
        {"atr_period": 14, "breakout_lookback": 24, "atr_expansion_min": 1.2},
        {"atr_period": 14, "breakout_lookback": 48, "atr_expansion_min": 1.1},
        {"atr_period": 7, "breakout_lookback": 12, "atr_expansion_min": 1.3},
    ],
    "ema_crossover": [
        {"fast_ema_bars": 12, "slow_ema_bars": 48},
        {"fast_ema_bars": 12, "slow_ema_bars": 72},
        {"fast_ema_bars": 24, "slow_ema_bars": 48},
        {"fast_ema_bars": 24, "slow_ema_bars": 72},
    ],
    "bollinger_mean_reversion": [
        {"window": 20, "num_std": 2.0},
        {"window": 20, "num_std": 2.5},
        {"window": 30, "num_std": 2.0},
        {"window": 30, "num_std": 2.5},
    ],
    "volume_confirmed_breakout": [
        {"lookback": 24, "volume_ma_bars": 24, "vol_mult": 1.2},
        {"lookback": 24, "volume_ma_bars": 24, "vol_mult": 1.5},
        {"lookback": 48, "volume_ma_bars": 48, "vol_mult": 1.2},
        {"lookback": 48, "volume_ma_bars": 48, "vol_mult": 1.5},
    ],
    "macd_crossover": [
        {"macd_fast": 12, "macd_slow": 26, "macd_signal": 9},
        {"macd_fast": 8, "macd_slow": 21, "macd_signal": 5},
        {"macd_fast": 12, "macd_slow": 36, "macd_signal": 9},
    ],
    "donchian_breakout": [
        {"lookback": 24},
        {"lookback": 48},
    ],
    "atr_trend_trailing_stop": [
        {"trend_ema_bars": 100, "atr_period": 14, "atr_mult": 1.5},
        {"trend_ema_bars": 100, "atr_period": 14, "atr_mult": 2.0},
        {"trend_ema_bars": 200, "atr_period": 14, "atr_mult": 1.5},
    ],
    "vwap_mean_reversion": [
        {"window": 24},
        {"window": 48},
    ],
    "zscore_mean_reversion": [
        {"window": 48, "z_entry": -2.0, "z_exit": -0.5},
        {"window": 72, "z_entry": -2.0, "z_exit": -0.5},
        {"window": 48, "z_entry": -2.5, "z_exit": -1.0},
    ],
    "keltner_mean_reversion": [
        {"ema_window": 48, "atr_period": 14, "atr_mult": 1.5},
        {"ema_window": 48, "atr_period": 14, "atr_mult": 2.0},
        {"ema_window": 72, "atr_period": 14, "atr_mult": 1.5},
    ],
    "cci_mean_reversion": [
        {"period": 20, "cci_entry": -150.0, "cci_exit": -50.0},
        {"period": 30, "cci_entry": -150.0, "cci_exit": -50.0},
        {"period": 20, "cci_entry": -200.0, "cci_exit": -50.0},
    ],
    "return_reversal": [
        {"lookback_bars": 1, "drop_pct": 2.0},
        {"lookback_bars": 1, "drop_pct": 3.0},
        {"lookback_bars": 3, "drop_pct": 4.0},
    ],
    "trend_conditioned_rsi_mr": [
        {"ema_len": 200, "rsi_period": 14, "oversold": 30.0, "exit_rsi": 50.0},
        {"ema_len": 200, "rsi_period": 14, "oversold": 35.0, "exit_rsi": 50.0},
        {"ema_len": 100, "rsi_period": 14, "oversold": 30.0, "exit_rsi": 50.0},
    ],
    "trend_pullback_continuation": [
        {"fast_ema_bars": 50, "slow_ema_bars": 200, "pullback_bars": 5, "reentry_breakout_bars": 20},
        {"fast_ema_bars": 34, "slow_ema_bars": 144, "pullback_bars": 4, "reentry_breakout_bars": 14},
        {"fast_ema_bars": 72, "slow_ema_bars": 200, "pullback_bars": 6, "reentry_breakout_bars": 24},
    ],
    "regime_gated_mean_reversion": [
        {
            "regime_ema_len": 200,
            "rsi_period": 14,
            "oversold": 30.0,
            "exit_rsi": 55.0,
            "z_window": 48,
            "z_entry": -1.5,
            "z_exit": -0.2,
        },
        {
            "regime_ema_len": 200,
            "rsi_period": 14,
            "oversold": 28.0,
            "exit_rsi": 52.0,
            "z_window": 72,
            "z_entry": -1.8,
            "z_exit": -0.3,
        },
        {
            "regime_ema_len": 144,
            "rsi_period": 10,
            "oversold": 32.0,
            "exit_rsi": 55.0,
            "z_window": 48,
            "z_entry": -1.4,
            "z_exit": -0.1,
        },
    ],
}


# -----------------------------------------------------------------------------
# Load data
# -----------------------------------------------------------------------------

def load_ohlcv(path: Path) -> pd.DataFrame:
    """Load OHLCV CSV; set Timestamp as index.

    Required columns:
      - Timestamp, High, Low, Close
    """
    df = pd.read_csv(path)

    required_cols = ["Timestamp", "High", "Low", "Close"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {path}: {missing}. Found columns={list(df.columns)}")

    df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
    df = df.dropna(subset=["Timestamp"])
    df = df.set_index("Timestamp").sort_index()
    return df


def survives_filter(row: pd.Series) -> bool:
    """
    True if candidate passes pre-filter: cagr > 0, profit_factor > 1.0,
    max_drawdown < 0.60 (drawdown less severe than 60%), n_trades >= 30.
    """
    return (
        row["cagr"] > FILTER_CAGR_MIN
        and row["profit_factor"] > FILTER_PROFIT_FACTOR_MIN
        and row["max_drawdown"] < FILTER_MAX_DRAWDOWN_MAX
        and row["n_trades"] >= FILTER_N_TRADES_MIN
    )


# -----------------------------------------------------------------------------
# Main: run all templates × param sets, collect metrics, write CSV, print top
# -----------------------------------------------------------------------------

def run_screener(
    data_path: Path,
    fee_bps: float = 25.0,
    slippage_bps: float = 10.0,
    output_path: Path | None = None,
    filtered_output_path: Path | None = None,
    asset: str = DEFAULT_ASSET,
    max_hold_bars: int | None = None,
) -> pd.DataFrame:
    """
    Run all strategy templates over their parameter grids; return results DataFrame
    with survives_filter column. Writes full CSV and filtered CSV.
    """
    if output_path is None or filtered_output_path is None:
        safe_asset = (asset or DEFAULT_ASSET).strip().lower()
        output_path = OUTPUT_DIR / f"fast_strategy_screener_results_{safe_asset}.csv"
        filtered_output_path = OUTPUT_DIR / f"fast_strategy_screener_filtered_results_{safe_asset}.csv"

    df = load_ohlcv(data_path)
    rows: List[Dict[str, Any]] = []

    for strategy_name, signal_fn in TEMPLATES.items():
        param_list = PARAM_GRIDS.get(strategy_name, [{}])
        for param_set in param_list:
            entry_signal, exit_signal = signal_fn(df, **param_set)
            _, _, metrics = run_backtest(
                df,
                entry_signal,
                exit_signal,
                fee_bps,
                slippage_bps,
                max_hold_bars=max_hold_bars,
            )
            rows.append({
                "strategy_name": strategy_name,
                "parameter_set": str(param_set),
                "total_return_pct": metrics["total_return_pct"],
                "cagr": metrics["cagr"],
                "max_drawdown": metrics["max_drawdown"],
                "calmar": metrics["calmar"],
                "win_rate": metrics["win_rate"],
                "profit_factor": metrics["profit_factor"],
                "n_trades": metrics["n_trades"],
                "avg_hold_hours": metrics["avg_hold_hours"],
            })

    results = pd.DataFrame(rows)
    results["passes_filter"] = results.apply(survives_filter, axis=1)
    # Backward compatibility with previous output column name.
    results["survives_filter"] = results["passes_filter"]

    results = _compute_ranked_score(results)

    # Ranking: cagr desc, profit_factor desc, max_drawdown asc (less severe = better)
    results = results.sort_values(
        by=["score", "cagr", "profit_factor", "max_drawdown"],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_path, index=False)

    filtered = results[results["passes_filter"]].copy()
    filtered.to_csv(filtered_output_path, index=False)

    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fast strategy screener: evaluate deterministic templates on hourly OHLCV (asset-agnostic).",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=None,
        help="Path to OHLCV CSV (must include Timestamp, High, Low, Close). If omitted, inferred from --asset.",
    )
    parser.add_argument(
        "--asset",
        type=str,
        default=DEFAULT_ASSET,
        help="Base asset ticker used to infer default data path (repo convention: data/{asset}usd_3600s_2019-01-01_to_2025-12-30.csv).",
    )
    parser.add_argument(
        "--fee_bps",
        type=float,
        default=25.0,
        help="Fee in basis points (default 25)",
    )
    parser.add_argument(
        "--slippage_bps",
        type=float,
        default=10.0,
        help="Slippage in basis points (default 10)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output CSV path",
    )
    parser.add_argument(
        "--filtered_output",
        type=Path,
        default=None,
        help="Filtered results CSV path",
    )
    parser.add_argument(
        "--filter_cagr_min",
        type=float,
        default=FILTER_CAGR_MIN,
        help="Survival filter: require cagr >= this value.",
    )
    parser.add_argument(
        "--filter_profit_factor_min",
        type=float,
        default=FILTER_PROFIT_FACTOR_MIN,
        help="Survival filter: require profit_factor >= this value.",
    )
    parser.add_argument(
        "--filter_max_drawdown_max",
        type=float,
        default=FILTER_MAX_DRAWDOWN_MAX,
        help="Survival filter: require max_drawdown < this value (e.g. 0.60 = 60%).",
    )
    parser.add_argument(
        "--filter_n_trades_min",
        type=int,
        default=FILTER_N_TRADES_MIN,
        help="Survival filter: require n_trades >= this value.",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=10,
        help="Number of top candidates to print (default 10)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["ranking", "filter", "both"],
        default="both",
        help="Output mode: ranking, filter, or both (default both).",
    )
    parser.add_argument(
        "--w_cagr",
        type=float,
        default=0.4,
        help="Ranking score weight for CAGR (higher = better).",
    )
    parser.add_argument(
        "--w_calmar",
        type=float,
        default=0.3,
        help="Ranking score weight for Calmar (higher = better).",
    )
    parser.add_argument(
        "--w_profit_factor",
        type=float,
        default=0.2,
        help="Ranking score weight for Profit Factor (higher = better).",
    )
    parser.add_argument(
        "--w_drawdown_good",
        type=float,
        default=0.1,
        help="Ranking score weight for drawdown-good rank (smaller DD -> better).",
    )
    parser.add_argument(
        "--ranked_output",
        type=Path,
        default=None,
        help="Ranked results CSV path",
    )
    parser.add_argument(
        "--max_hold_bars",
        type=int,
        default=None,
        help="Optional risk cap: force an exit after holding for this many bars (in bars).",
    )
    args = parser.parse_args()

    asset = (args.asset or DEFAULT_ASSET).strip().lower()
    data_path: Path
    if args.data is not None:
        data_path = args.data
    else:
        data_path = Path(str(DEFAULT_DATA_TEMPLATE).format(asset=asset))
        if not data_path.exists():
            raise FileNotFoundError(
                f"Default data inferred for asset={asset} but file does not exist: {data_path}. "
                f"Pass --data explicitly or add a matching CSV under data/."
            )

    output_path: Path
    filtered_output_path: Path
    if args.output is not None:
        output_path = args.output
    else:
        output_path = OUTPUT_DIR / f"fast_strategy_screener_results_{asset}.csv"

    if args.filtered_output is not None:
        filtered_output_path = args.filtered_output
    else:
        filtered_output_path = OUTPUT_DIR / f"fast_strategy_screener_filtered_results_{asset}.csv"

    # Apply filter overrides (survives_filter reads module-level globals).
    # Using globals() avoids Python "global used prior to declaration" scoping issues.
    globals()["FILTER_CAGR_MIN"] = args.filter_cagr_min
    globals()["FILTER_PROFIT_FACTOR_MIN"] = args.filter_profit_factor_min
    globals()["FILTER_MAX_DRAWDOWN_MAX"] = args.filter_max_drawdown_max
    globals()["FILTER_N_TRADES_MIN"] = args.filter_n_trades_min

    # Apply scoring weights (score uses percentile ranks; weights are normalized).
    globals()["SCORE_W_CAGR"] = args.w_cagr
    globals()["SCORE_W_CALMAR"] = args.w_calmar
    globals()["SCORE_W_PROFIT_FACTOR"] = args.w_profit_factor
    globals()["SCORE_W_DRAWDOWN_GOOD"] = args.w_drawdown_good

    results = run_screener(
        data_path=data_path,
        fee_bps=args.fee_bps,
        slippage_bps=args.slippage_bps,
        output_path=output_path,
        filtered_output_path=filtered_output_path,
        asset=asset,
        max_hold_bars=args.max_hold_bars,
    )

    ranked_output_path: Path
    if args.ranked_output is not None:
        ranked_output_path = args.ranked_output
    else:
        ranked_output_path = OUTPUT_DIR / f"fast_strategy_screener_ranked_{asset}.csv"
    ranked_output_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(ranked_output_path, index=False)

    n_total = len(results)
    n_surviving = int(results["passes_filter"].sum())

    filter_cagr_min = globals().get("FILTER_CAGR_MIN", FILTER_CAGR_MIN)
    filter_profit_factor_min = globals().get("FILTER_PROFIT_FACTOR_MIN", FILTER_PROFIT_FACTOR_MIN)
    filter_max_drawdown_max = globals().get("FILTER_MAX_DRAWDOWN_MAX", FILTER_MAX_DRAWDOWN_MAX)
    filter_n_trades_min = globals().get("FILTER_N_TRADES_MIN", FILTER_N_TRADES_MIN)

    cagr_pass = int((results["cagr"] > filter_cagr_min).sum())
    pf_pass = int((results["profit_factor"] > filter_profit_factor_min).sum())
    dd_pass = int((results["max_drawdown"] < filter_max_drawdown_max).sum())
    nt_pass = int((results["n_trades"] >= filter_n_trades_min).sum())

    def fail_reason(row: pd.Series) -> str:
        if row["cagr"] <= filter_cagr_min:
            return f"cagr<={filter_cagr_min}"
        if row["profit_factor"] <= filter_profit_factor_min:
            return f"PF<={filter_profit_factor_min}"
        if row["max_drawdown"] >= filter_max_drawdown_max:
            return f"DD>={filter_max_drawdown_max}"
        if row["n_trades"] < filter_n_trades_min:
            return f"n_trades<{filter_n_trades_min}"
        return "unknown"

    print("Fast Strategy Screener")
    print(f"  Asset: {asset}")
    print(f"  Total candidates tested: {n_total}")
    print(f"  Candidates passing filter: {n_surviving}")
    print(f"  Mode: {args.mode}")
    print(
        f"  Filter: cagr>{filter_cagr_min} | PF>{filter_profit_factor_min} | "
        f"DD<{filter_max_drawdown_max} | n_trades>={filter_n_trades_min}"
    )
    print(
        "  Score: "
        f"{args.w_cagr:.3f}*CAGR_rank + {args.w_calmar:.3f}*Calmar_rank + "
        f"{args.w_profit_factor:.3f}*PF_rank + {args.w_drawdown_good:.3f}*(1-DD_rank)"
    )
    print("  Pass counts: "
          f"cagr={cagr_pass}/{n_total}, PF={pf_pass}/{n_total}, DD={dd_pass}/{n_total}, n_trades={nt_pass}/{n_total}")

    if args.mode in ("ranking", "both"):
        print("-" * 80)
        print(f"Top {args.top} Strategies (Ranked):")
        top_full = results.head(args.top)
        for _, row in top_full.iterrows():
            ps = row["parameter_set"]
            ps_short = (ps[:57] + "...") if len(ps) > 60 else ps
            status = "PASS" if bool(row["passes_filter"]) else "FAIL"
            print(
                f"  {row['strategy_name']} | {ps_short} | "
                f"score={row['score']:.4f} | "
                f"CAGR={row['cagr']*100:.1f}% DD={row['max_drawdown']*100:.1f}% "
                f"Calmar={row['calmar']:.2f} PF={row['profit_factor']:.2f} "
                f"trades={int(row['n_trades'])} | {status}"
            )

    if args.mode in ("filter", "both"):
        survivors = results[results["passes_filter"]]
        print("-" * 80)
        if survivors.empty:
            print("No survivors under the current filter thresholds.")
            if args.mode == "filter":
                print("Top ranked candidates are still available in ranked CSV output.")
        else:
            print("Top survivors:")
            top_surv = survivors.head(args.top)
            for _, row in top_surv.iterrows():
                ps = row["parameter_set"]
                ps_short = (ps[:57] + "...") if len(ps) > 60 else ps
                reason = "" if bool(row["passes_filter"]) else f" ({fail_reason(row)})"
                print(
                    f"  {row['strategy_name']} | {ps_short} | "
                    f"score={row['score']:.4f} | "
                    f"CAGR={row['cagr']*100:.1f}% PF={row['profit_factor']:.2f} "
                    f"DD={row['max_drawdown']*100:.1f}% n={row['n_trades']}{reason}"
                )

    print("-" * 80)
    print(f"Full results: {output_path}")
    print(f"Filtered results: {filtered_output_path}")
    print(f"Ranked results: {ranked_output_path}")


if __name__ == "__main__":
    main()
