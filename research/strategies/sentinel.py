"""
Sentinel Strategy - Momentum Trend Following with Capital Protection

Philosophy:
- Catch the BIG moves in BTC (these make all the money)
- Stay out of chop/consolidation
- Use volatility-based stops that respect BTC's swings
- Risk a fixed % per trade, never more
- Simple rules that don't overfit

Key insight: BTC has HUGE runs (+100%, +200%) followed by brutal corrections.
We want to catch the runs and survive the corrections.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple
import pandas as pd
import numpy as np


@dataclass
class SentinelParams:
    # Trend regime - keep it simple
    regime_sma: int = 200           # Bull/bear regime
    trend_sma: int = 50             # Medium-term trend
    
    # Entry - momentum breakout
    breakout_period: int = 20       # Breakout above N-bar high
    volume_confirm_bars: int = 5    # Volume should be above average
    
    # ATR for position sizing and stops
    atr_period: int = 14
    
    # Risk management
    risk_per_trade_pct: float = 2.0     # Risk 2% per trade
    initial_stop_atr: float = 3.0       # Stop at 3x ATR
    trailing_stop_atr: float = 3.0      # Trail at 3x ATR (from peak)
    
    # Exit conditions  
    max_hold_bars: int = 504            # 3 weeks max (504 hours)
    profit_target_pct: float | None = None  # Optional profit target (None = let it run)
    
    # Circuit breaker
    max_drawdown_pct: float = 25.0      # Stop trading if DD > 25%


def _sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n, min_periods=n).mean()


def _atr(df: pd.DataFrame, n: int) -> pd.Series:
    high = df["High"]
    low = df["Low"]
    close = df["Close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low).abs(),
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(n, min_periods=n).mean()


def build_signals_sentinel(df: pd.DataFrame, params: SentinelParams) -> pd.DataFrame:
    """
    Build entry/exit signals for Sentinel strategy.
    Simple momentum breakout in uptrend.
    """
    df = df.copy()
    close = df["Close"].astype(float)
    high = df["High"].astype(float)
    volume = df["Volume"].astype(float)
    
    # Trend indicators
    sma_regime = _sma(close, params.regime_sma)
    sma_trend = _sma(close, params.trend_sma)
    
    # ATR
    atr = _atr(df, params.atr_period)
    
    # Regime: bull = price above 200 SMA AND 50 SMA above 200 SMA
    bull_regime = (close > sma_regime) & (sma_trend > sma_regime)
    
    # Trend direction: 50 SMA rising
    trend_up = sma_trend > sma_trend.shift(3)  # Rising over 3 bars
    
    # Breakout: close above N-bar high
    rolling_high = high.rolling(params.breakout_period).max().shift(1)
    breakout = close > rolling_high
    
    # Volume confirmation: above recent average
    vol_avg = volume.rolling(params.volume_confirm_bars).mean().shift(1)
    vol_ok = volume > vol_avg * 0.8  # At least 80% of average
    
    # ENTRY: Bull regime + Uptrend + Breakout + Volume OK
    enter_long = bull_regime & trend_up & breakout & vol_ok
    
    # EXIT signal: trend breakdown
    # Price closes below 50 SMA OR 50 SMA crosses below 200 SMA
    trend_break = (close < sma_trend) | (sma_trend < sma_regime)
    exit_long = trend_break
    
    # Stop price for position sizing
    stop_price = close - (params.initial_stop_atr * atr)
    
    sig = pd.DataFrame({
        "enter_long": enter_long.fillna(False).astype(bool),
        "exit_long": exit_long.fillna(False).astype(bool),
        "stop_price": stop_price,
        "atr": atr,
    })
    
    return sig


def run_sentinel_backtest(
    df: pd.DataFrame,
    signals: pd.DataFrame,
    params: SentinelParams,
    initial_cash: float = 10000.0,
    fee_bps: float = 6.0,
    slippage_bps: float = 10.0,
    min_notional_usd: float = 10.0,
) -> Tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Backtest engine for Sentinel strategy.
    """
    df = df.copy().reset_index(drop=True)
    sig = signals.copy().reset_index(drop=True)
    
    cash = float(initial_cash)
    btc = 0.0
    
    trades = []
    equity_rows = []
    
    in_pos = False
    entry_px = None
    entry_ts = None
    entry_cost = None
    entry_qty = None
    
    highest_since_entry = 0.0
    current_stop = 0.0
    bars_in_position = 0
    
    peak_equity = initial_cash
    trading_enabled = True
    
    def apply_cost(price: float, side: str) -> float:
        slip = slippage_bps / 10_000.0
        if side == "BUY":
            return price * (1 + slip)
        return price * (1 - slip)
    
    def fee_amount(notional: float) -> float:
        return abs(notional) * (fee_bps / 10_000.0)
    
    for i in range(len(df)):
        ts = df.loc[i, "Timestamp"]
        price = float(df.loc[i, "Close"])
        high_price = float(df.loc[i, "High"])
        
        equity = cash + btc * price
        equity_rows.append({
            "Timestamp": ts,
            "equity": equity,
            "cash": cash,
            "btc": btc,
            "price": price,
            "in_position": in_pos,
        })
        
        # Circuit breaker check
        if equity > peak_equity:
            peak_equity = equity
        current_dd_pct = ((peak_equity - equity) / peak_equity) * 100
        if current_dd_pct >= params.max_drawdown_pct:
            trading_enabled = False
        elif current_dd_pct < params.max_drawdown_pct * 0.5:
            trading_enabled = True
        
        enter = bool(sig.loc[i, "enter_long"])
        exit_sig = bool(sig.loc[i, "exit_long"])
        atr_val = float(sig.loc[i, "atr"]) if not pd.isna(sig.loc[i, "atr"]) else 0
        
        # Position management
        if in_pos:
            bars_in_position += 1
            
            # Update trailing stop
            if high_price > highest_since_entry:
                highest_since_entry = high_price
                if atr_val > 0:
                    new_stop = highest_since_entry - (params.trailing_stop_atr * atr_val)
                    current_stop = max(current_stop, new_stop)
            
            # Check exit conditions
            should_exit = False
            exit_reason = ""
            
            # 1. Signal exit (trend breakdown)
            if exit_sig:
                should_exit = True
                exit_reason = "trend"
            
            # 2. Stop loss
            if price <= current_stop:
                should_exit = True
                exit_reason = "stop"
            
            # 3. Time exit
            if bars_in_position >= params.max_hold_bars:
                should_exit = True
                exit_reason = "time"
            
            # 4. Profit target (if set)
            if params.profit_target_pct is not None and entry_px:
                target = entry_px * (1 + params.profit_target_pct / 100)
                if price >= target:
                    should_exit = True
                    exit_reason = "target"
            
            if should_exit:
                fill = apply_cost(price, "SELL")
                notional = btc * fill
                fee = fee_amount(notional)
                
                exit_net = notional - fee
                pnl_usd = exit_net - entry_cost
                pnl_pct = pnl_usd / entry_cost if entry_cost > 0 else 0
                
                trades.append({
                    "entry_ts": entry_ts,
                    "exit_ts": ts,
                    "entry_px": entry_px,
                    "exit_px": fill,
                    "btc_qty": entry_qty,
                    "entry_cost": entry_cost,
                    "exit_net": exit_net,
                    "pnl_usd": pnl_usd,
                    "pnl_pct": pnl_pct,
                    "exit_reason": exit_reason,
                    "bars_held": bars_in_position,
                })
                
                cash = cash + notional - fee
                btc = 0.0
                in_pos = False
                entry_px = None
                entry_ts = None
                entry_cost = None
                entry_qty = None
                highest_since_entry = 0.0
                current_stop = 0.0
                bars_in_position = 0
        
        # Entry logic
        if (not in_pos) and enter and trading_enabled:
            stop_price = float(sig.loc[i, "stop_price"])
            
            if stop_price > 0 and stop_price < price and atr_val > 0:
                risk_per_unit = price - stop_price
                
                # Position size: risk_budget / risk_per_unit
                risk_budget = equity * (params.risk_per_trade_pct / 100.0)
                target_qty = risk_budget / risk_per_unit
                target_usd = target_qty * price
                target_usd = min(target_usd, cash * 0.95)
                
                if target_usd >= min_notional_usd:
                    fill = apply_cost(price, "BUY")
                    qty = target_usd / fill
                    notional = qty * fill
                    fee = fee_amount(notional)
                    total_cost = notional + fee
                    
                    if total_cost <= cash and qty > 0:
                        cash -= total_cost
                        btc = qty
                        in_pos = True
                        
                        entry_px = fill
                        entry_ts = ts
                        entry_cost = total_cost
                        entry_qty = qty
                        highest_since_entry = price
                        current_stop = stop_price
                        bars_in_position = 0
    
    # Close any open position
    if in_pos:
        price = float(df["Close"].iloc[-1])
        ts = df["Timestamp"].iloc[-1]
        fill = apply_cost(price, "SELL")
        notional = btc * fill
        fee = fee_amount(notional)
        
        exit_net = notional - fee
        pnl_usd = exit_net - entry_cost
        pnl_pct = pnl_usd / entry_cost if entry_cost > 0 else 0
        
        trades.append({
            "entry_ts": entry_ts,
            "exit_ts": ts,
            "entry_px": entry_px,
            "exit_px": fill,
            "btc_qty": entry_qty,
            "entry_cost": entry_cost,
            "exit_net": exit_net,
            "pnl_usd": pnl_usd,
            "pnl_pct": pnl_pct,
            "exit_reason": "end",
            "bars_held": bars_in_position,
        })
        cash = cash + notional - fee
        btc = 0.0
    
    trades_df = pd.DataFrame(trades)
    equity_df = pd.DataFrame(equity_rows)
    
    summary = _compute_summary(trades_df, equity_df, initial_cash)
    
    return trades_df, equity_df, summary


def _compute_summary(trades: pd.DataFrame, equity_curve: pd.DataFrame, initial_cash: float) -> dict:
    """Compute performance metrics."""
    eq = equity_curve["equity"].values
    
    total_return = (eq[-1] / eq[0]) - 1.0 if len(eq) > 1 else 0.0
    
    # Drawdown
    peak = np.maximum.accumulate(eq)
    dd = (eq / np.maximum(peak, 1e-12)) - 1.0
    max_dd = float(dd.min()) if len(dd) else 0.0
    
    # Time underwater
    underwater = dd < 0
    max_underwater = 0
    current = 0
    for u in underwater:
        if u:
            current += 1
            max_underwater = max(max_underwater, current)
        else:
            current = 0
    
    # Returns analysis
    rets = np.diff(eq) / np.maximum(eq[:-1], 1e-12) if len(eq) > 1 else np.array([0.0])
    
    years = len(eq) / 8760
    annual_return = ((1 + total_return) ** (1 / max(years, 0.01))) - 1 if years > 0 else 0
    
    if len(rets) > 1 and rets.std() > 0:
        sharpe_annual = (rets.mean() / rets.std()) * np.sqrt(8760)
    else:
        sharpe_annual = 0.0
    
    downside = rets[rets < 0]
    if len(downside) > 1 and downside.std() > 0:
        sortino_annual = (rets.mean() / downside.std()) * np.sqrt(8760)
    else:
        sortino_annual = 0.0
    
    calmar = abs(annual_return / (max_dd + 1e-12)) if max_dd != 0 else 0
    
    # Trade stats
    if trades.empty:
        trade_stats = {
            "n_trades": 0, "win_rate_pct": 0, "avg_trade_pct": 0,
            "avg_winner_pct": 0, "avg_loser_pct": 0, "profit_factor": 0,
            "best_trade_pct": 0, "worst_trade_pct": 0, "avg_bars_held": 0,
            "exit_reasons": {},
        }
    else:
        pnl = trades["pnl_pct"].values
        winners = pnl[pnl > 0]
        losers = pnl[pnl < 0]
        
        gross_profit = winners.sum() if len(winners) else 0
        gross_loss = abs(losers.sum()) if len(losers) else 1e-10
        
        trade_stats = {
            "n_trades": len(trades),
            "win_rate_pct": (len(winners) / len(trades)) * 100,
            "avg_trade_pct": float(pnl.mean() * 100),
            "avg_winner_pct": float(winners.mean() * 100) if len(winners) else 0,
            "avg_loser_pct": float(losers.mean() * 100) if len(losers) else 0,
            "profit_factor": gross_profit / gross_loss,
            "best_trade_pct": float(pnl.max() * 100),
            "worst_trade_pct": float(pnl.min() * 100),
            "avg_bars_held": float(trades["bars_held"].mean()),
            "exit_reasons": trades["exit_reason"].value_counts().to_dict() if "exit_reason" in trades.columns else {},
        }
    
    return {
        "total_return_pct": float(total_return * 100),
        "annual_return_pct": float(annual_return * 100),
        "final_equity": float(eq[-1]),
        "max_drawdown_pct": float(max_dd * 100),
        "max_underwater_bars": int(max_underwater),
        "sharpe_annual": float(sharpe_annual),
        "sortino_annual": float(sortino_annual),
        "calmar_ratio": float(calmar),
        "years_tested": float(years),
        **trade_stats,
    }






























