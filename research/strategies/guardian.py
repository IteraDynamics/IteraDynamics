"""
Guardian Strategy - Capital Preservation + Active Trading

Philosophy:
- Trade WITH the trend, not against it (multi-timeframe confirmation)
- Risk a fixed % of equity per trade (position sizing based on stop distance)
- Use ATR-based trailing stops to let winners run, cut losers fast
- Drawdown circuit breaker - step aside when equity drops significantly
- Cooldown after losses to avoid revenge trading
- Volatility regime filter - only trade when there's enough movement

This is NOT a savings account (we trade actively) but it's NOT a casino either.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple
import pandas as pd
import numpy as np


@dataclass
class GuardianParams:
    # Trend filters
    sma_fast: int = 20          # Short-term trend
    sma_slow: int = 50          # Medium-term trend  
    sma_regime: int = 200       # Long-term regime filter
    
    # Entry conditions
    rsi_period: int = 14        # RSI for overbought/oversold
    rsi_entry_min: float = 30   # Don't buy if RSI too low (catching falling knife)
    rsi_entry_max: float = 70   # Don't buy if RSI too high (chasing)
    
    # Volatility
    atr_period: int = 14
    atr_filter_lookback: int = 50  # Compare current ATR to median
    min_atr_ratio: float = 0.8     # Min ATR vs median (avoid dead markets)
    max_atr_ratio: float = 3.0     # Max ATR vs median (avoid chaos)
    
    # Risk management - THE KEY TO CAPITAL PRESERVATION
    risk_per_trade_pct: float = 1.0   # Risk 1% of equity per trade
    atr_stop_mult: float = 2.0        # Stop loss = 2x ATR below entry
    atr_trail_mult: float = 2.5       # Trailing stop = 2.5x ATR from high
    
    # Circuit breakers
    max_drawdown_pct: float = 15.0    # Stop trading if DD exceeds this
    cooldown_bars_after_loss: int = 6 # Wait 6 bars after a loss
    
    # Time exit (optional safety valve)
    max_hold_bars: int | None = 168   # Max 1 week hold (168 hours)


def _sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n, min_periods=n).mean()


def _ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()


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


def _rsi(close: pd.Series, period: int) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    
    avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / (avg_loss + 1e-10)
    return 100 - (100 / (1 + rs))


def build_signals_guardian(df: pd.DataFrame, params: GuardianParams) -> pd.DataFrame:
    """
    Build entry/exit signals for Guardian strategy.
    
    Returns DataFrame with columns:
        - enter_long: bool
        - exit_long: bool
        - stop_price: float (ATR-based stop)
        - risk_pct: float (for position sizing)
    """
    df = df.copy()
    close = df["Close"].astype(float)
    high = df["High"].astype(float)
    low = df["Low"].astype(float)
    
    n = len(df)
    
    # Calculate indicators
    sma_fast = _sma(close, params.sma_fast)
    sma_slow = _sma(close, params.sma_slow)
    sma_regime = _sma(close, params.sma_regime)
    
    atr = _atr(df, params.atr_period)
    atr_median = atr.rolling(params.atr_filter_lookback, min_periods=params.atr_filter_lookback).median()
    atr_ratio = atr / (atr_median + 1e-10)
    
    rsi = _rsi(close, params.rsi_period)
    
    # Trend conditions
    # Bull regime: price above 200 SMA
    bull_regime = close > sma_regime
    
    # Uptrend: fast SMA above slow SMA (relaxed - don't require both rising)
    uptrend = sma_fast > sma_slow
    
    # Momentum: price making higher highs
    higher_close = close > close.shift(1)
    higher_low = low > low.shift(1)
    momentum_up = higher_close | higher_low
    
    # Pullback entry: price near or above fast SMA (within 1%)
    near_sma = (close >= sma_fast * 0.99) & (close <= sma_fast * 1.03)
    price_above_fast = close > sma_fast
    pullback_entry = near_sma | price_above_fast
    
    # Breakout entry: close above recent high (5-bar high)
    recent_high = high.rolling(5).max().shift(1)
    breakout = close > recent_high
    
    # RSI filter
    rsi_ok = (rsi >= params.rsi_entry_min) & (rsi <= params.rsi_entry_max)
    
    # Volatility filter (not too dead, not too crazy)
    vol_ok = (atr_ratio >= params.min_atr_ratio) & (atr_ratio <= params.max_atr_ratio)
    
    # ENTRY: Bull regime + Uptrend + (Pullback OR Breakout) + Momentum + RSI OK + Vol OK
    # More permissive: need trend alignment + any of the entry triggers
    enter_long = bull_regime & uptrend & (pullback_entry | breakout) & momentum_up & rsi_ok & vol_ok
    
    # EXIT: Price crosses below slow SMA OR trend breaks
    downtrend = sma_fast < sma_slow
    price_below_slow = close < sma_slow
    exit_long = downtrend | price_below_slow
    
    # Calculate stop price and risk for position sizing
    stop_price = close - (params.atr_stop_mult * atr)
    risk_pct_per_unit = (close - stop_price) / close  # % risk per unit
    
    # Trailing stop high-water mark (will be used in backtest)
    trail_stop_distance = params.atr_trail_mult * atr
    
    sig = pd.DataFrame({
        "enter_long": enter_long.fillna(False).astype(bool),
        "exit_long": exit_long.fillna(False).astype(bool),
        "stop_price": stop_price,
        "trail_distance": trail_stop_distance,
        "risk_pct_per_unit": risk_pct_per_unit,
        "atr": atr,
    })
    
    return sig


def run_guardian_backtest(
    df: pd.DataFrame,
    signals: pd.DataFrame,
    params: GuardianParams,
    initial_cash: float = 100.0,
    fee_bps: float = 6.0,
    slippage_bps: float = 10.0,
    min_notional_usd: float = 5.0,
) -> Tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Custom backtest engine for Guardian that handles:
    - Risk-based position sizing
    - ATR trailing stops
    - Drawdown circuit breaker
    - Loss cooldown
    
    Returns: (trades_df, equity_df, summary_dict)
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
    entry_notional = None
    entry_qty = None
    
    # Trailing stop tracking
    highest_since_entry = 0.0
    current_stop = 0.0
    
    # Circuit breaker state
    peak_equity = initial_cash
    trading_enabled = True
    
    # Cooldown tracking
    bars_since_last_loss = 999  # Start high so we can trade
    
    # Hold time tracking
    bars_in_position = 0
    
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
        
        # Mark equity
        equity = cash + btc * price
        equity_rows.append({
            "Timestamp": ts,
            "equity": equity,
            "cash": cash,
            "btc": btc,
            "price": price,
            "in_position": in_pos,
            "trading_enabled": trading_enabled,
        })
        
        # Update peak equity and check circuit breaker
        if equity > peak_equity:
            peak_equity = equity
        
        current_dd_pct = ((peak_equity - equity) / peak_equity) * 100
        if current_dd_pct >= params.max_drawdown_pct:
            trading_enabled = False
        elif current_dd_pct < params.max_drawdown_pct * 0.5:
            # Re-enable trading when DD recovers to half the threshold
            trading_enabled = True
        
        # Update cooldown counter
        bars_since_last_loss += 1
        
        enter = bool(sig.loc[i, "enter_long"]) if "enter_long" in sig.columns else False
        exit_sig = bool(sig.loc[i, "exit_long"]) if "exit_long" in sig.columns else False
        
        # In position - check exits
        if in_pos:
            bars_in_position += 1
            
            # Update trailing stop
            if high_price > highest_since_entry:
                highest_since_entry = high_price
                trail_dist = float(sig.loc[i, "trail_distance"]) if not pd.isna(sig.loc[i, "trail_distance"]) else 0
                new_trail_stop = highest_since_entry - trail_dist
                current_stop = max(current_stop, new_trail_stop)
            
            # Check exit conditions
            should_exit = False
            exit_reason = ""
            
            # 1. Signal exit
            if exit_sig:
                should_exit = True
                exit_reason = "signal"
            
            # 2. Stop loss hit
            if price <= current_stop:
                should_exit = True
                exit_reason = "stop"
            
            # 3. Time exit
            if params.max_hold_bars and bars_in_position >= params.max_hold_bars:
                should_exit = True
                exit_reason = "time"
            
            if should_exit:
                fill = apply_cost(price, "SELL")
                notional = btc * fill
                fee = fee_amount(notional)
                
                exit_notional_net = notional - fee
                entry_notional_net = float(entry_notional) if entry_notional else 0
                pnl_usd = exit_notional_net - entry_notional_net
                pnl_pct = (pnl_usd / max(entry_notional_net, 1e-12)) if entry_notional_net > 0 else 0
                
                # Track if this was a loss for cooldown
                if pnl_usd < 0:
                    bars_since_last_loss = 0
                
                trades.append({
                    "entry_ts": entry_ts,
                    "exit_ts": ts,
                    "entry_px": entry_px,
                    "exit_px": fill,
                    "btc_qty": entry_qty,
                    "entry_notional_net": entry_notional_net,
                    "exit_notional_net": exit_notional_net,
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
                entry_notional = None
                entry_qty = None
                highest_since_entry = 0.0
                current_stop = 0.0
                bars_in_position = 0
        
        # Not in position - check entry
        if (not in_pos) and enter and trading_enabled and bars_since_last_loss >= params.cooldown_bars_after_loss:
            # Risk-based position sizing
            stop_price = float(sig.loc[i, "stop_price"]) if not pd.isna(sig.loc[i, "stop_price"]) else 0
            
            if stop_price > 0 and stop_price < price:
                risk_per_unit = price - stop_price
                risk_pct_of_price = risk_per_unit / price
                
                # How much can we risk? risk_per_trade_pct of equity
                risk_budget = equity * (params.risk_per_trade_pct / 100.0)
                
                # Position size: risk_budget / risk_per_unit
                target_qty = risk_budget / risk_per_unit
                
                # Convert to USD and cap at available cash
                target_usd = target_qty * price
                target_usd = min(target_usd, cash * 0.95)  # Keep 5% buffer
                
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
                        entry_notional = notional + fee
                        entry_qty = qty
                        highest_since_entry = price
                        current_stop = stop_price
                        bars_in_position = 0
    
    # Force close any open position at end
    if in_pos:
        price = float(df["Close"].iloc[-1])
        ts = df["Timestamp"].iloc[-1]
        fill = apply_cost(price, "SELL")
        notional = btc * fill
        fee = fee_amount(notional)
        
        exit_notional_net = notional - fee
        entry_notional_net = float(entry_notional) if entry_notional else 0
        pnl_usd = exit_notional_net - entry_notional_net
        pnl_pct = (pnl_usd / max(entry_notional_net, 1e-12)) if entry_notional_net > 0 else 0
        
        trades.append({
            "entry_ts": entry_ts,
            "exit_ts": ts,
            "entry_px": entry_px,
            "exit_px": fill,
            "btc_qty": entry_qty,
            "entry_notional_net": entry_notional_net,
            "exit_notional_net": exit_notional_net,
            "pnl_usd": pnl_usd,
            "pnl_pct": pnl_pct,
            "exit_reason": "end",
            "bars_held": bars_in_position,
        })
        
        cash = cash + notional - fee
        btc = 0.0
    
    trades_df = pd.DataFrame(trades)
    equity_df = pd.DataFrame(equity_rows)
    
    # Calculate summary stats
    summary = _compute_guardian_summary(trades_df, equity_df, initial_cash)
    
    return trades_df, equity_df, summary


def _compute_guardian_summary(trades: pd.DataFrame, equity_curve: pd.DataFrame, initial_cash: float) -> dict:
    """Compute professional-grade performance metrics."""
    eq = equity_curve["equity"].values
    
    # Returns
    total_return = (eq[-1] / eq[0]) - 1.0 if len(eq) > 1 else 0.0
    
    # Drawdown analysis
    peak = np.maximum.accumulate(eq)
    dd = (eq / np.maximum(peak, 1e-12)) - 1.0
    max_dd = float(dd.min()) if len(dd) else 0.0
    
    # Time underwater
    underwater = dd < 0
    underwater_periods = []
    current_underwater = 0
    for u in underwater:
        if u:
            current_underwater += 1
        elif current_underwater > 0:
            underwater_periods.append(current_underwater)
            current_underwater = 0
    if current_underwater > 0:
        underwater_periods.append(current_underwater)
    
    max_underwater_bars = max(underwater_periods) if underwater_periods else 0
    avg_underwater_bars = np.mean(underwater_periods) if underwater_periods else 0
    
    # Per-bar returns for Sharpe/Sortino
    rets = np.diff(eq) / np.maximum(eq[:-1], 1e-12) if len(eq) > 1 else np.array([0.0])
    
    # Annualized Sharpe (assuming hourly bars, ~8760 hours/year)
    if len(rets) > 1 and rets.std() > 0:
        sharpe_hourly = rets.mean() / rets.std()
        sharpe_annual = sharpe_hourly * np.sqrt(8760)
    else:
        sharpe_annual = 0.0
    
    # Sortino (downside deviation)
    downside = rets[rets < 0]
    if len(downside) > 1:
        downside_std = downside.std()
        sortino_hourly = rets.mean() / (downside_std + 1e-12)
        sortino_annual = sortino_hourly * np.sqrt(8760)
    else:
        sortino_annual = 0.0
    
    # Calmar ratio (annual return / max DD)
    years = len(eq) / 8760
    annual_return = ((1 + total_return) ** (1 / max(years, 0.01))) - 1 if years > 0 else 0
    calmar = abs(annual_return / (max_dd + 1e-12)) if max_dd != 0 else 0
    
    # Trade stats
    if trades.empty:
        win_rate = 0.0
        avg_trade = 0.0
        avg_winner = 0.0
        avg_loser = 0.0
        profit_factor = 0.0
        n_trades = 0
        best = 0.0
        worst = 0.0
        avg_bars_held = 0.0
        exit_reasons = {}
    else:
        pnl_pct = trades["pnl_pct"].values
        n_trades = len(trades)
        
        winners = pnl_pct[pnl_pct > 0]
        losers = pnl_pct[pnl_pct < 0]
        
        win_rate = (len(winners) / n_trades) * 100 if n_trades > 0 else 0
        avg_trade = float(pnl_pct.mean() * 100)
        avg_winner = float(winners.mean() * 100) if len(winners) > 0 else 0
        avg_loser = float(losers.mean() * 100) if len(losers) > 0 else 0
        
        gross_profit = winners.sum() if len(winners) > 0 else 0
        gross_loss = abs(losers.sum()) if len(losers) > 0 else 1e-10
        profit_factor = gross_profit / gross_loss
        
        best = float(pnl_pct.max() * 100)
        worst = float(pnl_pct.min() * 100)
        
        avg_bars_held = float(trades["bars_held"].mean()) if "bars_held" in trades.columns else 0
        
        if "exit_reason" in trades.columns:
            exit_reasons = trades["exit_reason"].value_counts().to_dict()
        else:
            exit_reasons = {}
    
    return {
        # Returns
        "total_return_pct": float(total_return * 100),
        "annual_return_pct": float(annual_return * 100),
        "final_equity": float(eq[-1]) if len(eq) else initial_cash,
        
        # Risk metrics
        "max_drawdown_pct": float(max_dd * 100),
        "max_underwater_bars": int(max_underwater_bars),
        "avg_underwater_bars": float(avg_underwater_bars),
        
        # Risk-adjusted returns
        "sharpe_annual": float(sharpe_annual),
        "sortino_annual": float(sortino_annual),
        "calmar_ratio": float(calmar),
        
        # Trade stats
        "n_trades": n_trades,
        "win_rate_pct": float(win_rate),
        "avg_trade_pct": float(avg_trade),
        "avg_winner_pct": float(avg_winner),
        "avg_loser_pct": float(avg_loser),
        "profit_factor": float(profit_factor),
        "best_trade_pct": float(best),
        "worst_trade_pct": float(worst),
        "avg_bars_held": float(avg_bars_held),
        "exit_reasons": exit_reasons,
        
        # Context
        "bars_tested": len(equity_curve),
        "years_tested": float(years),
    }

