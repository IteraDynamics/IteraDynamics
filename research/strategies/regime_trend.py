"""
Regime Trend Strategy - Simple but Robust

Philosophy:
- BTC trends hard in both directions
- Don't try to time entries perfectly - just be in the right regime
- Stay long during bull markets (above 200 SMA with confirmation)
- Go flat during bear markets
- Minimal trading = minimal costs = more robust returns

This is the "simple but effective" approach that actually works.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple
import pandas as pd
import numpy as np


@dataclass  
class RegimeTrendParams:
    # Regime detection (using longer periods reduces noise)
    regime_sma: int = 200           # Primary regime filter
    confirm_sma: int = 50           # Confirmation filter
    
    # Hysteresis to reduce whipsaws
    # Only enter when price is X% above the SMA
    # Only exit when price is X% below the SMA  
    entry_buffer_pct: float = 2.0   # Enter when 2% above SMA
    exit_buffer_pct: float = 2.0    # Exit when 2% below SMA
    
    # Additional confirmation
    sma_slope_bars: int = 10        # Check if SMA is rising over N bars
    
    # Risk management
    position_size_pct: float = 90.0  # Deploy 90% of capital (keep 10% buffer)
    
    # Trailing stop to lock in profits
    use_trailing_stop: bool = True
    trailing_stop_pct: float = 15.0  # Trail 15% from peak
    
    # Circuit breaker (emergency stop)
    max_loss_pct: float = 20.0       # Hard stop if position down > 20%


def _sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n, min_periods=n).mean()


def build_regime_signals(df: pd.DataFrame, params: RegimeTrendParams) -> pd.DataFrame:
    """
    Build simple regime-based signals.
    
    Bull regime = price above 200 SMA (with buffer) AND 50 SMA above 200 SMA
    Bear regime = anything else
    """
    df = df.copy()
    close = df["Close"].astype(float)
    
    sma_regime = _sma(close, params.regime_sma)
    sma_confirm = _sma(close, params.confirm_sma)
    
    # Entry threshold: must be above 200 SMA (with buffer)
    entry_threshold = sma_regime * (1 + params.entry_buffer_pct / 100)
    
    # Exit threshold: use 200 SMA (stable)
    exit_threshold = sma_regime * (1 - params.exit_buffer_pct / 100)
    
    # SMA slope check (is the trend actually up?)
    sma_rising = sma_regime > sma_regime.shift(params.sma_slope_bars)
    
    # Confirmation: 50 SMA above 200 SMA
    sma_confirmed = sma_confirm > sma_regime
    
    # Entry: price above entry threshold + confirmation + SMA rising
    can_enter = (close > entry_threshold) & sma_confirmed & sma_rising
    
    # Exit: price below exit threshold OR confirmation fails
    should_exit = (close < exit_threshold) | (~sma_confirmed)
    
    sig = pd.DataFrame({
        "can_enter": can_enter.fillna(False).astype(bool),
        "should_exit": should_exit.fillna(True).astype(bool),  # Default to exit if uncertain
        "sma_regime": sma_regime,
        "sma_confirm": sma_confirm,
    })
    
    return sig


def run_regime_backtest(
    df: pd.DataFrame,
    signals: pd.DataFrame,
    params: RegimeTrendParams,
    initial_cash: float = 10000.0,
    fee_bps: float = 6.0,
    slippage_bps: float = 10.0,
) -> Tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Backtest the regime trend strategy.
    Very simple: be long during bull regime, flat otherwise.
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
    peak_price = 0.0  # For trailing stop
    
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
        
        equity = cash + btc * price
        equity_rows.append({
            "Timestamp": ts,
            "equity": equity,
            "cash": cash,
            "btc": btc,
            "price": price,
            "in_position": in_pos,
        })
        
        can_enter = bool(sig.loc[i, "can_enter"])
        should_exit = bool(sig.loc[i, "should_exit"])
        
        # Exit logic
        if in_pos:
            # Update peak price for trailing stop
            if price > peak_price:
                peak_price = price
            
            exit_now = False
            exit_reason = ""
            
            # Regime says exit
            if should_exit:
                exit_now = True
                exit_reason = "regime"
            
            # Trailing stop: price dropped X% from peak
            if params.use_trailing_stop and peak_price > 0:
                trail_stop = peak_price * (1 - params.trailing_stop_pct / 100)
                if price < trail_stop:
                    exit_now = True
                    exit_reason = "trail"
            
            # Hard stop: position down too much from entry
            if entry_px and price < entry_px * (1 - params.max_loss_pct / 100):
                exit_now = True
                exit_reason = "stop"
            
            if exit_now:
                fill = apply_cost(price, "SELL")
                notional = btc * fill
                fee = fee_amount(notional)
                
                exit_net = notional - fee
                pnl_usd = exit_net - entry_cost if entry_cost else 0
                pnl_pct = pnl_usd / entry_cost if entry_cost and entry_cost > 0 else 0
                
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
                })
                
                cash = cash + notional - fee
                btc = 0.0
                in_pos = False
                entry_px = None
                entry_ts = None
                entry_cost = None
                entry_qty = None
        
        # Entry logic
        if (not in_pos) and can_enter:
            # Deploy configured % of capital
            target_usd = equity * (params.position_size_pct / 100)
            target_usd = min(target_usd, cash * 0.98)  # Keep small buffer
            
            if target_usd > 10:  # Min $10 trade
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
                    peak_price = price  # Initialize peak for trailing stop
    
    # Close any open position
    if in_pos:
        price = float(df["Close"].iloc[-1])
        ts = df["Timestamp"].iloc[-1]
        fill = apply_cost(price, "SELL")
        notional = btc * fill
        fee = fee_amount(notional)
        
        exit_net = notional - fee
        pnl_usd = exit_net - entry_cost if entry_cost else 0
        pnl_pct = pnl_usd / entry_cost if entry_cost and entry_cost > 0 else 0
        
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
    
    # Time underwater analysis
    underwater = dd < 0
    max_underwater = 0
    current = 0
    for u in underwater:
        if u:
            current += 1
            max_underwater = max(max_underwater, current)
        else:
            current = 0
    
    # Returns
    rets = np.diff(eq) / np.maximum(eq[:-1], 1e-12) if len(eq) > 1 else np.array([0.0])
    
    years = len(eq) / 8760
    annual_return = ((1 + total_return) ** (1 / max(years, 0.01))) - 1 if years > 0 else 0
    
    # Risk-adjusted metrics
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
    
    # Time in market
    in_market = equity_curve["in_position"].sum() / len(equity_curve) * 100 if len(equity_curve) > 0 else 0
    
    # Trade stats
    if trades.empty:
        trade_stats = {
            "n_trades": 0, "win_rate_pct": 0, "avg_trade_pct": 0,
            "avg_winner_pct": 0, "avg_loser_pct": 0, "profit_factor": 0,
            "best_trade_pct": 0, "worst_trade_pct": 0,
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
            "win_rate_pct": (len(winners) / len(trades)) * 100 if len(trades) > 0 else 0,
            "avg_trade_pct": float(pnl.mean() * 100),
            "avg_winner_pct": float(winners.mean() * 100) if len(winners) else 0,
            "avg_loser_pct": float(losers.mean() * 100) if len(losers) else 0,
            "profit_factor": gross_profit / gross_loss,
            "best_trade_pct": float(pnl.max() * 100),
            "worst_trade_pct": float(pnl.min() * 100),
            "exit_reasons": trades["exit_reason"].value_counts().to_dict() if "exit_reason" in trades.columns else {},
        }
    
    return {
        "total_return_pct": float(total_return * 100),
        "annual_return_pct": float(annual_return * 100),
        "final_equity": float(eq[-1]),
        "max_drawdown_pct": float(max_dd * 100),
        "max_underwater_bars": int(max_underwater),
        "time_in_market_pct": float(in_market),
        "sharpe_annual": float(sharpe_annual),
        "sortino_annual": float(sortino_annual),
        "calmar_ratio": float(calmar),
        "years_tested": float(years),
        **trade_stats,
    }

