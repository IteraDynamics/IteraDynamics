#!/usr/bin/env python3
"""
portfolio_backtest.py

Backtest a portfolio combining Core + Sleeve strategies with static weights.

Usage:
    python portfolio_backtest.py \
        --data data/btcusd_3600s_2023-01-01_to_2025-12-30.csv \
        --core_strategy research.strategies.sg_core_exposure_v2 \
        --core_func generate_intent \
        --sleeve_strategy research.strategies.sg_mean_reversion_a \
        --sleeve_func generate_intent \
        --w_core 0.7 \
        --w_sleeve 0.3 \
        --start 2023-01-01 \
        --end 2025-12-30 \
        --lookback 200 \
        --fees 0.0006 \
        --slippage 0.0005
"""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import numpy as np
import pandas as pd

# Add runtime/argus to path
sys.path.insert(0, str(Path(__file__).parent / 'runtime' / 'argus'))


def load_strategy(module_name: str, func_name: str) -> Callable:
    """Dynamically import a strategy function"""
    mod = importlib.import_module(module_name)
    return getattr(mod, func_name)


def simple_backtest(
    strategy_func: Callable,
    df: pd.DataFrame,
    name: str,
    fees: float = 0.0006,
    slippage: float = 0.0005
) -> pd.DataFrame:
    """
    Run a simple backtest and return equity curve DataFrame
    
    Returns DataFrame with columns: timestamp, equity, position, price
    """
    
    equity = 1.0
    position = 0.0
    entry_price = 0.0
    
    records = []
    
    for i in range(100, len(df)):
        window = df.iloc[:i]
        current_price = float(df.iloc[i]['Close'])
        timestamp = df.index[i]
        
        try:
            intent = strategy_func(window, {}, closed_only=True)
            action = intent.get('action', 'HOLD')
            
            if action == "ENTER_LONG" and position == 0:
                # Enter with slippage and fees
                fill_price = current_price * (1 + slippage)
                position = intent.get('desired_exposure_frac', 0.5)
                entry_price = fill_price
                equity *= (1 - fees)  # Pay entry fee
                
            elif action == "EXIT_LONG" and position > 0:
                # Exit with slippage and fees
                fill_price = current_price * (1 - slippage)
                pnl = (fill_price / entry_price - 1) * position
                equity *= (1 + pnl) * (1 - fees)  # Apply PnL and exit fee
                position = 0
            
            # Calculate mark-to-market equity
            if position > 0:
                mtm_equity = equity * (1 + (current_price / entry_price - 1) * position)
            else:
                mtm_equity = equity
            
        except Exception as e:
            print(f"Error at bar {i} for {name}: {e}")
            mtm_equity = equity
        
        records.append({
            'timestamp': timestamp,
            'equity': mtm_equity,
            'position': position,
            'price': current_price
        })
    
    return pd.DataFrame(records).set_index('timestamp')


def calculate_metrics(equity_curve: pd.Series, name: str) -> Dict[str, float]:
    """Calculate performance metrics from equity curve"""
    
    total_return = (equity_curve.iloc[-1] - 1) * 100
    
    # Calculate annualized metrics
    years = len(equity_curve) / (365.25 * 24)  # Assuming hourly data
    cagr = (equity_curve.iloc[-1] ** (1/years) - 1) * 100
    
    # Drawdown
    running_max = equity_curve.expanding().max()
    drawdown = ((equity_curve - running_max) / running_max * 100)
    max_dd = drawdown.min()
    
    # Calmar ratio
    calmar = cagr / abs(max_dd) if max_dd != 0 else 0
    
    # Sortino ratio
    returns = equity_curve.pct_change().dropna()
    downside_returns = returns[returns < 0]
    downside_std = downside_returns.std() * np.sqrt(365.25 * 24)  # Annualized
    sortino = (cagr / 100) / downside_std if downside_std > 0 else 0
    
    # Sharpe ratio
    returns_std = returns.std() * np.sqrt(365.25 * 24)  # Annualized
    sharpe = (cagr / 100) / returns_std if returns_std > 0 else 0
    
    return {
        'name': name,
        'total_return_pct': total_return,
        'cagr_pct': cagr,
        'max_drawdown_pct': max_dd,
        'calmar': calmar,
        'sortino': sortino,
        'sharpe': sharpe,
        'final_equity': equity_curve.iloc[-1]
    }


def print_metrics(metrics: Dict[str, float]) -> None:
    """Pretty print metrics"""
    print(f"\n{'='*60}")
    print(f"{metrics['name']}")
    print(f"{'='*60}")
    print(f"Total Return:    {metrics['total_return_pct']:>10.2f}%")
    print(f"CAGR:            {metrics['cagr_pct']:>10.2f}%")
    print(f"Max Drawdown:    {metrics['max_drawdown_pct']:>10.2f}%")
    print(f"Calmar Ratio:    {metrics['calmar']:>10.2f}")
    print(f"Sortino Ratio:   {metrics['sortino']:>10.2f}")
    print(f"Sharpe Ratio:    {metrics['sharpe']:>10.2f}")
    print(f"Final Equity:    {metrics['final_equity']:>10.4f}")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description='Portfolio backtest: Core + Sleeve')
    
    # Data
    parser.add_argument('--data', required=True, help='Path to OHLCV CSV')
    
    # Core strategy
    parser.add_argument('--core_strategy', required=True, help='Core strategy module path')
    parser.add_argument('--core_func', default='generate_intent', help='Core strategy function')
    
    # Sleeve strategy
    parser.add_argument('--sleeve_strategy', required=True, help='Sleeve strategy module path')
    parser.add_argument('--sleeve_func', default='generate_intent', help='Sleeve strategy function')
    
    # Weights
    parser.add_argument('--w_core', type=float, default=0.7, help='Core weight (0-1)')
    parser.add_argument('--w_sleeve', type=float, default=0.3, help='Sleeve weight (0-1)')
    
    # Date range
    parser.add_argument('--start', help='Start date (YYYY-MM-DD), optional')
    parser.add_argument('--end', help='End date (YYYY-MM-DD), optional')
    
    # Backtest params
    parser.add_argument('--lookback', type=int, default=200, help='Lookback bars before start')
    parser.add_argument('--fees', type=float, default=0.0006, help='Fee per trade (0.0006 = 6bps)')
    parser.add_argument('--slippage', type=float, default=0.0005, help='Slippage per trade (0.0005 = 5bps)')
    
    args = parser.parse_args()
    
    # Validate weights
    if args.w_core + args.w_sleeve != 1.0:
        print(f"Warning: Weights sum to {args.w_core + args.w_sleeve}, normalizing...")
        total = args.w_core + args.w_sleeve
        args.w_core /= total
        args.w_sleeve /= total
        print(f"Normalized: w_core={args.w_core:.3f}, w_sleeve={args.w_sleeve:.3f}")
    
    # Load data
    print(f"\nLoading data from: {args.data}")
    df = pd.read_csv(args.data)
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df = df.set_index('Timestamp')
    
    # Filter date range if specified
    if args.start:
        df = df[df.index >= args.start]
    if args.end:
        df = df[df.index <= args.end]
    
    print(f"Data range: {df.index[0]} to {df.index[-1]}")
    print(f"Total bars: {len(df)}")
    
    # Load strategies
    print(f"\nLoading strategies...")
    core_func = load_strategy(args.core_strategy, args.core_func)
    print(f"  Core: {args.core_strategy}.{args.core_func}")
    
    sleeve_func = load_strategy(args.sleeve_strategy, args.sleeve_func)
    print(f"  Sleeve: {args.sleeve_strategy}.{args.sleeve_func}")
    
    # Run backtests
    print(f"\nRunning backtests (fees={args.fees:.4f}, slippage={args.slippage:.4f})...")
    
    print("\n[1/2] Backtesting Core...")
    core_df = simple_backtest(core_func, df, "Core", args.fees, args.slippage)
    
    print("[2/2] Backtesting Sleeve...")
    sleeve_df = simple_backtest(sleeve_func, df, "Sleeve", args.fees, args.slippage)
    
    # Align timestamps
    common_idx = core_df.index.intersection(sleeve_df.index)
    core_equity = core_df.loc[common_idx, 'equity']
    sleeve_equity = sleeve_df.loc[common_idx, 'equity']
    
    # Combine with weights
    print(f"\nCombining portfolio (w_core={args.w_core:.3f}, w_sleeve={args.w_sleeve:.3f})...")
    combined_equity = args.w_core * core_equity + args.w_sleeve * sleeve_equity
    
    # Calculate metrics
    core_metrics = calculate_metrics(core_equity, f"Core ({args.w_core*100:.0f}%)")
    sleeve_metrics = calculate_metrics(sleeve_equity, f"Sleeve ({args.w_sleeve*100:.0f}%)")
    combined_metrics = calculate_metrics(combined_equity, "Combined Portfolio")
    
    # Print results
    print_metrics(core_metrics)
    print_metrics(sleeve_metrics)
    print_metrics(combined_metrics)
    
    # Save combined curve
    output_df = pd.DataFrame({
        'timestamp': common_idx,
        'core_equity': core_equity.values,
        'sleeve_equity': sleeve_equity.values,
        'combined_equity': combined_equity.values,
        'price': core_df.loc[common_idx, 'price'].values
    })
    
    output_file = 'combined_curve.csv'
    output_df.to_csv(output_file, index=False)
    print(f"\n✅ Saved combined equity curve to: {output_file}")
    
    # Summary table
    print(f"\n{'='*80}")
    print("SUMMARY COMPARISON")
    print(f"{'='*80}")
    print(f"{'Metric':<20} {'Core':>15} {'Sleeve':>15} {'Combined':>15}")
    print(f"{'-'*80}")
    print(f"{'CAGR %':<20} {core_metrics['cagr_pct']:>14.2f}% {sleeve_metrics['cagr_pct']:>14.2f}% {combined_metrics['cagr_pct']:>14.2f}%")
    print(f"{'Max DD %':<20} {core_metrics['max_drawdown_pct']:>14.2f}% {sleeve_metrics['max_drawdown_pct']:>14.2f}% {combined_metrics['max_drawdown_pct']:>14.2f}%")
    print(f"{'Calmar':<20} {core_metrics['calmar']:>15.2f} {sleeve_metrics['calmar']:>15.2f} {combined_metrics['calmar']:>15.2f}")
    print(f"{'Sortino':<20} {core_metrics['sortino']:>15.2f} {sleeve_metrics['sortino']:>15.2f} {combined_metrics['sortino']:>15.2f}")
    print(f"{'Sharpe':<20} {core_metrics['sharpe']:>15.2f} {sleeve_metrics['sharpe']:>15.2f} {combined_metrics['sharpe']:>15.2f}")
    print(f"{'Final Equity':<20} {core_metrics['final_equity']:>15.4f} {sleeve_metrics['final_equity']:>15.4f} {combined_metrics['final_equity']:>15.4f}")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()
