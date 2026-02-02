"""
Sentinel Strategy Backtest Runner

Momentum breakout strategy optimized for BTC's trending nature.
"""
from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

from engine.backtest_core import load_flight_recorder, buy_and_hold_baseline, BacktestConfig
from strategies.sentinel import SentinelParams, build_signals_sentinel, run_sentinel_backtest


def main():
    print("=" * 60)
    print("SENTINEL STRATEGY BACKTEST")
    print("Momentum Breakout with Capital Protection")
    print("=" * 60)
    
    # Load data
    data_path = Path("research/backtests/data/flight_recorder.csv")
    if not data_path.exists():
        data_path = Path("data/btcusd_3600s_2019-01-01_to_2025-12-30.csv")
    
    print(f"\nLoading data from: {data_path}")
    df = load_flight_recorder(str(data_path))
    print(f"Loaded {len(df):,} bars from {df['Timestamp'].iloc[0]} to {df['Timestamp'].iloc[-1]}")
    
    # Strategy parameters
    params = SentinelParams(
        # Trend regime
        regime_sma=200,
        trend_sma=50,
        
        # Breakout entry
        breakout_period=20,        # 20-bar breakout
        volume_confirm_bars=5,
        
        # ATR settings
        atr_period=14,
        
        # Risk management
        risk_per_trade_pct=2.5,    # Risk 2.5% per trade
        initial_stop_atr=3.5,      # Wide initial stop
        trailing_stop_atr=3.0,     # Trail at 3x ATR
        
        # Exit conditions
        max_hold_bars=504,         # 3 weeks max
        profit_target_pct=None,    # Let winners run
        
        # Circuit breaker
        max_drawdown_pct=25.0,
    )
    
    print("\n--- Parameters ---")
    print(f"  Regime: SMA({params.regime_sma}), Trend: SMA({params.trend_sma})")
    print(f"  Breakout: {params.breakout_period}-bar high")
    print(f"  Risk per trade: {params.risk_per_trade_pct}%")
    print(f"  Stops: Initial {params.initial_stop_atr}x ATR, Trail {params.trailing_stop_atr}x ATR")
    print(f"  Max Hold: {params.max_hold_bars} bars ({params.max_hold_bars/24:.0f} days)")
    print(f"  Circuit Breaker: {params.max_drawdown_pct}% max DD")
    
    # Build signals
    print("\nBuilding signals...")
    sig = build_signals_sentinel(df, params)
    
    # Run backtest
    fee_bps = 6.0
    slippage_bps = 4.0
    initial_cash = 10000.0
    
    print(f"\n--- Backtest Settings ---")
    print(f"  Initial Capital: ${initial_cash:,.2f}")
    print(f"  Fee: {fee_bps} bps, Slippage: {slippage_bps} bps")
    
    print("\nRunning Sentinel backtest...")
    trades_df, equity_df, summary = run_sentinel_backtest(
        df, sig, params,
        initial_cash=initial_cash,
        fee_bps=fee_bps,
        slippage_bps=slippage_bps,
    )
    
    # Buy & Hold comparison
    bh_cfg = BacktestConfig(fee_bps=fee_bps, slippage_bps=slippage_bps, initial_cash=initial_cash)
    bh = buy_and_hold_baseline(df, bh_cfg)
    
    # Print results
    print("\n" + "=" * 60)
    print("SENTINEL STRATEGY RESULTS")
    print("=" * 60)
    
    print("\n--- Performance ---")
    print(f"  {'Total Return:':<25} {summary['total_return_pct']:>10.2f}%")
    print(f"  {'Annual Return:':<25} {summary['annual_return_pct']:>10.2f}%")
    print(f"  {'Final Equity:':<25} ${summary['final_equity']:>10,.2f}")
    
    print("\n--- Risk Metrics ---")
    print(f"  {'Max Drawdown:':<25} {summary['max_drawdown_pct']:>10.2f}%")
    print(f"  {'Max Time Underwater:':<25} {summary['max_underwater_bars']:>10,} bars ({summary['max_underwater_bars']/24:.0f} days)")
    print(f"  {'Sharpe (Annual):':<25} {summary['sharpe_annual']:>10.2f}")
    print(f"  {'Sortino (Annual):':<25} {summary['sortino_annual']:>10.2f}")
    print(f"  {'Calmar Ratio:':<25} {summary['calmar_ratio']:>10.2f}")
    
    print("\n--- Trade Statistics ---")
    print(f"  {'Number of Trades:':<25} {summary['n_trades']:>10}")
    print(f"  {'Win Rate:':<25} {summary['win_rate_pct']:>10.1f}%")
    print(f"  {'Avg Trade:':<25} {summary['avg_trade_pct']:>10.2f}%")
    print(f"  {'Avg Winner:':<25} {summary['avg_winner_pct']:>10.2f}%")
    print(f"  {'Avg Loser:':<25} {summary['avg_loser_pct']:>10.2f}%")
    print(f"  {'Profit Factor:':<25} {summary['profit_factor']:>10.2f}")
    print(f"  {'Best Trade:':<25} {summary['best_trade_pct']:>10.2f}%")
    print(f"  {'Worst Trade:':<25} {summary['worst_trade_pct']:>10.2f}%")
    print(f"  {'Avg Bars Held:':<25} {summary['avg_bars_held']:>10.1f} ({summary['avg_bars_held']/24:.1f} days)")
    
    if summary.get('exit_reasons'):
        print("\n--- Exit Reasons ---")
        for reason, count in summary['exit_reasons'].items():
            pct = (count / summary['n_trades']) * 100 if summary['n_trades'] > 0 else 0
            print(f"  {reason:<25} {count:>5} ({pct:>5.1f}%)")
    
    print("\n--- Buy & Hold Comparison ---")
    print(f"  {'B&H Return:':<25} {bh['total_return_pct']:>10.2f}%")
    print(f"  {'B&H Max Drawdown:':<25} {bh['max_drawdown_pct']:>10.2f}%")
    
    # Risk-adjusted comparison
    return_ratio = summary['total_return_pct'] / bh['total_return_pct'] if bh['total_return_pct'] != 0 else 0
    dd_improvement = 1 - abs(summary['max_drawdown_pct'] / bh['max_drawdown_pct']) if bh['max_drawdown_pct'] != 0 else 0
    
    print("\n--- Strategy vs B&H ---")
    print(f"  {'Return Capture:':<25} {return_ratio*100:>10.1f}% of B&H")
    print(f"  {'Drawdown Reduction:':<25} {dd_improvement*100:>10.1f}% less")
    
    # Quality assessment
    print("\n" + "=" * 60)
    print("QUALITY ASSESSMENT")
    print("=" * 60)
    
    checks = []
    if summary['total_return_pct'] > 0:
        checks.append(f"[OK] Profitable ({summary['total_return_pct']:.1f}%)")
    else:
        checks.append(f"[!!] Not profitable ({summary['total_return_pct']:.1f}%)")
    
    if summary['max_drawdown_pct'] > -30:
        checks.append(f"[OK] Controlled drawdown ({summary['max_drawdown_pct']:.1f}%)")
    else:
        checks.append(f"[!!] High drawdown ({summary['max_drawdown_pct']:.1f}%)")
    
    if summary['sharpe_annual'] > 0.5:
        checks.append(f"[OK] Good Sharpe ({summary['sharpe_annual']:.2f})")
    elif summary['sharpe_annual'] > 0.3:
        checks.append(f"[--] Marginal Sharpe ({summary['sharpe_annual']:.2f})")
    else:
        checks.append(f"[!!] Poor Sharpe ({summary['sharpe_annual']:.2f})")
    
    if summary['profit_factor'] > 1.3:
        checks.append(f"[OK] Strong edge ({summary['profit_factor']:.2f}x)")
    elif summary['profit_factor'] > 1.1:
        checks.append(f"[--] Marginal edge ({summary['profit_factor']:.2f}x)")
    else:
        checks.append(f"[!!] Weak edge ({summary['profit_factor']:.2f}x)")
    
    if summary['n_trades'] >= 30:
        checks.append(f"[OK] Statistical significance ({summary['n_trades']} trades)")
    else:
        checks.append(f"[!!] Low sample size ({summary['n_trades']} trades)")
    
    for c in checks:
        print(f"  {c}")
    
    ok_count = sum(1 for c in checks if c.startswith("[OK]"))
    print(f"\n  Passed {ok_count}/{len(checks)} quality checks")
    
    # Save results
    out_dir = Path("research/backtests/results")
    out_dir.mkdir(parents=True, exist_ok=True)
    trades_df.to_csv(out_dir / "sentinel_trades.csv", index=False)
    equity_df.to_csv(out_dir / "sentinel_equity.csv", index=False)
    print(f"\n  Results saved to {out_dir}")
    
    print("\n" + "=" * 60)
    
    return summary, trades_df, equity_df


if __name__ == "__main__":
    main()


