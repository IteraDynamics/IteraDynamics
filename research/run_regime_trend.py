"""
Regime Trend Strategy Runner

Simple regime-based trend following for BTC.
Be long in bull markets, flat in bear markets.
"""
from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

from engine.backtest_core import load_flight_recorder, buy_and_hold_baseline, BacktestConfig
from strategies.regime_trend import RegimeTrendParams, build_regime_signals, run_regime_backtest


def main():
    print("=" * 60)
    print("REGIME TREND STRATEGY")
    print("Simple but Robust - Be Long in Bull Markets")
    print("=" * 60)
    
    # Load data
    data_path = Path("research/backtests/data/flight_recorder.csv")
    if not data_path.exists():
        data_path = Path("data/btcusd_3600s_2019-01-01_to_2025-12-30.csv")
    
    print(f"\nLoading data from: {data_path}")
    df = load_flight_recorder(str(data_path))
    print(f"Loaded {len(df):,} bars from {df['Timestamp'].iloc[0]} to {df['Timestamp'].iloc[-1]}")
    
    # Parameters - BEST RISK-ADJUSTED VERSION
    # Achieved: 470% return, -38% DD, Sharpe 1.02, PF 1.84
    params = RegimeTrendParams(
        regime_sma=200,
        confirm_sma=50,
        entry_buffer_pct=3.0,       # Enter 3% above 200 SMA
        exit_buffer_pct=0.0,        # Exit at 200 SMA
        sma_slope_bars=5,           # Trend direction check
        position_size_pct=75.0,     # Deploy 75% of capital
        use_trailing_stop=True,     # KEY: Lock in profits
        trailing_stop_pct=12.0,     # Trail 12% from peak
        max_loss_pct=10.0,          # Hard stop at -10%
    )
    
    print("\n--- Parameters ---")
    print(f"  Regime SMA: {params.regime_sma}")
    print(f"  Confirm SMA: {params.confirm_sma}")
    print(f"  Entry Buffer: {params.entry_buffer_pct}% above SMA")
    print(f"  Exit Buffer: {params.exit_buffer_pct}% below SMA")
    print(f"  Position Size: {params.position_size_pct}%")
    print(f"  Trailing Stop: {params.trailing_stop_pct}%")
    print(f"  Hard Stop: {params.max_loss_pct}%")
    
    # Build signals
    print("\nBuilding signals...")
    sig = build_regime_signals(df, params)
    
    # Run backtest
    fee_bps = 6.0
    slippage_bps = 4.0
    initial_cash = 10000.0
    
    print(f"\n--- Backtest Settings ---")
    print(f"  Initial Capital: ${initial_cash:,.2f}")
    print(f"  Fee: {fee_bps} bps, Slippage: {slippage_bps} bps")
    
    print("\nRunning backtest...")
    trades_df, equity_df, summary = run_regime_backtest(
        df, sig, params,
        initial_cash=initial_cash,
        fee_bps=fee_bps,
        slippage_bps=slippage_bps,
    )
    
    # B&H comparison
    bh_cfg = BacktestConfig(fee_bps=fee_bps, slippage_bps=slippage_bps, initial_cash=initial_cash)
    bh = buy_and_hold_baseline(df, bh_cfg)
    
    # Results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    
    print("\n--- Performance ---")
    print(f"  {'Total Return:':<25} {summary['total_return_pct']:>10.2f}%")
    print(f"  {'Annual Return:':<25} {summary['annual_return_pct']:>10.2f}%")
    print(f"  {'Final Equity:':<25} ${summary['final_equity']:>10,.2f}")
    
    print("\n--- Risk ---")
    print(f"  {'Max Drawdown:':<25} {summary['max_drawdown_pct']:>10.2f}%")
    print(f"  {'Max Underwater:':<25} {summary['max_underwater_bars']:>10,} bars ({summary['max_underwater_bars']/24:.0f} days)")
    print(f"  {'Sharpe (Annual):':<25} {summary['sharpe_annual']:>10.2f}")
    print(f"  {'Sortino (Annual):':<25} {summary['sortino_annual']:>10.2f}")
    print(f"  {'Calmar Ratio:':<25} {summary['calmar_ratio']:>10.2f}")
    
    print("\n--- Activity ---")
    print(f"  {'Time in Market:':<25} {summary['time_in_market_pct']:>10.1f}%")
    print(f"  {'Number of Trades:':<25} {summary['n_trades']:>10}")
    print(f"  {'Win Rate:':<25} {summary['win_rate_pct']:>10.1f}%")
    print(f"  {'Profit Factor:':<25} {summary['profit_factor']:>10.2f}")
    
    if summary.get('exit_reasons'):
        print("\n--- Exit Reasons ---")
        for reason, count in summary['exit_reasons'].items():
            pct = (count / summary['n_trades']) * 100 if summary['n_trades'] > 0 else 0
            print(f"  {reason:<15} {count:>3} ({pct:>5.1f}%)")
    
    print("\n--- Buy & Hold ---")
    print(f"  {'B&H Return:':<25} {bh['total_return_pct']:>10.2f}%")
    print(f"  {'B&H Max DD:':<25} {bh['max_drawdown_pct']:>10.2f}%")
    
    # Comparison
    return_capture = summary['total_return_pct'] / bh['total_return_pct'] * 100 if bh['total_return_pct'] != 0 else 0
    dd_reduction = (1 - abs(summary['max_drawdown_pct'] / bh['max_drawdown_pct'])) * 100 if bh['max_drawdown_pct'] != 0 else 0
    
    print("\n--- Strategy vs B&H ---")
    print(f"  {'Return Capture:':<25} {return_capture:>10.1f}%")
    print(f"  {'DD Reduction:':<25} {dd_reduction:>10.1f}%")
    
    # Quality
    print("\n" + "=" * 60)
    print("QUALITY CHECK")
    print("=" * 60)
    
    passed = 0
    total = 5
    
    if summary['total_return_pct'] > 50:
        print("  [OK] Returns > 50%")
        passed += 1
    else:
        print(f"  [--] Returns only {summary['total_return_pct']:.1f}%")
    
    if summary['max_drawdown_pct'] > -35:
        print("  [OK] Max DD < 35%")
        passed += 1
    else:
        print(f"  [!!] Max DD is {summary['max_drawdown_pct']:.1f}%")
    
    if summary['sharpe_annual'] > 0.5:
        print(f"  [OK] Sharpe > 0.5 ({summary['sharpe_annual']:.2f})")
        passed += 1
    else:
        print(f"  [--] Sharpe only {summary['sharpe_annual']:.2f}")
    
    if summary['profit_factor'] > 1.5:
        print(f"  [OK] Profit Factor > 1.5 ({summary['profit_factor']:.2f})")
        passed += 1
    elif summary['profit_factor'] > 1.2:
        print(f"  [--] Profit Factor is {summary['profit_factor']:.2f}")
    else:
        print(f"  [!!] Profit Factor is {summary['profit_factor']:.2f}")
    
    if return_capture > 30 and dd_reduction > 40:
        print("  [OK] Good risk/reward trade-off")
        passed += 1
    else:
        print("  [--] Could improve risk/reward")
    
    print(f"\n  Score: {passed}/{total}")
    
    # Save
    out_dir = Path("research/backtests/results")
    out_dir.mkdir(parents=True, exist_ok=True)
    trades_df.to_csv(out_dir / "regime_trend_trades.csv", index=False)
    equity_df.to_csv(out_dir / "regime_trend_equity.csv", index=False)
    print(f"\n  Saved to {out_dir}")
    
    print("\n" + "=" * 60)
    
    return summary


if __name__ == "__main__":
    main()

