"""
Guardian Strategy Backtest Runner

Tests the Guardian capital-preservation strategy on historical BTC data.
"""
from __future__ import annotations

from pathlib import Path
import sys

# Add research to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from engine.backtest_core import load_flight_recorder, buy_and_hold_baseline, BacktestConfig
from strategies.guardian import GuardianParams, build_signals_guardian, run_guardian_backtest


def main():
    print("=" * 60)
    print("GUARDIAN STRATEGY BACKTEST")
    print("Capital Preservation + Active Trading")
    print("=" * 60)
    
    # Load data
    data_path = Path("research/backtests/data/flight_recorder.csv")
    if not data_path.exists():
        # Try alternative path
        data_path = Path("data/btcusd_3600s_2019-01-01_to_2025-12-30.csv")
    
    print(f"\nLoading data from: {data_path}")
    df = load_flight_recorder(str(data_path))
    print(f"Loaded {len(df):,} bars from {df['Timestamp'].iloc[0]} to {df['Timestamp'].iloc[-1]}")
    
    # Strategy parameters - FINAL TUNED VERSION
    # Philosophy: Trade with conviction, let winners run, cut losers
    params = GuardianParams(
        # Trend filters - standard multi-timeframe
        sma_fast=20,
        sma_slow=50,
        sma_regime=200,
        
        # Entry conditions - balanced
        rsi_period=14,
        rsi_entry_min=25,    # Allow buying dips but not capitulation
        rsi_entry_max=75,    # Allow momentum, avoid extreme overbought
        
        # Volatility - trade when there's movement
        atr_period=14,
        atr_filter_lookback=50,
        min_atr_ratio=0.5,   # Trade in calmer markets (more opportunities)
        max_atr_ratio=4.0,   # Can handle more volatility
        
        # RISK MANAGEMENT - KEY TUNING
        risk_per_trade_pct=2.5,   # Risk 2.5% per trade (higher conviction)
        atr_stop_mult=3.5,        # Wide initial stop = 3.5x ATR
        atr_trail_mult=3.0,       # Trail loosely = 3x ATR (let it breathe)
        
        # Circuit breakers - reasonable limits
        max_drawdown_pct=25.0,    # Stop at 25% DD
        cooldown_bars_after_loss=2,  # Quick recovery, don't sit out too long
        
        # Time exit - let winners run
        max_hold_bars=336,        # Max 2 weeks (336 hours)
    )
    
    print("\n--- Parameters ---")
    print(f"  Trend: SMA({params.sma_fast}/{params.sma_slow}/{params.sma_regime})")
    print(f"  RSI Filter: {params.rsi_entry_min}-{params.rsi_entry_max}")
    print(f"  Risk per trade: {params.risk_per_trade_pct}%")
    print(f"  Stop: {params.atr_stop_mult}x ATR, Trail: {params.atr_trail_mult}x ATR")
    print(f"  Max Drawdown Circuit Breaker: {params.max_drawdown_pct}%")
    print(f"  Loss Cooldown: {params.cooldown_bars_after_loss} bars")
    print(f"  Max Hold: {params.max_hold_bars} bars")
    
    # Build signals
    print("\nBuilding signals...")
    sig = build_signals_guardian(df, params)
    
    # Run backtest with realistic costs
    # Coinbase/Kraken fees: ~0.25-0.60%, we use 0.10% total (fee + slippage)
    fee_bps = 6.0       # 0.06% fee
    slippage_bps = 4.0  # 0.04% slippage
    initial_cash = 10000.0  # Start with $10k for realistic trade sizing
    
    print(f"\n--- Backtest Settings ---")
    print(f"  Initial Capital: ${initial_cash:,.2f}")
    print(f"  Fee: {fee_bps} bps ({fee_bps/100:.2f}%)")
    print(f"  Slippage: {slippage_bps} bps ({slippage_bps/100:.2f}%)")
    
    print("\nRunning Guardian backtest...")
    trades_df, equity_df, summary = run_guardian_backtest(
        df, sig, params,
        initial_cash=initial_cash,
        fee_bps=fee_bps,
        slippage_bps=slippage_bps,
    )
    
    # Buy & Hold comparison
    bh_cfg = BacktestConfig(
        fee_bps=fee_bps,
        slippage_bps=slippage_bps,
        initial_cash=initial_cash,
    )
    bh = buy_and_hold_baseline(df, bh_cfg)
    
    # Print results
    print("\n" + "=" * 60)
    print("GUARDIAN STRATEGY RESULTS")
    print("=" * 60)
    
    print("\n--- Performance ---")
    print(f"  {'Total Return:':<25} {summary['total_return_pct']:>10.2f}%")
    print(f"  {'Annual Return:':<25} {summary['annual_return_pct']:>10.2f}%")
    print(f"  {'Final Equity:':<25} ${summary['final_equity']:>10,.2f}")
    
    print("\n--- Risk Metrics (THE IMPORTANT STUFF) ---")
    print(f"  {'Max Drawdown:':<25} {summary['max_drawdown_pct']:>10.2f}%")
    print(f"  {'Max Time Underwater:':<25} {summary['max_underwater_bars']:>10,} bars ({summary['max_underwater_bars']/24:.0f} days)")
    print(f"  {'Sharpe Ratio (Annual):':<25} {summary['sharpe_annual']:>10.2f}")
    print(f"  {'Sortino Ratio (Annual):':<25} {summary['sortino_annual']:>10.2f}")
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
    print(f"  {'B&H Final Equity:':<25} ${bh['final_equity']:>10,.2f}")
    
    # Key comparison metrics
    print("\n--- Strategy vs Buy & Hold ---")
    return_ratio = summary['total_return_pct'] / bh['total_return_pct'] if bh['total_return_pct'] != 0 else 0
    dd_ratio = abs(summary['max_drawdown_pct'] / bh['max_drawdown_pct']) if bh['max_drawdown_pct'] != 0 else 0
    print(f"  {'Return Capture:':<25} {return_ratio*100:>10.1f}% of B&H")
    print(f"  {'Drawdown Reduction:':<25} {(1-dd_ratio)*100:>10.1f}% less DD")
    print(f"  {'Risk-Adjusted Edge:':<25} {'YES' if summary['sharpe_annual'] > 0.5 and summary['max_drawdown_pct'] > -25 else 'NEEDS WORK'}")
    
    # Save results
    out_dir = Path("research/backtests/results")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    trades_df.to_csv(out_dir / "guardian_trades.csv", index=False)
    equity_df.to_csv(out_dir / "guardian_equity.csv", index=False)
    
    print(f"\n--- Output Files ---")
    print(f"  Trades: {out_dir / 'guardian_trades.csv'}")
    print(f"  Equity: {out_dir / 'guardian_equity.csv'}")
    
    # Quality check
    print("\n" + "=" * 60)
    print("STRATEGY QUALITY ASSESSMENT")
    print("=" * 60)
    
    issues = []
    positives = []
    
    if summary['max_drawdown_pct'] > -25:
        positives.append(f"[OK] Drawdown under control ({summary['max_drawdown_pct']:.1f}%)")
    else:
        issues.append(f"[!!] Drawdown too high ({summary['max_drawdown_pct']:.1f}%)")
    
    if summary['sharpe_annual'] > 0.5:
        positives.append(f"[OK] Decent Sharpe ratio ({summary['sharpe_annual']:.2f})")
    else:
        issues.append(f"[!!] Low Sharpe ratio ({summary['sharpe_annual']:.2f})")
    
    if summary['win_rate_pct'] > 35:
        positives.append(f"[OK] Acceptable win rate ({summary['win_rate_pct']:.1f}%)")
    else:
        issues.append(f"[!!] Low win rate ({summary['win_rate_pct']:.1f}%)")
    
    if summary['profit_factor'] > 1.1:
        positives.append(f"[OK] Profitable edge ({summary['profit_factor']:.2f}x profit factor)")
    else:
        issues.append(f"[!!] Weak edge ({summary['profit_factor']:.2f}x profit factor)")
    
    if summary['n_trades'] > 30:
        positives.append(f"[OK] Sufficient trades for statistics ({summary['n_trades']})")
    else:
        issues.append(f"[!!] Few trades, results may be unreliable ({summary['n_trades']})")
    
    if summary['total_return_pct'] > 0:
        positives.append(f"[OK] Makes money ({summary['total_return_pct']:.1f}%)")
    else:
        issues.append(f"[!!] Loses money ({summary['total_return_pct']:.1f}%)")
    
    for p in positives:
        print(f"  {p}")
    for i in issues:
        print(f"  {i}")
    
    if len(issues) == 0:
        print("\n  >>> Strategy passes all quality checks!")
    elif len(issues) <= 2:
        print("\n  >>> Strategy has minor issues, consider parameter tuning")
    else:
        print("\n  >>> Strategy needs significant work")
    
    print("\n" + "=" * 60)
    
    return summary, trades_df, equity_df


if __name__ == "__main__":
    main()

