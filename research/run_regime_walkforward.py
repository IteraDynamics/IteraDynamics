"""
Walk-Forward Validation for Regime Trend Strategy

Proper out-of-sample testing:
- Train on historical data (6 months)
- Test on next period (1 month)
- Roll forward, repeat
- NO parameter tuning on test data

This prevents look-ahead bias and gives realistic performance estimates.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Dict, Tuple
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from engine.backtest_core import load_flight_recorder
from strategies.regime_trend import RegimeTrendParams, build_regime_signals, run_regime_backtest


def walk_forward_folds(
    df: pd.DataFrame,
    train_days: int = 180,
    test_days: int = 30,
    step_days: int = 30,
) -> List[Tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
    """
    Generate walk-forward folds:
    - Train on [train_start, train_end)
    - Test on [test_start, test_end)
    - Step forward by step_days
    """
    t0 = df["Timestamp"].min()
    t1 = df["Timestamp"].max()
    
    train_td = pd.Timedelta(days=train_days)
    test_td = pd.Timedelta(days=test_days)
    step_td = pd.Timedelta(days=step_days)
    
    folds = []
    test_start = t0 + train_td
    
    while test_start + test_td <= t1:
        train_start = test_start - train_td
        train_end = test_start
        test_end = test_start + test_td
        
        folds.append((train_start, train_end, test_start, test_end))
        test_start = test_start + step_td
    
    return folds


def run_fold(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    params: RegimeTrendParams,
    initial_cash: float,
    fee_bps: float,
    slippage_bps: float,
) -> Dict:
    """Run strategy on a single fold (train signals on train, test on test)."""
    
    # Build signals using ONLY training data for indicator calculations
    # Then apply to test period
    # Note: For regime detection, we need warmup period from train data
    
    # Combine train and test with train providing warmup for indicators
    combined = pd.concat([train_df, test_df], ignore_index=True)
    combined = combined.sort_values("Timestamp").reset_index(drop=True)
    
    # Build signals on combined data
    signals = build_regime_signals(combined, params)
    
    # Find where test period starts
    test_start_ts = test_df["Timestamp"].min()
    test_mask = combined["Timestamp"] >= test_start_ts
    test_start_idx = test_mask.idxmax()
    
    # Run backtest ONLY on test period
    test_data = combined.loc[test_start_idx:].reset_index(drop=True)
    test_signals = signals.loc[test_start_idx:].reset_index(drop=True)
    
    trades_df, equity_df, summary = run_regime_backtest(
        test_data, test_signals, params,
        initial_cash=initial_cash,
        fee_bps=fee_bps,
        slippage_bps=slippage_bps,
    )
    
    return {
        "trades_df": trades_df,
        "equity_df": equity_df,
        "summary": summary,
    }


def run_walkforward(
    df: pd.DataFrame,
    params: RegimeTrendParams,
    train_days: int = 180,
    test_days: int = 30,
    step_days: int = 30,
    initial_cash: float = 10000.0,
    fee_bps: float = 6.0,
    slippage_bps: float = 4.0,
) -> Dict:
    """
    Full walk-forward validation.
    Returns per-fold results and aggregate statistics.
    """
    folds = walk_forward_folds(df, train_days, test_days, step_days)
    
    if not folds:
        raise ValueError(f"No folds generated. Data span may be too short for {train_days}d train + {test_days}d test")
    
    per_fold = []
    all_trades = []
    equity_series = []
    
    current_cash = initial_cash
    
    for fold_id, (train_start, train_end, test_start, test_end) in enumerate(folds, 1):
        train_df = df[(df["Timestamp"] >= train_start) & (df["Timestamp"] < train_end)].copy()
        test_df = df[(df["Timestamp"] >= test_start) & (df["Timestamp"] < test_end)].copy()
        
        if len(train_df) < 24 * 30 or len(test_df) < 24 * 7:  # Min 30 days train, 7 days test
            continue
        
        result = run_fold(
            train_df, test_df, params,
            initial_cash=current_cash,
            fee_bps=fee_bps,
            slippage_bps=slippage_bps,
        )
        
        summary = result["summary"]
        trades = result["trades_df"]
        equity = result["equity_df"]
        
        # Update cash for next fold (compounding)
        current_cash = summary["final_equity"]
        
        # Track fold results
        fold_result = {
            "fold": fold_id,
            "train_start": train_start,
            "train_end": train_end,
            "test_start": test_start,
            "test_end": test_end,
            "return_pct": summary["total_return_pct"],
            "max_dd_pct": summary["max_drawdown_pct"],
            "n_trades": summary["n_trades"],
            "win_rate": summary["win_rate_pct"],
            "profit_factor": summary["profit_factor"],
            "sharpe": summary["sharpe_annual"],
            "final_equity": summary["final_equity"],
        }
        per_fold.append(fold_result)
        
        # Collect trades with fold info
        if not trades.empty:
            trades = trades.copy()
            trades["fold"] = fold_id
            all_trades.append(trades)
        
        # Collect equity with fold info
        equity = equity.copy()
        equity["fold"] = fold_id
        equity_series.append(equity)
    
    # Aggregate results
    if not per_fold:
        raise ValueError("No folds completed successfully")
    
    returns = [f["return_pct"] for f in per_fold]
    drawdowns = [f["max_dd_pct"] for f in per_fold]
    win_rates = [f["win_rate"] for f in per_fold if f["win_rate"] is not None]
    profit_factors = [f["profit_factor"] for f in per_fold if f["profit_factor"] is not None and f["profit_factor"] < float('inf')]
    sharpes = [f["sharpe"] for f in per_fold if f["sharpe"] is not None]
    trades_per_fold = [f["n_trades"] for f in per_fold]
    
    # Calculate overall compounded return
    total_return = (current_cash / initial_cash - 1) * 100
    
    aggregate = {
        "n_folds": len(per_fold),
        "total_trades": sum(trades_per_fold),
        "total_return_pct": total_return,
        "final_equity": current_cash,
        
        # Per-fold averages
        "avg_return_per_fold": np.mean(returns),
        "std_return_per_fold": np.std(returns, ddof=1) if len(returns) > 1 else 0,
        "median_return_per_fold": np.median(returns),
        
        # Risk metrics
        "avg_max_dd": np.mean(drawdowns),
        "worst_fold_dd": min(drawdowns),
        
        # Trade metrics
        "avg_win_rate": np.mean(win_rates) if win_rates else None,
        "avg_profit_factor": np.mean(profit_factors) if profit_factors else None,
        "avg_sharpe": np.mean(sharpes) if sharpes else None,
        
        # Consistency check
        "pct_profitable_folds": (np.array(returns) > 0).mean() * 100,
    }
    
    all_trades_df = pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()
    all_equity_df = pd.concat(equity_series, ignore_index=True) if equity_series else pd.DataFrame()
    
    return {
        "per_fold": per_fold,
        "aggregate": aggregate,
        "trades": all_trades_df,
        "equity": all_equity_df,
    }


def monte_carlo_analysis(
    trades_df: pd.DataFrame,
    n_simulations: int = 1000,
    initial_cash: float = 10000.0,
) -> Dict:
    """
    Monte Carlo analysis by resampling trade returns.
    This tests if results are robust to trade ordering.
    """
    if trades_df.empty or "pnl_pct" not in trades_df.columns:
        return {"error": "No trades for Monte Carlo"}
    
    returns = trades_df["pnl_pct"].values
    n_trades = len(returns)
    
    final_equities = []
    max_drawdowns = []
    
    for _ in range(n_simulations):
        # Resample trades with replacement
        resampled = np.random.choice(returns, size=n_trades, replace=True)
        
        # Simulate equity curve
        equity = initial_cash
        peak = initial_cash
        max_dd = 0.0
        
        for ret in resampled:
            equity *= (1 + ret)
            if equity > peak:
                peak = equity
            dd = (equity / peak - 1)
            if dd < max_dd:
                max_dd = dd
        
        final_equities.append(equity)
        max_drawdowns.append(max_dd)
    
    final_equities = np.array(final_equities)
    max_drawdowns = np.array(max_drawdowns)
    
    return {
        "n_simulations": n_simulations,
        "n_trades_resampled": n_trades,
        
        # Return distribution
        "return_mean": (final_equities.mean() / initial_cash - 1) * 100,
        "return_median": (np.median(final_equities) / initial_cash - 1) * 100,
        "return_5th_pct": (np.percentile(final_equities, 5) / initial_cash - 1) * 100,
        "return_95th_pct": (np.percentile(final_equities, 95) / initial_cash - 1) * 100,
        "return_std": (final_equities.std() / initial_cash) * 100,
        
        # Drawdown distribution
        "dd_mean": max_drawdowns.mean() * 100,
        "dd_median": np.median(max_drawdowns) * 100,
        "dd_5th_pct": np.percentile(max_drawdowns, 5) * 100,  # Best case
        "dd_95th_pct": np.percentile(max_drawdowns, 95) * 100,  # Worst case
        
        # Probability of profit
        "prob_profit": (final_equities > initial_cash).mean() * 100,
    }


def main():
    print("=" * 70)
    print("WALK-FORWARD VALIDATION + MONTE CARLO ANALYSIS")
    print("Proper Out-of-Sample Testing")
    print("=" * 70)
    
    # Load data
    data_path = Path("research/backtests/data/flight_recorder.csv")
    if not data_path.exists():
        data_path = Path("data/btcusd_3600s_2019-01-01_to_2025-12-30.csv")
    
    print(f"\nLoading: {data_path}")
    df = load_flight_recorder(str(data_path))
    print(f"Loaded {len(df):,} bars")
    print(f"Date range: {df['Timestamp'].min()} to {df['Timestamp'].max()}")
    
    # FIXED parameters - NO tuning on test data
    # These were chosen based on general trend-following principles
    params = RegimeTrendParams(
        regime_sma=200,
        confirm_sma=50,
        entry_buffer_pct=3.0,
        exit_buffer_pct=0.0,
        sma_slope_bars=5,
        position_size_pct=75.0,
        use_trailing_stop=True,
        trailing_stop_pct=12.0,
        max_loss_pct=10.0,
    )
    
    print("\n--- Fixed Parameters (NO tuning on test data) ---")
    print(f"  Regime SMA: {params.regime_sma}")
    print(f"  Entry Buffer: {params.entry_buffer_pct}%")
    print(f"  Trailing Stop: {params.trailing_stop_pct}%")
    print(f"  Position Size: {params.position_size_pct}%")
    
    # Walk-forward settings
    train_days = 180   # 6 months training
    test_days = 30     # 1 month testing
    step_days = 30     # Roll forward monthly
    
    print(f"\n--- Walk-Forward Settings ---")
    print(f"  Train Period: {train_days} days")
    print(f"  Test Period: {test_days} days")
    print(f"  Step Forward: {step_days} days")
    
    print("\n" + "-" * 70)
    print("Running Walk-Forward Validation...")
    print("-" * 70)
    
    wf_results = run_walkforward(
        df, params,
        train_days=train_days,
        test_days=test_days,
        step_days=step_days,
        initial_cash=10000.0,
        fee_bps=6.0,
        slippage_bps=4.0,
    )
    
    agg = wf_results["aggregate"]
    per_fold = wf_results["per_fold"]
    
    print(f"\n{'='*70}")
    print("WALK-FORWARD RESULTS (OUT-OF-SAMPLE)")
    print(f"{'='*70}")
    
    print(f"\n--- Aggregate Performance ---")
    print(f"  {'Folds Tested:':<30} {agg['n_folds']}")
    print(f"  {'Total Trades:':<30} {agg['total_trades']}")
    print(f"  {'Total Return (Compounded):':<30} {agg['total_return_pct']:.2f}%")
    print(f"  {'Final Equity:':<30} ${agg['final_equity']:,.2f}")
    
    print(f"\n--- Per-Fold Statistics ---")
    print(f"  {'Avg Return per Fold:':<30} {agg['avg_return_per_fold']:.2f}%")
    print(f"  {'Std Dev of Returns:':<30} {agg['std_return_per_fold']:.2f}%")
    print(f"  {'Median Return per Fold:':<30} {agg['median_return_per_fold']:.2f}%")
    print(f"  {'% Profitable Folds:':<30} {agg['pct_profitable_folds']:.1f}%")
    
    print(f"\n--- Risk Metrics ---")
    print(f"  {'Avg Max DD per Fold:':<30} {agg['avg_max_dd']:.2f}%")
    print(f"  {'Worst Fold DD:':<30} {agg['worst_fold_dd']:.2f}%")
    
    print(f"\n--- Trade Metrics ---")
    if agg['avg_win_rate']:
        print(f"  {'Avg Win Rate:':<30} {agg['avg_win_rate']:.1f}%")
    if agg['avg_profit_factor']:
        print(f"  {'Avg Profit Factor:':<30} {agg['avg_profit_factor']:.2f}")
    if agg['avg_sharpe']:
        print(f"  {'Avg Sharpe (Annualized):':<30} {agg['avg_sharpe']:.2f}")
    
    # Show fold-by-fold results
    print(f"\n--- Fold-by-Fold Results ---")
    print(f"{'Fold':<6} {'Test Period':<25} {'Return':>10} {'MaxDD':>10} {'Trades':>8} {'WinRate':>10}")
    print("-" * 75)
    for f in per_fold:
        test_period = f"{f['test_start'].strftime('%Y-%m')}"
        wr = f"{f['win_rate']:.1f}%" if f['win_rate'] is not None else "n/a"
        print(f"{f['fold']:<6} {test_period:<25} {f['return_pct']:>9.2f}% {f['max_dd_pct']:>9.2f}% {f['n_trades']:>8} {wr:>10}")
    
    # Monte Carlo Analysis
    print(f"\n{'='*70}")
    print("MONTE CARLO ANALYSIS (1000 Simulations)")
    print(f"{'='*70}")
    
    trades = wf_results["trades"]
    if not trades.empty:
        mc = monte_carlo_analysis(trades, n_simulations=1000, initial_cash=10000.0)
        
        print(f"\n--- Return Distribution ---")
        print(f"  {'Mean Return:':<30} {mc['return_mean']:.2f}%")
        print(f"  {'Median Return:':<30} {mc['return_median']:.2f}%")
        print(f"  {'5th Percentile:':<30} {mc['return_5th_pct']:.2f}% (worst case)")
        print(f"  {'95th Percentile:':<30} {mc['return_95th_pct']:.2f}% (best case)")
        print(f"  {'Probability of Profit:':<30} {mc['prob_profit']:.1f}%")
        
        print(f"\n--- Drawdown Distribution ---")
        print(f"  {'Mean Max DD:':<30} {mc['dd_mean']:.2f}%")
        print(f"  {'Median Max DD:':<30} {mc['dd_median']:.2f}%")
        print(f"  {'95th Percentile DD:':<30} {mc['dd_95th_pct']:.2f}% (worst case)")
    else:
        print("\n  No trades for Monte Carlo analysis")
    
    # Quality assessment
    print(f"\n{'='*70}")
    print("VALIDATION QUALITY ASSESSMENT")
    print(f"{'='*70}")
    
    checks = []
    
    if agg['total_return_pct'] > 0:
        checks.append(f"[OK] Strategy is profitable OOS: {agg['total_return_pct']:.1f}%")
    else:
        checks.append(f"[!!] Strategy loses money OOS: {agg['total_return_pct']:.1f}%")
    
    if agg['pct_profitable_folds'] > 50:
        checks.append(f"[OK] >50% of folds profitable: {agg['pct_profitable_folds']:.1f}%")
    else:
        checks.append(f"[!!] <50% of folds profitable: {agg['pct_profitable_folds']:.1f}%")
    
    if agg['avg_profit_factor'] and agg['avg_profit_factor'] > 1.0:
        checks.append(f"[OK] Avg profit factor > 1.0: {agg['avg_profit_factor']:.2f}")
    else:
        checks.append(f"[!!] Avg profit factor <= 1.0: {agg['avg_profit_factor']}")
    
    if agg['worst_fold_dd'] > -50:
        checks.append(f"[OK] No catastrophic fold DD: {agg['worst_fold_dd']:.1f}%")
    else:
        checks.append(f"[!!] Catastrophic fold DD: {agg['worst_fold_dd']:.1f}%")
    
    if agg['std_return_per_fold'] < abs(agg['avg_return_per_fold']) * 2:
        checks.append(f"[OK] Reasonable return consistency")
    else:
        checks.append(f"[!!] High variance in returns")
    
    for c in checks:
        print(f"  {c}")
    
    ok_count = sum(1 for c in checks if c.startswith("[OK]"))
    print(f"\n  Passed {ok_count}/{len(checks)} validation checks")
    
    # Save results
    out_dir = Path("research/backtests/results")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    pd.DataFrame(per_fold).to_csv(out_dir / "walkforward_folds.csv", index=False)
    wf_results["trades"].to_csv(out_dir / "walkforward_trades.csv", index=False)
    
    print(f"\n  Results saved to {out_dir}")
    print("=" * 70)
    
    return wf_results


if __name__ == "__main__":
    main()






























