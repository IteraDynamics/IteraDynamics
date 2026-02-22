"""
Portfolio Runner for Static Allocation (Core + Sleeve)
=======================================================

Runs two strategies independently using existing backtest_runner,
then combines their returns with static weights.

Env-based injection:
    ARGUS_DATA_FILE           = path to OHLCV CSV
    ARGUS_LOOKBACK            = lookback period (default: 200)
    PORT_CORE_MODULE          = e.g., "research.strategies.sg_core_exposure_v2"
    PORT_CORE_FUNC            = e.g., "generate_intent"
    PORT_SLEEVE_MODULE        = e.g., "research.strategies.sg_mean_reversion_a"
    PORT_SLEEVE_FUNC          = e.g., "generate_intent"
    PORT_W_CORE               = core weight (default: 0.5)
    PORT_W_SLEEVE             = sleeve weight (default: 0.5)

Optional:
    ARGUS_FEE_BPS             = trading fees in bps (default: 10)
    ARGUS_SLIPPAGE_BPS        = slippage in bps (default: 5)

Run from repo root:
    python -c "import sys; sys.path.insert(0, r'./runtime/argus'); from research.harness.portfolio_runner import main; main()"

Or with CLI args:
    python runtime/argus/research/harness/portfolio_runner.py --start 2023-01-01 --end 2025-12-30 --out portfolio_test
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd

# Setup path
_this_file = Path(__file__).resolve()
_argus_dir = _this_file.parent.parent.parent
if str(_argus_dir) not in sys.path:
    sys.path.insert(0, str(_argus_dir))

# Import existing backtest logic
from research.harness.backtest_runner import (
    load_flight_recorder,
    load_strategy_func,
    run_backtest,
    compute_sortino,
    compute_calmar,
    compute_drawdown_series,
)


# ---------------------------
# Configuration
# ---------------------------

def _env_str(name: str, default: str) -> str:
    v = os.environ.get(name, "").strip()
    return v if v else default


def _env_float(name: str, default: float) -> float:
    v = os.environ.get(name, "").strip()
    if not v:
        return default
    try:
        return float(v)
    except Exception:
        return default


def _env_int(name: str, default: int) -> int:
    v = os.environ.get(name, "").strip()
    if not v:
        return default
    try:
        return int(float(v))
    except Exception:
        return default


# ---------------------------
# Portfolio combination
# ---------------------------

def combine_equity_curves(
    core_equity: pd.Series,
    sleeve_equity: pd.Series,
    w_core: float,
    w_sleeve: float,
) -> pd.Series:
    """
    Combine two equity curves using static weights on per-bar returns.
    
    Returns combined equity curve aligned to common timestamps.
    """
    # Align timestamps
    common_idx = core_equity.index.intersection(sleeve_equity.index)
    core_eq = core_equity.loc[common_idx]
    sleeve_eq = sleeve_equity.loc[common_idx]
    
    # Compute per-bar returns
    core_ret = core_eq.pct_change().fillna(0.0)
    sleeve_ret = sleeve_eq.pct_change().fillna(0.0)
    
    # Weighted returns
    port_ret = w_core * core_ret + w_sleeve * sleeve_ret
    
    # Rebuild equity curve from returns
    port_equity = (1 + port_ret).cumprod()
    
    # Normalize to start at same initial equity as core
    port_equity = port_equity * core_eq.iloc[0]
    
    return port_equity


def compute_portfolio_metrics(
    equity_series: pd.Series,
    core_metrics: Dict[str, Any],
    sleeve_metrics: Dict[str, Any],
    w_core: float,
    w_sleeve: float,
) -> Dict[str, Any]:
    """Compute portfolio-level metrics."""
    
    equity_vals = equity_series.values
    
    # Total return (as decimal, like backtest_runner)
    total_return = (equity_vals[-1] / equity_vals[0]) - 1
    
    # CAGR
    years = len(equity_vals) / (365.25 * 24)  # Assuming hourly data
    cagr = (equity_vals[-1] / equity_vals[0]) ** (1 / years) - 1
    
    # Drawdown
    dd_series = compute_drawdown_series(equity_vals)
    max_dd = abs(dd_series.min())  # Report as positive, like backtest_runner
    
    # Calmar
    calmar = compute_calmar(cagr, max_dd)
    
    # Sortino
    returns = equity_series.pct_change().dropna().values
    sortino = compute_sortino(returns, risk_free=0.0, ann_factor=8760)
    
    # Weighted average exposure and time in market
    avg_exposure = (
        w_core * core_metrics.get("avg_exposure", 0.0)
        + w_sleeve * sleeve_metrics.get("avg_exposure", 0.0)
    )
    
    time_in_market = (
        w_core * core_metrics.get("time_in_market", 0.0)
        + w_sleeve * sleeve_metrics.get("time_in_market", 0.0)
    )
    
    return {
        "total_return": total_return,
        "cagr": cagr,
        "max_drawdown": max_dd,
        "calmar": calmar,
        "sortino": sortino,
        "avg_exposure": avg_exposure,
        "time_in_market": time_in_market,
        "final_equity": equity_vals[-1],
        "initial_equity": equity_vals[0],
    }


# ---------------------------
# Main runner
# ---------------------------

def run_portfolio_backtest(
    data_file: str,
    core_module: str,
    core_func: str,
    sleeve_module: str,
    sleeve_func: str,
    w_core: float = 0.5,
    w_sleeve: float = 0.5,
    lookback: int = 200,
    fee_bps: float = 10.0,
    slippage_bps: float = 5.0,
    start_date: str = None,
    end_date: str = None,
    output_prefix: str = "portfolio",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    Run portfolio backtest for Core + Sleeve.
    
    Returns:
        (core_equity_df, sleeve_equity_df, portfolio_equity_df, metrics)
    """
    
    print(f"\n{'='*80}")
    print("PORTFOLIO BACKTEST RUNNER")
    print(f"{'='*80}")
    print(f"Data file:       {data_file}")
    print(f"Core strategy:   {core_module}.{core_func}")
    print(f"Sleeve strategy: {sleeve_module}.{sleeve_func}")
    print(f"Weights:         Core={w_core:.2f}, Sleeve={w_sleeve:.2f}")
    print(f"Lookback:        {lookback}")
    print(f"Fees:            {fee_bps} bps")
    print(f"Slippage:        {slippage_bps} bps")
    if start_date or end_date:
        print(f"Date range:      {start_date or 'start'} to {end_date or 'end'}")
    print(f"{'='*80}\n")
    
    # Load data
    print("Loading data...")
    df = load_flight_recorder(data_file)
    
    # Apply date filter if specified
    if start_date:
        df = df[df['Timestamp'] >= pd.to_datetime(start_date, utc=True)]
    if end_date:
        df = df[df['Timestamp'] <= pd.to_datetime(end_date, utc=True)]
    
    df = df.reset_index(drop=True)
    print(f"  Range: {df['Timestamp'].iloc[0]} to {df['Timestamp'].iloc[-1]}")
    print(f"  Bars:  {len(df)}\n")
    
    # Load strategies
    print("Loading strategies...")
    core_strat = load_strategy_func(core_module, core_func)
    print(f"  ✓ Core:   {core_module}.{core_func}")
    
    sleeve_strat = load_strategy_func(sleeve_module, sleeve_func)
    print(f"  ✓ Sleeve: {sleeve_module}.{sleeve_func}\n")
    
    # Run Core backtest
    print("[1/2] Running Core backtest...")
    core_equity_df, core_metrics = run_backtest(
        df,
        core_strat,
        lookback=lookback,
        fee_bps=fee_bps,
        slippage_bps=slippage_bps,
        closed_only=True,
    )
    print(f"  ✓ Core CAGR:   {core_metrics['cagr']*100:.2f}%")
    print(f"  ✓ Core Calmar: {core_metrics['calmar']:.2f}\n")
    
    # Run Sleeve backtest
    print("[2/2] Running Sleeve backtest...")
    sleeve_equity_df, sleeve_metrics = run_backtest(
        df,
        sleeve_strat,
        lookback=lookback,
        fee_bps=fee_bps,
        slippage_bps=slippage_bps,
        closed_only=True,
    )
    print(f"  ✓ Sleeve CAGR:   {sleeve_metrics['cagr']*100:.2f}%")
    print(f"  ✓ Sleeve Calmar: {sleeve_metrics['calmar']:.2f}\n")
    
    # Combine equity curves
    print("Combining portfolio with static weights...")
    core_equity_series = core_equity_df.set_index('Timestamp')['equity']
    sleeve_equity_series = sleeve_equity_df.set_index('Timestamp')['equity']
    
    portfolio_equity_series = combine_equity_curves(
        core_equity_series,
        sleeve_equity_series,
        w_core,
        w_sleeve,
    )
    
    # Compute portfolio metrics
    portfolio_metrics = compute_portfolio_metrics(
        portfolio_equity_series,
        core_metrics,
        sleeve_metrics,
        w_core,
        w_sleeve,
    )
    
    # Build portfolio equity DataFrame
    portfolio_equity_df = pd.DataFrame({
        'Timestamp': portfolio_equity_series.index,
        'equity': portfolio_equity_series.values,
    })
    
    # Save artifacts
    print(f"\nSaving artifacts (prefix: {output_prefix})...")
    core_equity_df.to_csv(f"{output_prefix}_core_equity.csv", index=False)
    print(f"  ✓ {output_prefix}_core_equity.csv")
    
    sleeve_equity_df.to_csv(f"{output_prefix}_sleeve_equity.csv", index=False)
    print(f"  ✓ {output_prefix}_sleeve_equity.csv")
    
    portfolio_equity_df.to_csv(f"{output_prefix}_portfolio_equity.csv", index=False)
    print(f"  ✓ {output_prefix}_portfolio_equity.csv")
    
    # Build full metrics output
    full_metrics = {
        "weights": {"core": w_core, "sleeve": w_sleeve},
        "config": {
            "data_file": data_file,
            "lookback": lookback,
            "fee_bps": fee_bps,
            "slippage_bps": slippage_bps,
            "start_date": start_date,
            "end_date": end_date,
        },
        "core": core_metrics,
        "sleeve": sleeve_metrics,
        "portfolio": portfolio_metrics,
    }
    
    with open(f"{output_prefix}_portfolio_metrics.json", "w") as f:
        json.dump(full_metrics, f, indent=2, default=str)
    print(f"  ✓ {output_prefix}_portfolio_metrics.json\n")
    
    # Print summary
    print(f"{'='*80}")
    print("METRICS SUMMARY")
    print(f"{'='*80}")
    print(f"{'Metric':<25} {'Core':>15} {'Sleeve':>15} {'Portfolio':>15}")
    print(f"{'-'*80}")
    print(f"{'Total Return %':<25} {core_metrics['total_return']*100:>14.2f}% {sleeve_metrics['total_return']*100:>14.2f}% {portfolio_metrics['total_return']*100:>14.2f}%")
    print(f"{'CAGR %':<25} {core_metrics['cagr']*100:>14.2f}% {sleeve_metrics['cagr']*100:>14.2f}% {portfolio_metrics['cagr']*100:>14.2f}%")
    print(f"{'Max Drawdown %':<25} {core_metrics['max_drawdown']*100:>14.2f}% {sleeve_metrics['max_drawdown']*100:>14.2f}% {portfolio_metrics['max_drawdown']*100:>14.2f}%")
    print(f"{'Calmar':<25} {core_metrics['calmar']:>15.2f} {sleeve_metrics['calmar']:>15.2f} {portfolio_metrics['calmar']:>15.2f}")
    print(f"{'Sortino':<25} {core_metrics['sortino']:>15.2f} {sleeve_metrics['sortino']:>15.2f} {portfolio_metrics['sortino']:>15.2f}")
    print(f"{'Avg Exposure %':<25} {core_metrics['avg_exposure']*100:>14.2f}% {sleeve_metrics['avg_exposure']*100:>14.2f}% {portfolio_metrics['avg_exposure']*100:>14.2f}%")
    print(f"{'Time in Market %':<25} {core_metrics['time_in_market']*100:>14.2f}% {sleeve_metrics['time_in_market']*100:>14.2f}% {portfolio_metrics['time_in_market']*100:>14.2f}%")
    print(f"{'='*80}\n")
    
    return core_equity_df, sleeve_equity_df, portfolio_equity_df, full_metrics


def main():
    """Entry point with CLI args and env-based config."""
    
    parser = argparse.ArgumentParser(
        description="Portfolio backtest: Core + Sleeve with static weights"
    )
    parser.add_argument("--start", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", help="End date (YYYY-MM-DD)")
    parser.add_argument("--out", default="portfolio", help="Output prefix")
    
    args = parser.parse_args()
    
    # Load config from env
    data_file = _env_str("ARGUS_DATA_FILE", "./flight_recorder.csv")
    lookback = _env_int("ARGUS_LOOKBACK", 200)
    fee_bps = _env_float("ARGUS_FEE_BPS", 10.0)
    slippage_bps = _env_float("ARGUS_SLIPPAGE_BPS", 5.0)
    
    core_module = _env_str("PORT_CORE_MODULE", "")
    core_func = _env_str("PORT_CORE_FUNC", "generate_intent")
    sleeve_module = _env_str("PORT_SLEEVE_MODULE", "")
    sleeve_func = _env_str("PORT_SLEEVE_FUNC", "generate_intent")
    
    w_core = _env_float("PORT_W_CORE", 0.5)
    w_sleeve = _env_float("PORT_W_SLEEVE", 0.5)
    
    # Validate
    if not core_module:
        raise ValueError("PORT_CORE_MODULE environment variable required")
    if not sleeve_module:
        raise ValueError("PORT_SLEEVE_MODULE environment variable required")
    
    # Normalize weights
    total_w = w_core + w_sleeve
    if abs(total_w - 1.0) > 0.01:
        print(f"⚠️  Normalizing weights: {w_core} + {w_sleeve} = {total_w}")
        w_core /= total_w
        w_sleeve /= total_w
        print(f"   Normalized: Core={w_core:.3f}, Sleeve={w_sleeve:.3f}\n")
    
    # Run
    run_portfolio_backtest(
        data_file=data_file,
        core_module=core_module,
        core_func=core_func,
        sleeve_module=sleeve_module,
        sleeve_func=sleeve_func,
        w_core=w_core,
        w_sleeve=w_sleeve,
        lookback=lookback,
        fee_bps=fee_bps,
        slippage_bps=slippage_bps,
        start_date=args.start,
        end_date=args.end,
        output_prefix=args.out,
    )


if __name__ == "__main__":
    main()
