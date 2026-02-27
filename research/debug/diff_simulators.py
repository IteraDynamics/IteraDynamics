# research/debug/diff_simulators.py
"""
Behavioral diff between two simulation engines (instrumentation only).

Loads:
  - debug/harness_btc_trace.csv (from research.harness.backtest_runner)
  - debug/geometry_btc_trace.csv (from research/experiments/portfolio_geometry_validation.py)

Aligns by timestamp (inner join), computes bar-to-bar returns from equity (simple or log),
and compares return series. Divergence is reported when return difference exceeds tolerance.
Equity scale (e.g. 10000 vs 1.0) is irrelevant because we compare returns.

Usage:
  python research/debug/diff_simulators.py
  python research/debug/diff_simulators.py --tolerance 1e-6 --mode log --start 2019-06-01 --end 2020-01-01
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _load_trace(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Trace file not found: {path}\n"
            "Generate traces first (see instructions printed when files are missing)."
        )
    df = pd.read_csv(path)
    for col in ("timestamp", "close_price", "exposure", "next_bar_return", "portfolio_return", "equity"):
        if col not in df.columns:
            raise ValueError(f"Trace missing column: {col}")
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp"])
    return df


def _align_by_timestamp(
    h: pd.DataFrame, g: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Inner join on timestamp; returns (harness_aligned, geometry_aligned), sorted by timestamp."""
    h = h.sort_values("timestamp").drop_duplicates(subset=["timestamp"], keep="last")
    g = g.sort_values("timestamp").drop_duplicates(subset=["timestamp"], keep="last")
    merged = pd.merge(
        h[["timestamp"]],
        g[["timestamp"]],
        on="timestamp",
        how="inner",
    )
    merged = merged.sort_values("timestamp").reset_index(drop=True)
    h_aligned = merged.merge(h, on="timestamp", how="left")
    g_aligned = merged.merge(g, on="timestamp", how="left")
    return h_aligned, g_aligned


def _equity_returns_simple(equity: np.ndarray) -> np.ndarray:
    """r_t = equity_t / equity_{t-1} - 1. First element is NaN (no prior bar)."""
    out = np.full(len(equity), np.nan, dtype=float)
    if len(equity) < 2:
        return out
    prev = equity[:-1]
    curr = equity[1:]
    valid = (prev > 0) & (curr > 0)
    out[1:] = np.where(valid, (curr / prev) - 1.0, np.nan)
    return out


def _equity_returns_log(equity: np.ndarray) -> np.ndarray:
    """lr_t = log(equity_t) - log(equity_{t-1}). First element is NaN. Requires equity > 0."""
    out = np.full(len(equity), np.nan, dtype=float)
    if len(equity) < 2:
        return out
    prev = equity[:-1]
    curr = equity[1:]
    if np.any(prev <= 0) or np.any(curr <= 0):
        raise ValueError("Log return mode requires equity > 0 for all bars; use --mode simple or fix data.")
    out[1:] = np.log(curr) - np.log(prev)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Diff harness vs geometry by bar-to-bar returns (scale-invariant)")
    ap.add_argument(
        "--harness",
        type=str,
        default=None,
        help="Path to harness trace CSV (default: repo_root/debug/harness_btc_trace.csv)",
    )
    ap.add_argument(
        "--geometry",
        type=str,
        default=None,
        help="Path to geometry trace CSV (default: repo_root/debug/geometry_btc_trace.csv)",
    )
    ap.add_argument(
        "--tolerance",
        type=float,
        default=1e-8,
        help="Absolute return-diff tolerance (default: 1e-8)",
    )
    ap.add_argument(
        "--mode",
        choices=["simple", "log"],
        default="simple",
        help="Return type: simple (r = E_t/E_{t-1}-1) or log (default: simple)",
    )
    ap.add_argument(
        "--start",
        type=str,
        default=None,
        help="Optional start date (inclusive) for comparison window (YYYY-MM-DD)",
    )
    ap.add_argument(
        "--end",
        type=str,
        default=None,
        help="Optional end date (inclusive) for comparison window (YYYY-MM-DD)",
    )
    args = ap.parse_args()

    root = _repo_root()
    harness_path = Path(args.harness) if args.harness else root / "debug" / "harness_btc_trace.csv"
    geometry_path = Path(args.geometry) if args.geometry else root / "debug" / "geometry_btc_trace.csv"

    if not harness_path.exists() or not geometry_path.exists():
        missing = [p for p in (harness_path, geometry_path) if not p.exists()]
        print("Trace file(s) not found. Generate them first, then re-run this script.")
        for p in missing:
            print(f"  Missing: {p}")
        print("\nGenerate traces (use same data + env + gross mode for both):")
        print("  Quick pass: set ARGUS_DEBUG_TRACE_MAX_BARS=2000 and use --debug_trace_max_bars 2000")
        print("  1. Harness:")
        print("     $env:ARGUS_DATA_FILE='path/to/btc.csv'")
        print("     $env:ARGUS_FEE_BPS=0")
        print("     $env:ARGUS_SLIPPAGE_BPS=0")
        print("     $env:ARGUS_STRATEGY_MODULE='research.strategies.sg_core_exposure_v2'")
        print("     $env:ARGUS_DEBUG_TRACE_MAX_BARS=2000   # optional")
        print("     # load .env (macro EMA=2000, bear cap=0.00) then:")
        print('     python -c "import sys; sys.path.insert(0, r\'./runtime/argus\'); from research.harness.backtest_runner import main; main()"')
        print("  2. Geometry (replace paths with your BTC/ETH CSV paths):")
        print('     python research/experiments/portfolio_geometry_validation.py --mode gross --btc_data_file path/to/btc.csv --eth_data_file path/to/eth.csv --debug_trace_max_bars 2000')
        print("     # add --env_file path/to/.env if you use one")
        return 1

    print("Loading traces...")
    h_df = _load_trace(harness_path)
    g_df = _load_trace(geometry_path)
    print(f"  Harness:  {len(h_df)} rows")
    print(f"  Geometry: {len(g_df)} rows")

    h_aligned, g_aligned = _align_by_timestamp(h_df, g_df)
    n = len(h_aligned)
    print(f"  Aligned (inner join on timestamp): {n} rows")

    if n == 0:
        print("No common timestamps. Possible bar dropping or different date windows.")
        return 1

    # Optional date window filter (strict: only keep rows in [start, end])
    if args.start is not None or args.end is not None:
        ts = pd.to_datetime(h_aligned["timestamp"], utc=True)
        if args.start is not None:
            start_utc = pd.Timestamp(args.start, tz="UTC")
            mask = ts >= start_utc
        else:
            mask = np.ones(n, dtype=bool)
        if args.end is not None:
            end_utc = pd.Timestamp(args.end, tz="UTC")
            mask = mask & (ts <= end_utc)
        h_aligned = h_aligned.loc[mask].reset_index(drop=True)
        g_aligned = g_aligned.loc[mask].reset_index(drop=True)
        n = len(h_aligned)
        print(f"  After --start/--end window: {n} rows")

    if n < 2:
        print("Fewer than 2 aligned bars; cannot compute returns.")
        return 1

    # Equity series (aligned, sorted by timestamp)
    h_equity = h_aligned["equity"].to_numpy(dtype=float)
    g_equity = g_aligned["equity"].to_numpy(dtype=float)

    # Bar-to-bar returns from equity (scale-invariant)
    if args.mode == "log":
        try:
            h_ret = _equity_returns_log(h_equity)
            g_ret = _equity_returns_log(g_equity)
        except ValueError as e:
            print(f"Error: {e}")
            return 1
    else:
        h_ret = _equity_returns_simple(h_equity)
        g_ret = _equity_returns_simple(g_equity)

    # Drop first bar (no prior for return)
    h_ret = h_ret[1:]
    g_ret = g_ret[1:]
    timestamps = h_aligned["timestamp"].iloc[1:].values
    h_equity_tail = h_equity[1:]
    g_equity_tail = g_equity[1:]
    n_ret = len(h_ret)

    # Return diff (absolute)
    ret_diff = np.abs(h_ret - g_ret)
    valid = ~(np.isnan(h_ret) | np.isnan(g_ret))
    ret_diff_valid = np.where(valid, ret_diff, np.nan)
    tol = args.tolerance

    over_tol = valid & (ret_diff > tol)
    count_over = int(np.sum(over_tol))
    n_valid = int(np.sum(valid))
    pct_over = (100.0 * count_over / n_valid) if n_valid > 0 else 0.0
    max_abs_diff = float(np.nanmax(ret_diff_valid)) if n_valid > 0 else float("nan")
    mean_abs_diff = float(np.nanmean(ret_diff_valid)) if n_valid > 0 else float("nan")

    # First divergence index (first bar where return diff > tolerance)
    first_div_idx = None
    for i in range(n_ret):
        if over_tol[i]:
            first_div_idx = i
            break

    # --- Report ---
    print()
    print("Return comparison (scale-invariant; bar-to-bar returns from equity)")
    print(f"  Mode: {args.mode}  Tolerance: {tol}")
    print(f"  Valid return pairs: {n_valid}  (first aligned bar dropped)")
    print()
    print("Summary:")
    print(f"  max_abs_return_diff:  {max_abs_diff}")
    print(f"  mean_abs_return_diff:  {mean_abs_diff}")
    print(f"  count_over_tolerance:  {count_over}")
    print(f"  % over tolerance:     {pct_over:.2f}%")
    print()

    if first_div_idx is not None:
        ts = timestamps[first_div_idx]
        hr = h_ret[first_div_idx]
        gr = g_ret[first_div_idx]
        diff = float(h_ret[first_div_idx] - g_ret[first_div_idx])
        heq = h_equity_tail[first_div_idx]
        geq = g_equity_tail[first_div_idx]
        print("=" * 60)
        print("FIRST DIVERGENCE (abs(return_diff) > tolerance)")
        print("=" * 60)
        print(f"timestamp:        {ts}")
        print(f"harness return:   {hr}")
        print(f"geometry return: {gr}")
        print(f"return diff:      {diff}")
        print(f"harness equity:  {heq}  (context)")
        print(f"geometry equity: {geq}  (context)")
        print("=" * 60)
        return 0

    print("No divergence: return series match within tolerance.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
