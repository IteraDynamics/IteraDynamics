# research/debug/diff_simulators.py
"""
Behavioral diff between two simulation engines (instrumentation only).

Loads:
  - debug/harness_btc_trace.csv (from research.harness.backtest_runner)
  - debug/geometry_btc_trace.csv (from research/experiments/portfolio_geometry_validation.py)

Aligns by timestamp (inner join), computes bar-to-bar returns from equity (simple or log),
and compares (a) return series and (b) equity after scaling geometry to USD. At first
divergence prints diagnostic fields (desired_exposure_frac, applied_exposure, bar_return_*,
fee_slippage_this_bar, rebalanced) for that bar and 1-2 prior bars.

Root cause / alignment (for matching runs):
  - Harness first row = bar lookback (default 200); geometry trace uses BTC-only timeline
  and slices from bar 200 so the first row is the same bar. Both use closed-bar t→t+1:
  exposure at bar t applies to return from bar t to t+1.

Usage:
  python research/debug/diff_simulators.py
  python research/debug/diff_simulators.py --tolerance 1e-6 --mode log --start 2019-06-01 --end 2020-01-01
  python research/debug/diff_simulators.py --initial_equity 10000
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _load_trace(path: Path, *, require_equity_column: bool = True) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Trace file not found: {path}\n"
            "Generate traces first (see instructions printed when files are missing)."
        )
    df = pd.read_csv(path)
    for col in ("timestamp", "close_price", "exposure", "next_bar_return", "portfolio_return"):
        if col not in df.columns:
            raise ValueError(f"Trace missing column: {col}")
    # Equity: either "equity" (harness / legacy geometry) or "equity_index" (+ optional "equity_usd")
    if require_equity_column:
        has_equity = "equity" in df.columns
        has_index = "equity_index" in df.columns
        if not (has_equity or has_index):
            raise ValueError("Trace must have 'equity' or 'equity_index' column")
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
    ap.add_argument(
        "--initial_equity",
        type=float,
        default=10000.0,
        help="Initial equity in USD; used to scale geometry equity_index to USD when equity_usd not in trace (default: 10000)",
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
        print("\nGenerate traces (use same data + SAME ENV + gross mode for both):")
        print("  Strategy/regime params (SG_CORE_*, REGIME_*) must match or returns will differ.")
        print("  Set ARGUS_ENV_FILE to the same .env as geometry's --env_file so the harness loads it.")
        print("  Quick pass: ARGUS_DEBUG_TRACE_MAX_BARS=2000 and --debug_trace_max_bars 2000")
        print("  1. Harness (same .env via ARGUS_ENV_FILE):")
        print("     $env:ARGUS_ENV_FILE='research/configs/core_v2/btc_core_v2_tuned_2026_02_27.env'")
        print("     $env:ARGUS_DATA_FILE='data/btcusd_3600s_2019-01-01_to_2025-12-30.csv'")
        print("     $env:ARGUS_FEE_BPS=0")
        print("     $env:ARGUS_SLIPPAGE_BPS=0")
        print("     $env:ARGUS_STRATEGY_MODULE='research.strategies.sg_core_exposure_v2'")
        print("     $env:ARGUS_DEBUG_TRACE_MAX_BARS=2000")
        print('     python -c "import sys; sys.path.insert(0, r\'./runtime/argus\'); from research.harness.backtest_runner import main; main()"')
        print("  2. Geometry (same .env via --env_file):")
        print('     python research/experiments/portfolio_geometry_validation.py --mode gross --btc_data_file data/btcusd_3600s_2019-01-01_to_2025-12-30.csv --eth_data_file data/ethusd_3600s_2019-01-01_to_2025-12-30.csv --debug_trace_max_bars 2000 --initial_equity 10000 --env_file research/configs/core_v2/btc_core_v2_tuned_2026_02_27.env')
        return 1

    print("Loading traces...")
    h_df = _load_trace(harness_path)
    g_df = _load_trace(geometry_path)
    print(f"  Harness:  {len(h_df)} rows")
    print(f"  Geometry: {len(g_df)} rows")

    # Harness: equity in USD; geometry: equity_index (1.0) and optionally equity_usd
    if "equity" not in h_df.columns:
        raise ValueError("Harness trace must have 'equity' column (USD)")
    h_equity_col = "equity"
    if "equity_index" in g_df.columns:
        g_equity_col = "equity_index"  # for return computation (scale-invariant)
        g_has_usd = "equity_usd" in g_df.columns
    else:
        g_equity_col = "equity"
        g_has_usd = False

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

    # Equity series (aligned): harness in USD; geometry as index for returns
    h_equity = h_aligned["equity"].to_numpy(dtype=float)
    g_equity = g_aligned[g_equity_col].to_numpy(dtype=float)
    # Geometry equity in USD for scale comparison (use equity_usd column or scale index)
    if g_has_usd:
        g_equity_usd = g_aligned["equity_usd"].to_numpy(dtype=float)
    else:
        g_equity_usd = g_equity * float(args.initial_equity)

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
    g_equity_usd_tail = g_equity_usd[1:]
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

    # Treat as "aligned" if only float noise (max diff << 1e-6)
    FLOAT_NOISE_TOL = 1e-6
    negligible_diff = max_abs_diff < FLOAT_NOISE_TOL if n_valid > 0 else True

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
    if negligible_diff and first_div_idx is not None:
        print(f"  -> max diff < {FLOAT_NOISE_TOL} (float noise); simulators aligned.")
    print()

    DIAG_COLS = [
        "desired_exposure_frac", "applied_exposure", "bar_return_px", "bar_return_applied",
        "fee_slippage_this_bar", "rebalanced",
    ]

    def _print_diagnostic_rows(
        h_aligned: pd.DataFrame,
        g_aligned: pd.DataFrame,
        indices: list[int],
        label: str,
    ) -> None:
        """Print diagnostic columns for given row indices (indices into full aligned frames)."""
        for idx in indices:
            if idx < 0 or idx >= len(h_aligned):
                continue
            ts = h_aligned["timestamp"].iloc[idx]
            print(f"  --- {label} row idx={idx} ts={ts} ---")
            print(f"      close:        h={h_aligned['close_price'].iloc[idx]:.4f}  g={g_aligned['close_price'].iloc[idx]:.4f}")
            for col in DIAG_COLS:
                if col in h_aligned.columns and col in g_aligned.columns:
                    hv = h_aligned[col].iloc[idx]
                    gv = g_aligned[col].iloc[idx]
                    print(f"      {col:24s} h={hv}  g={gv}")
            if "equity" in h_aligned.columns and "equity_usd" in g_aligned.columns:
                print(f"      {'equity (USD)':24s} h={h_aligned['equity'].iloc[idx]:.2f}  g={g_aligned['equity_usd'].iloc[idx]:.2f}")
            elif "equity" in h_aligned.columns and "equity_index" in g_aligned.columns:
                print(f"      {'equity':24s} h={h_aligned['equity'].iloc[idx]:.2f}  g_idx={g_aligned['equity_index'].iloc[idx]}")

    if first_div_idx is not None:
        ts = timestamps[first_div_idx]
        hr = h_ret[first_div_idx]
        gr = g_ret[first_div_idx]
        diff = float(h_ret[first_div_idx] - g_ret[first_div_idx])
        heq = h_equity_tail[first_div_idx]
        geq = g_equity_tail[first_div_idx]
        geq_usd = g_equity_usd_tail[first_div_idx]
        print("=" * 60)
        print("FIRST DIVERGENCE (abs(return_diff) > tolerance)")
        print("=" * 60)
        print(f"timestamp:        {ts}")
        print(f"harness return:   {hr}")
        print(f"geometry return: {gr}")
        print(f"return diff:      {diff}")
        print(f"harness equity (USD):  {heq}")
        print(f"geometry equity (USD, scaled): {geq_usd}  (raw index: {geq})")
        print()
        print("Diagnostics at divergence bar and prior 1-2 bars (return at first_div is from row first_div -> first_div+1):")
        # first_div_idx: return from aligned row first_div_idx to first_div_idx+1
        prior_start = max(0, first_div_idx - 2)
        end_inclusive = min(len(h_aligned), first_div_idx + 2)
        _print_diagnostic_rows(
            h_aligned, g_aligned,
            list(range(prior_start, end_inclusive)),
            "bar",
        )
        if negligible_diff:
            print("Note: max_abs_return_diff < 1e-6 (floating-point noise only). Simulators aligned.")
        print("=" * 60)
        return 0

    # Scaled equity comparison: harness USD vs geometry USD (from column or index * initial_equity)
    equity_diff = np.abs(h_equity_tail - g_equity_usd_tail)
    valid_eq = ~(np.isnan(h_equity_tail) | np.isnan(g_equity_usd_tail))
    max_equity_diff = float(np.nanmax(np.where(valid_eq, equity_diff, np.nan))) if np.any(valid_eq) else 0.0
    # Relative tolerance for equity: e.g. 1e-6 of typical scale
    equity_tol = max(1e-6 * args.initial_equity, 1e-6)
    equity_match = max_equity_diff <= equity_tol

    print("Scaled equity comparison (harness USD vs geometry USD):")
    print(f"  max_abs_equity_diff: {max_equity_diff}")
    print(f"  tolerance:          {equity_tol}")
    print(f"  scaled equity match: {'yes' if equity_match else 'no'}")
    print()

    if equity_match:
        print("No divergence: return series match within tolerance.")
        print("Equity differs only by scale (returns match and scaled equity matches).")
    else:
        print("No return divergence, but scaled equity differs beyond tolerance.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
