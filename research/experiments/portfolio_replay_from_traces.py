"""
Portfolio Replay from Harness Traces (BTC-only, research)

Goal:
  Combine multiple strategy return streams using their actual Argus harness outputs,
  without reimplementing any strategy logic.

Inputs:
  - Two CSV trace files produced by the Argus harness, e.g.:
      debug/harness_core_trace.csv
      debug/harness_vb_trace.csv

  Each trace is expected to have:
    - A timestamp column: 'Timestamp' or 'timestamp'
    - Either:
        * an 'equity' column, OR
        * a 'portfolio_return' column, OR
        * 'exposure' and 'next_bar_return' columns

  Returns are derived strictly from trace-level information (no strategy calls).

Outputs:
  - Metrics CSV: one row per (w_core, w_vb) weight pair:
        research/experiments/output/portfolio_replay_metrics.csv  (default)
  - Optional equity curve CSV:
        research/experiments/output/portfolio_replay_equity.csv   (default)

Usage (from repo root):

  python research/experiments/portfolio_replay_from_traces.py ^
    --core_trace debug/harness_core_trace.csv ^
    --vb_trace debug/harness_vb_trace.csv

  # Custom weights and output paths:
  python research/experiments/portfolio_replay_from_traces.py ^
    --core_trace debug/harness_core_trace.csv ^
    --vb_trace debug/harness_vb_trace.csv ^
    --weights "100,0;80,20;70,30;50,50;0,100" ^
    --out_metrics research/experiments/output/portfolio_replay_metrics.csv ^
    --out_equity research/experiments/output/portfolio_replay_equity.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = REPO_ROOT / "research" / "experiments" / "output"
DEFAULT_METRICS_CSV = OUTPUT_DIR / "portfolio_replay_metrics.csv"
DEFAULT_EQUITY_CSV = OUTPUT_DIR / "portfolio_replay_equity.csv"


# ---------------------------------------------------------------------
# Helpers: loading, alignment, returns
# ---------------------------------------------------------------------


def _read_trace(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise ValueError(f"Trace file not found: {path}")
    df = pd.read_csv(path)
    # Detect timestamp column
    ts_col = None
    for cand in ("Timestamp", "timestamp", "ts"):
        if cand in df.columns:
            ts_col = cand
            break
    if ts_col is None:
        raise ValueError(f"Trace {path} missing Timestamp/timestamp column")
    df[ts_col] = pd.to_datetime(df[ts_col], utc=True, errors="coerce")
    df = df.dropna(subset=[ts_col]).sort_values(ts_col).reset_index(drop=True)
    df = df.rename(columns={ts_col: "Timestamp"})
    return df


def _compute_returns_from_trace(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-bar returns r[t] from a harness trace.

    Preference order:
      1) If 'equity' exists:
           r[t] = equity[t] / equity[t-1] - 1
      2) Else if 'portfolio_return' exists:
           r[t] = portfolio_return[t]
      3) Else if 'exposure' and 'next_bar_return' exist:
           r[t] = exposure[t] * next_bar_return[t]
    """
    out = df.copy()
    n = len(out)
    if n < 2:
        out["return"] = 0.0
        return out

    if "equity" in out.columns:
        eq = out["equity"].to_numpy(dtype=float)
        r = np.full(n, np.nan, dtype=float)
        r[1:] = (eq[1:] / np.maximum(eq[:-1], 1e-12)) - 1.0
        r[0] = 0.0
        out["return"] = r
        return out

    if "portfolio_return" in out.columns:
        r = out["portfolio_return"].to_numpy(dtype=float)
        r[0] = 0.0
        out["return"] = r
        return out

    if "exposure" in out.columns and "next_bar_return" in out.columns:
        expo = out["exposure"].to_numpy(dtype=float)
        px_ret = out["next_bar_return"].to_numpy(dtype=float)
        r = expo * px_ret
        r[0] = 0.0
        out["return"] = r
        return out

    raise ValueError(
        "Trace is missing required columns for return computation. "
        "Need one of: equity; portfolio_return; or (exposure, next_bar_return)."
    )


def _align_traces(core_df: pd.DataFrame, vb_df: pd.DataFrame) -> pd.DataFrame:
    """
    Inner join on Timestamp and return a merged DF with:
      Timestamp, core_return, vb_return
    """
    c = core_df[["Timestamp", "return"]].rename(columns={"return": "core_return"})
    v = vb_df[["Timestamp", "return"]].rename(columns={"return": "vb_return"})
    merged = pd.merge(c, v, on="Timestamp", how="inner").sort_values("Timestamp").reset_index(drop=True)
    if merged.empty or len(merged) < 2:
        raise ValueError("Aligned trace timeline too small after intersection join")
    return merged


# ---------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------


def _max_drawdown(equity: np.ndarray) -> float:
    peak = np.maximum.accumulate(equity)
    dd = (equity / np.maximum(peak, 1e-12)) - 1.0
    return float(np.nanmin(dd))


def _sortino(returns: np.ndarray, periods_per_year: float) -> float:
    r = returns.copy()
    r = r[~np.isnan(r)]
    if len(r) < 5:
        return float("nan")
    mean = np.mean(r)
    downside = r[r < 0]
    if len(downside) == 0:
        return float("inf") if mean > 0 else 0.0
    downside_dev = np.sqrt(np.mean(downside**2))
    if downside_dev <= 0:
        return float("nan")
    return float((mean / downside_dev) * np.sqrt(periods_per_year))


def _cagr(equity: np.ndarray, years: float) -> float:
    if years <= 0:
        return float("nan")
    start = equity[0]
    end = equity[-1]
    if start <= 0 or end <= 0:
        return float("nan")
    return float((end / start) ** (1.0 / years) - 1.0)


def _compute_portfolio_metrics(
    ts: pd.Series,
    portfolio_returns: np.ndarray,
    equity: np.ndarray,
) -> Dict[str, float]:
    ts_utc = pd.to_datetime(ts, utc=True)
    t_min = ts_utc.min()
    t_max = ts_utc.max()
    period_seconds = (t_max - t_min).total_seconds()
    years = period_seconds / (365.25 * 24 * 3600.0) if period_seconds > 0 else float("nan")

    cagr_v = _cagr(equity, years=years)
    maxdd_v = _max_drawdown(equity)
    calmar_v = float("nan") if (np.isnan(maxdd_v) or abs(maxdd_v) < 1e-12) else (cagr_v / abs(maxdd_v))

    # Assume hourly bars for Sortino
    n_bars = len(portfolio_returns)
    periods_per_year = (n_bars / years) if years and years > 0 else (365.25 * 24.0)
    sortino_v = _sortino(portfolio_returns, periods_per_year=periods_per_year)

    total_return = (equity[-1] / equity[0] - 1.0) if equity[0] > 0 else float("nan")

    return {
        "CAGR": cagr_v,
        "MaxDD": maxdd_v,
        "Calmar": calmar_v,
        "Sortino": sortino_v,
        "TotalReturn": total_return,
        "Years": years,
    }


# ---------------------------------------------------------------------
# Main replay logic
# ---------------------------------------------------------------------


def replay_portfolio(
    core_trace_path: Path,
    vb_trace_path: Path,
    weights: List[Tuple[float, float]],
    initial_equity: float = 10000.0,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Replay Core + VB portfolios for a list of (w_core, w_vb) weight pairs.
    Returns (metrics_df, equity_df_wide).
    """
    core_df = _compute_returns_from_trace(_read_trace(core_trace_path))
    vb_df = _compute_returns_from_trace(_read_trace(vb_trace_path))
    merged = _align_traces(core_df, vb_df)

    ts = merged["Timestamp"]
    r_core = merged["core_return"].to_numpy(dtype=float)
    r_vb = merged["vb_return"].to_numpy(dtype=float)

    metrics_rows: List[Dict[str, float]] = []
    equity_cols: Dict[str, np.ndarray] = {"Timestamp": ts.to_numpy()}

    for w_core, w_vb in weights:
        name = f"{int(w_core*100):03d}_{int(w_vb*100):03d}"
        port_ret = w_core * r_core + w_vb * r_vb

        eq = np.full(len(port_ret), np.nan, dtype=float)
        eq[0] = 1.0
        for i in range(1, len(port_ret)):
            r = port_ret[i]
            if np.isnan(r):
                eq[i] = eq[i - 1]
            else:
                eq[i] = eq[i - 1] * (1.0 + r)

        m = _compute_portfolio_metrics(ts, port_ret, eq)
        metrics_rows.append(
            {
                "w_core": w_core,
                "w_vb": w_vb,
                **m,
            }
        )
        equity_cols[f"equity_{name}"] = eq * float(initial_equity)

    metrics_df = pd.DataFrame(metrics_rows)
    equity_df = pd.DataFrame(equity_cols)
    equity_df["Timestamp"] = pd.to_datetime(equity_df["Timestamp"], utc=True)
    return metrics_df, equity_df


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------


def _parse_weights(s: str) -> List[Tuple[float, float]]:
    """
    Parse weight string like:
        "100,0;80,20;70,30;50,50;0,100"
    into list of (w_core, w_vb) in [0,1].
    """
    parts = [p.strip() for p in s.split(";") if p.strip()]
    out: List[Tuple[float, float]] = []
    for p in parts:
        a, b = [x.strip() for x in p.split(",")]
        wc = float(a) / 100.0
        wv = float(b) / 100.0
        out.append((wc, wv))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Replay BTC portfolios from Argus harness traces (Core + Volatility Breakout).",
    )
    ap.add_argument(
        "--core_trace",
        type=str,
        required=True,
        help="Path to Core harness trace CSV (e.g. debug/harness_core_trace.csv)",
    )
    ap.add_argument(
        "--vb_trace",
        type=str,
        required=True,
        help="Path to Volatility Breakout harness trace CSV (e.g. debug/harness_vb_trace.csv)",
    )
    ap.add_argument(
        "--weights",
        type=str,
        default="100,0;80,20;70,30;50,50;0,100",
        help='Semicolon-separated list of weight pairs as "core_pct,vb_pct"; default "100,0;80,20;70,30;50,50;0,100".',
    )
    ap.add_argument(
        "--initial_equity",
        type=float,
        default=10000.0,
        help="Initial equity for replayed portfolios (default 10000).",
    )
    ap.add_argument(
        "--out_metrics",
        type=str,
        default=str(DEFAULT_METRICS_CSV),
        help="Output CSV path for portfolio metrics.",
    )
    ap.add_argument(
        "--out_equity",
        type=str,
        default=str(DEFAULT_EQUITY_CSV),
        help="Output CSV path for equity curves (wide format).",
    )
    args = ap.parse_args()

    core_path = Path(args.core_trace)
    if not core_path.is_absolute():
        core_path = (REPO_ROOT / args.core_trace).resolve()
    vb_path = Path(args.vb_trace)
    if not vb_path.is_absolute():
        vb_path = (REPO_ROOT / args.vb_trace).resolve()

    # Harness always writes VB run to harness_btc_trace.csv; if user didn't rename, use it
    if not vb_path.exists() and "harness_vb_trace" in str(vb_path):
        fallback = vb_path.parent / "harness_btc_trace.csv"
        if fallback.exists():
            print(f"  (VB trace not found at {vb_path.name}; using {fallback.name})")
            vb_path = fallback

    if not core_path.exists():
        raise FileNotFoundError(f"Core trace not found: {core_path}")
    if not vb_path.exists():
        raise FileNotFoundError(
            f"VB trace not found: {vb_path}. "
            "Run the VB harness and rename debug/harness_btc_trace.csv to debug/harness_vb_trace.csv, "
            "or pass --vb_trace debug/harness_btc_trace.csv"
        )

    weights = _parse_weights(args.weights)

    print("Portfolio replay from traces")
    print(f"  Core trace : {core_path}")
    print(f"  VB trace   : {vb_path}")
    print(f"  Weights    : {weights}")
    print(f"  Init equity: {args.initial_equity}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    metrics_df, equity_df = replay_portfolio(
        core_trace_path=core_path,
        vb_trace_path=vb_path,
        weights=weights,
        initial_equity=args.initial_equity,
    )

    metrics_path = Path(args.out_metrics)
    if not metrics_path.is_absolute():
        metrics_path = (REPO_ROOT / args.out_metrics).resolve()
    equity_path = Path(args.out_equity)
    if not equity_path.is_absolute():
        equity_path = (REPO_ROOT / args.out_equity).resolve()

    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    equity_path.parent.mkdir(parents=True, exist_ok=True)

    metrics_df.to_csv(metrics_path, index=False)
    equity_df.to_csv(equity_path, index=False)

    print(f"Wrote metrics: {metrics_path}")
    print(f"Wrote equity : {equity_path}")


if __name__ == "__main__":
    main()

