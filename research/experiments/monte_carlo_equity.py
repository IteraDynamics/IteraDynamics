"""
monte_carlo_equity.py

Monte Carlo (block bootstrap) on an EQUITY CURVE CSV produced by calendar_equity_sim.py.

Reads an equity curve CSV (Timestamp + Equity columns), converts to per-bar returns,
then resamples returns using:
  - mode=block (default): sample contiguous blocks of returns to preserve streaks
  - mode=iidr: iid resample of returns

Outputs a samples CSV with:
  ending_equity, total_return, cagr, max_drawdown

Example (PowerShell):
  python .\monte_carlo_equity.py `
    --curve ".\wf_runs\20260109_110637\EQ_Aprime_h48_c064_max025.csv" `
    --iters 20000 `
    --mode block --block_size 3 `
    --seed 42 `
    --out ".\wf_runs\20260109_110637\MC_EQ_max025_block3.csv"
"""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def pick_column(cols, candidates):
    lower = {c.lower(): c for c in cols}
    for cand in candidates:
        if cand.lower() in lower:
            return lower[cand.lower()]
    return None


def load_equity_curve(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    ts_col = pick_column(df.columns, ["timestamp", "time", "date", "ts"])
    eq_col = pick_column(df.columns, ["equity", "equity_curve", "nav", "balance"])

    if ts_col is None or eq_col is None:
        raise ValueError(
            f"Could not find Timestamp/Equity columns in {path}.\n"
            f"Columns found: {list(df.columns)}\n"
            "Expected something like: Timestamp, equity"
        )

    out = df[[ts_col, eq_col]].copy()
    out.columns = ["Timestamp", "Equity"]
    out["Timestamp"] = pd.to_datetime(out["Timestamp"], errors="coerce")
    out = out.dropna(subset=["Timestamp", "Equity"]).sort_values("Timestamp").reset_index(drop=True)

    out["Equity"] = pd.to_numeric(out["Equity"], errors="coerce")
    out = out.dropna(subset=["Equity"]).reset_index(drop=True)

    if len(out) < 10:
        raise ValueError(f"Equity curve too short after cleanup: n={len(out)}")

    if (out["Equity"] <= 0).any():
        raise ValueError("Equity must be > 0 for all rows.")

    return out


def compute_returns(eq: np.ndarray) -> np.ndarray:
    # r[t] corresponds to return from t-1 -> t
    r = eq[1:] / eq[:-1] - 1.0
    r = r[np.isfinite(r)]
    return r


def max_drawdown(equity: np.ndarray) -> float:
    # returns negative number (e.g. -0.12 = -12% drawdown)
    peak = np.maximum.accumulate(equity)
    dd = equity / peak - 1.0
    return float(np.min(dd))


def infer_years(ts: pd.Series) -> float:
    span_seconds = (ts.iloc[-1] - ts.iloc[0]).total_seconds()
    years = span_seconds / (365.25 * 24 * 3600)
    return float(years) if years > 0 else 0.0


def bootstrap_indices_block(n: int, iters: int, block_size: int, rng: np.random.Generator) -> np.ndarray:
    """
    Returns an (iters, n) matrix of indices into returns[0..n-1],
    built by concatenating random blocks of length block_size.
    """
    if block_size <= 0:
        raise ValueError("--block_size must be >= 1")
    if block_size > n:
        block_size = n

    # starting positions for blocks
    max_start = n - block_size
    if max_start < 0:
        max_start = 0

    out = np.empty((iters, n), dtype=np.int32)
    for k in range(iters):
        filled = 0
        while filled < n:
            start = int(rng.integers(0, max_start + 1))
            take = min(block_size, n - filled)
            out[k, filled:filled + take] = np.arange(start, start + take, dtype=np.int32)
            filled += take
    return out


def bootstrap_indices_iidr(n: int, iters: int, rng: np.random.Generator) -> np.ndarray:
    return rng.integers(0, n, size=(iters, n), dtype=np.int32)


def summarize(samples: pd.DataFrame) -> None:
    def q(x):
        return np.quantile(x, [0.01, 0.05, 0.10, 0.50, 0.90, 0.95, 0.99])

    tr = samples["total_return"].values
    cagr = samples["cagr"].values
    mdd = samples["max_drawdown"].values

    print("\n================ EQUITY MONTE CARLO SUMMARY ================")
    print(f"iters: {len(samples)}")
    print(f"P(total_return > 0): {float((tr > 0).mean()):.3f}")
    print(f"P(CAGR > 0):         {float((cagr > 0).mean()):.3f}")

    trq = q(tr)
    print("\ntotal_return quantiles:")
    print(f"  q01: {trq[0]: .6f}")
    print(f"  q05: {trq[1]: .6f}")
    print(f"  q10: {trq[2]: .6f}")
    print(f"  q50: {trq[3]: .6f}")
    print(f"  q90: {trq[4]: .6f}")
    print(f"  q95: {trq[5]: .6f}")
    print(f"  q99: {trq[6]: .6f}")

    cagrq = q(cagr)
    print("\nCAGR quantiles:")
    print(f"  q01: {cagrq[0]: .6f}")
    print(f"  q05: {cagrq[1]: .6f}")
    print(f"  q10: {cagrq[2]: .6f}")
    print(f"  q50: {cagrq[3]: .6f}")
    print(f"  q90: {cagrq[4]: .6f}")
    print(f"  q95: {cagrq[5]: .6f}")
    print(f"  q99: {cagrq[6]: .6f}")

    mddq = q(mdd)
    print("\nmax_drawdown quantiles (negative = drawdown):")
    print(f"  q01: {mddq[0]: .6f}   ({mddq[0]*100:.2f}%)")
    print(f"  q05: {mddq[1]: .6f}   ({mddq[1]*100:.2f}%)")
    print(f"  q10: {mddq[2]: .6f}   ({mddq[2]*100:.2f}%)")
    print(f"  q50: {mddq[3]: .6f}   ({mddq[3]*100:.2f}%)")
    print(f"  q90: {mddq[4]: .6f}   ({mddq[4]*100:.2f}%)")
    print(f"  q95: {mddq[5]: .6f}   ({mddq[5]*100:.2f}%)")
    print(f"  q99: {mddq[6]: .6f}   ({mddq[6]*100:.2f}%)")
    print("============================================================\n")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--curve", type=str, required=True, help="Path to equity curve CSV from calendar_equity_sim.py")
    p.add_argument("--iters", type=int, default=20000)
    p.add_argument("--mode", type=str, default="block", choices=["block", "iidr"])
    p.add_argument("--block_size", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=str, required=True, help="Output CSV for MC samples")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    curve_path = Path(args.curve)
    if not curve_path.exists():
        raise FileNotFoundError(f"Equity curve not found: {curve_path}")

    df = load_equity_curve(curve_path)
    years = infer_years(df["Timestamp"])
    if years <= 0:
        raise ValueError("Could not infer a positive time span for CAGR.")

    eq = df["Equity"].astype(float).values
    rets = compute_returns(eq)
    n = len(rets)
    if n < 5:
        raise ValueError(f"Not enough return steps to bootstrap: n={n}")

    rng = np.random.default_rng(args.seed)

    if args.mode == "block":
        idx = bootstrap_indices_block(n=n, iters=args.iters, block_size=args.block_size, rng=rng)
    else:
        idx = bootstrap_indices_iidr(n=n, iters=args.iters, rng=rng)

    base_eq0 = float(eq[0])

    samples = []
    # iterate row-wise to keep memory safe; n is modest for ~1-2y hourly
    for k in range(args.iters):
        r = rets[idx[k]]
        # reconstruct equity path
        e = np.empty(n + 1, dtype=float)
        e[0] = base_eq0
        e[1:] = base_eq0 * np.cumprod(1.0 + r)

        end_eq = float(e[-1])
        total_ret = end_eq / base_eq0 - 1.0
        cagr = (end_eq / base_eq0) ** (1.0 / years) - 1.0
        mdd = max_drawdown(e)

        samples.append((end_eq, total_ret, cagr, mdd))

    out_df = pd.DataFrame(samples, columns=["ending_equity", "total_return", "cagr", "max_drawdown"])

    print(f"CURVE: {curve_path} | rows={len(df)} rets={n} years={years:.4f}")
    print(f"mode={args.mode} iters={args.iters} seed={args.seed}")
    if args.mode == "block":
        print(f"block_size={args.block_size}")

    summarize(out_df)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)
    print(f"Saved MC samples -> {out_path}")


if __name__ == "__main__":
    main()
