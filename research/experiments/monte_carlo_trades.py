"""
monte_carlo_trades.py

Monte Carlo stress test on realized trade returns exported by offline_walkforward.py.

Purpose:
- Given a trades CSV (e.g. A_h48_c070_trades.csv), estimate distribution of outcomes
  by resampling / reordering trade returns.

This answers:
- Is the result dominated by a few trades?
- What do typical / worst-case outcomes look like under randomness?

Important:
- This is NOT market-regime simulation.
- It is a distributional robustness test of your realized trade P&L sequence.

Inputs expected (from offline_walkforward.py trades_out):
- net_ret column (float, per-trade net return after costs)
Optional:
- fold column (int) if you want fold-aware bootstrap

Default approach:
- IID bootstrap of net_ret with replacement, preserving number of trades.

Options:
- --mode iidr   : IID resample with replacement (default)
- --mode permute: shuffle existing sequence (no replacement)
- --mode block  : block bootstrap to preserve some autocorrelation

Run (PowerShell):
  python .\monte_carlo_trades.py --trades ".\wf_runs\20260109_082547\A_h48_c070_trades.csv" --iters 20000
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Optional

import numpy as np
import pandas as pd


# -----------------------------
# Config
# -----------------------------

@dataclass
class MCConfig:
    iters: int = 20000
    seed: int = 42
    mode: str = "iidr"          # iidr | permute | block
    block_size: int = 5         # for block bootstrap
    compounding: bool = False   # False = additive sum of returns (matches walkforward summary)
    start_equity: float = 1.0   # only used if compounding=True
    max_trades: Optional[int] = None  # cap trades (for quick tests)


# -----------------------------
# Core helpers
# -----------------------------

def load_net_returns(trades_path: Path, max_trades: Optional[int] = None) -> np.ndarray:
    df = pd.read_csv(trades_path)
    if "net_ret" not in df.columns:
        raise ValueError(f"Trades CSV missing 'net_ret' column. Found: {list(df.columns)}")

    net = df["net_ret"].astype(float).values
    net = net[np.isfinite(net)]

    if max_trades is not None and len(net) > max_trades:
        net = net[:max_trades]

    if len(net) < 5:
        raise ValueError(f"Too few trades found ({len(net)}). Monte Carlo not meaningful.")

    return net


def equity_from_returns(rets: np.ndarray, compounding: bool, start_equity: float = 1.0) -> np.ndarray:
    """
    Convert per-trade returns into an equity curve.
    - additive: E_t = E_0 + cumsum(rets)
    - compounding: E_t = E_0 * cumprod(1+rets)
    """
    rets = np.asarray(rets, dtype=float)
    if compounding:
        return start_equity * np.cumprod(1.0 + rets)
    else:
        return start_equity + np.cumsum(rets)


def max_drawdown(equity: np.ndarray) -> float:
    """
    Max drawdown as a fraction (0.25 = -25% peak-to-trough).
    """
    eq = np.asarray(equity, dtype=float)
    peaks = np.maximum.accumulate(eq)
    dd = (eq - peaks) / peaks
    return float(np.min(dd))


def block_bootstrap_sample(x: np.ndarray, rng: np.random.Generator, block_size: int) -> np.ndarray:
    """
    Sample contiguous blocks with replacement until length matches x.
    """
    n = len(x)
    b = max(1, int(block_size))
    out = np.empty(n, dtype=float)

    i = 0
    while i < n:
        start = rng.integers(0, n)
        end = min(start + b, n)
        block = x[start:end]
        take = min(len(block), n - i)
        out[i:i+take] = block[:take]
        i += take
    return out


def sample_returns(net: np.ndarray, rng: np.random.Generator, cfg: MCConfig) -> np.ndarray:
    n = len(net)
    if cfg.mode == "iidr":
        idx = rng.integers(0, n, size=n)
        return net[idx]
    elif cfg.mode == "permute":
        out = net.copy()
        rng.shuffle(out)
        return out
    elif cfg.mode == "block":
        return block_bootstrap_sample(net, rng, cfg.block_size)
    else:
        raise ValueError(f"Unknown mode: {cfg.mode}. Use iidr|permute|block.")


# -----------------------------
# Monte Carlo
# -----------------------------

def run_mc(net: np.ndarray, cfg: MCConfig) -> pd.DataFrame:
    rng = np.random.default_rng(cfg.seed)

    rows = []
    for _ in range(cfg.iters):
        rets = sample_returns(net, rng, cfg)
        eq = equity_from_returns(rets, compounding=cfg.compounding, start_equity=cfg.start_equity)

        total_return = float(eq[-1] - cfg.start_equity) if not cfg.compounding else float(eq[-1] / cfg.start_equity - 1.0)
        mdd = max_drawdown(eq)

        # additional diagnostics
        win_rate = float((rets > 0).mean())
        avg_ret = float(np.mean(rets))
        med_ret = float(np.median(rets))

        rows.append((total_return, mdd, win_rate, avg_ret, med_ret))

    return pd.DataFrame(rows, columns=["total_return", "max_drawdown", "win_rate", "avg_ret", "median_ret"])


def summarize(df: pd.DataFrame) -> str:
    def pct(x): return f"{100.0*x:.2f}%"

    tr = df["total_return"].values
    mdd = df["max_drawdown"].values  # negative numbers
    wr = df["win_rate"].values

    # quantiles
    q = [0.01, 0.05, 0.10, 0.50, 0.90, 0.95, 0.99]
    tr_q = np.quantile(tr, q)
    mdd_q = np.quantile(mdd, q)

    lines = []
    lines.append("\n================ MONTE CARLO SUMMARY ================")
    lines.append(f"iters: {len(df)}")
    lines.append(f"total_return mean:   {tr.mean():.6f}")
    lines.append(f"total_return median: {np.median(tr):.6f}")
    lines.append(f"P(total_return > 0): {(tr > 0).mean():.3f}")
    lines.append("")
    lines.append("total_return quantiles:")
    for qq, vv in zip(q, tr_q):
        lines.append(f"  q{int(qq*100):02d}: {vv:.6f}")
    lines.append("")
    lines.append("max_drawdown quantiles (negative = drawdown):")
    for qq, vv in zip(q, mdd_q):
        lines.append(f"  q{int(qq*100):02d}: {vv:.6f}   ({pct(vv)})")
    lines.append("")
    lines.append(f"win_rate mean: {wr.mean():.3f}")
    lines.append("=====================================================\n")
    return "\n".join(lines)


# -----------------------------
# CLI
# -----------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--trades", type=str, required=True, help="Path to trades CSV (must include net_ret)")
    p.add_argument("--iters", type=int, default=20000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--mode", type=str, default="iidr", choices=["iidr", "permute", "block"])
    p.add_argument("--block_size", type=int, default=5)
    p.add_argument("--compounding", action="store_true", help="Use compounding equity instead of additive sum")
    p.add_argument("--out", type=str, default=None, help="Optional output CSV of MC results")
    p.add_argument("--max_trades", type=int, default=None, help="Optional cap on trades for quick testing")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    trades_path = Path(args.trades)
    if not trades_path.exists():
        raise FileNotFoundError(f"Trades CSV not found: {trades_path}")

    cfg = MCConfig(
        iters=int(args.iters),
        seed=int(args.seed),
        mode=str(args.mode),
        block_size=int(args.block_size),
        compounding=bool(args.compounding),
        max_trades=args.max_trades,
    )

    net = load_net_returns(trades_path, max_trades=cfg.max_trades)

    print(f"TRADES: {trades_path} | n={len(net)}")
    print(f"mode={cfg.mode} iters={cfg.iters} seed={cfg.seed} compounding={cfg.compounding}")
    if cfg.mode == "block":
        print(f"block_size={cfg.block_size}")

    df = run_mc(net, cfg)
    print(summarize(df))

    if args.out:
        out_path = Path(args.out)
        df.to_csv(out_path, index=False)
        print(f"Saved MC samples to: {out_path}")


if __name__ == "__main__":
    main()
