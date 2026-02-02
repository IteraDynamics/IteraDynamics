"""
position_sizing_experiments.py

Run position sizing experiments on a trades CSV produced by offline_walkforward.py
(e.g. A_h48_c070_trades.csv).

Assumptions:
- trades CSV has at least: entry_ts, exit_ts, net_ret
- net_ret is per-trade return for 1.0 notional (already includes costs)
- Single-position system => one trade at a time (already enforced upstream)
- We model equity compounding: equity *= (1 + exposure * net_ret)
- exposure is capped by --max_exposure

Outputs:
- A results CSV with metrics per sizing configuration
- Optional per-trade equity curve CSV per configuration

Examples:
  python .\position_sizing_experiments.py --trades ".\wf_runs\20260109_082547\A_h48_c070_trades.csv" --out ".\wf_runs\20260109_082547\SIZING_A_results.csv"

PowerShell tip (no line breaks):
  python .\position_sizing_experiments.py --trades ".\wf_runs\20260109_082547\A_h48_c070_trades.csv" --out ".\wf_runs\20260109_082547\SIZING_A_results.csv" --curves_dir ".\wf_runs\20260109_082547\curves"
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd


# -----------------------------
# Metrics helpers
# -----------------------------

def max_drawdown(equity: np.ndarray) -> float:
    """Return max drawdown as a negative number (e.g. -0.25 = -25%)."""
    eq = np.asarray(equity, dtype=float)
    if eq.size < 2:
        return 0.0
    peak = np.maximum.accumulate(eq)
    dd = (eq / peak) - 1.0
    return float(dd.min())

def cagr_from_years(e0: float, e1: float, years: float) -> float:
    if years <= 0 or e0 <= 0:
        return 0.0
    return float((e1 / e0) ** (1.0 / years) - 1.0)

def sharpe_per_trade(trade_returns: np.ndarray) -> float:
    r = np.asarray(trade_returns, dtype=float)
    r = r[np.isfinite(r)]
    if r.size < 2:
        return 0.0
    mu = r.mean()
    sd = r.std(ddof=1)
    if sd == 0:
        return 0.0
    return float(mu / sd)

def profit_factor(trade_returns: np.ndarray) -> float:
    r = np.asarray(trade_returns, dtype=float)
    wins = r[r > 0].sum()
    losses = -r[r < 0].sum()
    if losses == 0:
        return float("inf") if wins > 0 else 0.0
    return float(wins / losses)


# -----------------------------
# Sizing policies
# -----------------------------

@dataclass
class SizingConfig:
    policy: str                        # fixed | kelly | frac_kelly | dd_throttle
    f: float = 0.25                    # base fraction for fixed OR fraction of Kelly
    dd_limit: float = 0.10             # throttle if drawdown worse than this (e.g. 0.10 = 10%)
    throttle_mult: float = 0.5         # exposure multiplier when throttled
    max_exposure: float = 1.0          # cap exposure (1.0 = 100% of equity per trade)


def compute_kelly_fraction(net_rets: np.ndarray) -> float:
    """
    Approximate Kelly f* for returns (not binary bets):
      f* = mean(r) / var(r)
    where r is per-trade return at 1.0 exposure.

    This is a rough approximation; we’ll use fractional Kelly in practice.
    """
    r = np.asarray(net_rets, dtype=float)
    r = r[np.isfinite(r)]
    if r.size < 3:
        return 0.0
    mu = r.mean()
    var = r.var(ddof=1)
    if var <= 0:
        return 0.0
    return float(mu / var)


def policy_exposure(
    policy: str,
    base_f: float,
    kelly_f: float,
    cur_dd: float,
    dd_limit: float,
    throttle_mult: float,
) -> float:
    """
    Return exposure fraction of equity for next trade.
    cur_dd is negative number (e.g. -0.08).
    dd_limit is positive (e.g. 0.10).
    """
    if policy == "fixed":
        return base_f

    if policy == "kelly":
        return kelly_f

    if policy == "frac_kelly":
        return base_f * kelly_f

    if policy == "dd_throttle":
        # base exposure is base_f; if drawdown worse than -dd_limit, reduce exposure
        if cur_dd <= -abs(dd_limit):
            return base_f * throttle_mult
        return base_f

    raise ValueError(f"Unknown policy: {policy}")


# -----------------------------
# Simulation
# -----------------------------

def simulate_equity(
    df_trades: pd.DataFrame,
    cfg: SizingConfig,
    years: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns (equity_curve, applied_exposures).
    equity starts at 1.0
    """
    r = df_trades["net_ret"].astype(float).to_numpy()
    n = len(r)
    equity = np.ones(n + 1, dtype=float)
    exposures = np.zeros(n, dtype=float)

    # Kelly computed once from the trade series (we can upgrade this later to rolling Kelly)
    kelly_f = compute_kelly_fraction(r)

    for i in range(n):
        cur_dd = max_drawdown(equity[: i + 1])
        exp = policy_exposure(
            policy=cfg.policy,
            base_f=cfg.f,
            kelly_f=kelly_f,
            cur_dd=cur_dd,
            dd_limit=cfg.dd_limit,
            throttle_mult=cfg.throttle_mult,
        )

        exp = float(np.clip(exp, 0.0, cfg.max_exposure))
        exposures[i] = exp

        # equity update (compounding)
        equity[i + 1] = equity[i] * (1.0 + exp * r[i])

        # guard: if equity hits zero or negative, stop
        if equity[i + 1] <= 0:
            equity[i + 1 :] = equity[i + 1]
            break

    return equity, exposures


def estimate_years_from_trades(df_trades: pd.DataFrame) -> float:
    """
    Estimate total years spanned by the trades based on entry_ts/exist_ts.
    If timestamps missing, fallback to trades count (assume 48h per trade).
    """
    for col in ["entry_ts", "exit_ts"]:
        if col in df_trades.columns:
            df_trades[col] = pd.to_datetime(df_trades[col], errors="coerce")
    if "entry_ts" in df_trades.columns and "exit_ts" in df_trades.columns:
        t0 = df_trades["entry_ts"].min()
        t1 = df_trades["exit_ts"].max()
        if pd.notna(t0) and pd.notna(t1) and t1 > t0:
            days = (t1 - t0).total_seconds() / 86400.0
            return float(days / 365.25)

    # fallback: assume 48h per trade
    return float(len(df_trades) * 2.0 / 365.25)


def run_experiments(
    trades_df: pd.DataFrame,
    max_exposure: float,
    curves_dir: Optional[Path],
) -> pd.DataFrame:
    # years estimate for CAGR
    years = estimate_years_from_trades(trades_df)

    # ---- define experiment grid ----
    fixed_fs = [0.10, 0.25, 0.50, 0.75, 1.00]
    frac_kelly_mults = [0.10, 0.25, 0.50]      # 10% Kelly, 25% Kelly, 50% Kelly
    dd_limits = [0.10, 0.15, 0.20]             # throttle at 10/15/20% drawdown
    throttle_mults = [0.25, 0.50]              # when throttled, cut to 25% or 50% of base

    rows: List[Dict] = []
    r = trades_df["net_ret"].astype(float).to_numpy()
    base_pf = profit_factor(r)
    base_wr = float((r > 0).mean()) if len(r) else 0.0

    # Kelly baseline (computed on full series)
    kelly_f = compute_kelly_fraction(r)

    def record(name: str, cfg: SizingConfig):
        eq, exp = simulate_equity(trades_df, cfg, years=years)
        # realized per-trade returns after sizing (for PF/Sharpe per trade)
        sized = trades_df["net_ret"].astype(float).to_numpy() * exp
        out = {
            "name": name,
            "policy": cfg.policy,
            "base_f": cfg.f,
            "dd_limit": cfg.dd_limit,
            "throttle_mult": cfg.throttle_mult,
            "max_exposure": cfg.max_exposure,
            "trades": int(len(trades_df)),
            "years_est": years,
            "kelly_f_est": kelly_f,
            "ending_equity": float(eq[-1]),
            "total_return": float(eq[-1] - 1.0),
            "cagr": cagr_from_years(1.0, float(eq[-1]), years),
            "max_drawdown": max_drawdown(eq),
            "win_rate_base": base_wr,
            "profit_factor_base": base_pf,
            "profit_factor_sized": profit_factor(sized),
            "sharpe_per_trade_sized": sharpe_per_trade(sized),
        }
        rows.append(out)

        if curves_dir is not None:
            curves_dir.mkdir(parents=True, exist_ok=True)
            curve_df = trades_df.copy()
            curve_df["exposure"] = exp
            curve_df["sized_ret"] = curve_df["net_ret"].astype(float) * curve_df["exposure"]
            curve_df["equity"] = eq[1:]
            curve_df.to_csv(curves_dir / f"{name}.curve.csv", index=False)

    # Fixed fraction sizing
    for f in fixed_fs:
        cfg = SizingConfig(policy="fixed", f=f, max_exposure=max_exposure)
        record(f"fixed_f{f:.2f}", cfg)

    # Full Kelly (for reference only)
    cfg = SizingConfig(policy="kelly", f=1.0, max_exposure=max_exposure)
    record("kelly_full", cfg)

    # Fractional Kelly
    for m in frac_kelly_mults:
        cfg = SizingConfig(policy="frac_kelly", f=m, max_exposure=max_exposure)
        record(f"frac_kelly_{m:.2f}", cfg)

    # Drawdown throttle (fixed base f=0.50; adjust thresholds)
    base_f = 0.50
    for ddl in dd_limits:
        for tm in throttle_mults:
            cfg = SizingConfig(
                policy="dd_throttle",
                f=base_f,
                dd_limit=ddl,
                throttle_mult=tm,
                max_exposure=max_exposure,
            )
            record(f"dd_throttle_f{base_f:.2f}_dd{ddl:.2f}_m{tm:.2f}", cfg)

    return pd.DataFrame(rows).sort_values(["cagr", "total_return"], ascending=False).reset_index(drop=True)


# -----------------------------
# CLI
# -----------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--trades", type=str, required=True, help="Path to trades CSV (A_h48_c070_trades.csv)")
    p.add_argument("--out", type=str, required=True, help="Output results CSV")
    p.add_argument("--curves_dir", type=str, default=None, help="Optional dir to write per-config equity curves")
    p.add_argument("--max_exposure", type=float, default=1.0, help="Cap exposure per trade (default 1.0 = 100%)")
    return p.parse_args()


def main():
    args = parse_args()
    trades_path = Path(args.trades)
    out_path = Path(args.out)
    curves_dir = Path(args.curves_dir) if args.curves_dir else None

    if not trades_path.exists():
        raise FileNotFoundError(f"Trades CSV not found: {trades_path}")

    df = pd.read_csv(trades_path)

    if "net_ret" not in df.columns:
        raise ValueError(f"Trades CSV missing required column: net_ret | cols={list(df.columns)}")

    # Sort chronologically if timestamps exist
    if "entry_ts" in df.columns:
        df["entry_ts"] = pd.to_datetime(df["entry_ts"], errors="coerce")
        df = df.sort_values("entry_ts").reset_index(drop=True)

    results = run_experiments(df, max_exposure=float(args.max_exposure), curves_dir=curves_dir)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(out_path, index=False)

    # Print top 10
    print("\n================ POSITION SIZING RESULTS (TOP 10) ================")
    show = results.head(10)
    cols = ["name", "policy", "base_f", "cagr", "total_return", "max_drawdown", "ending_equity", "profit_factor_sized"]
    print(show[cols].to_string(index=False))
    print("==================================================================\n")
    print(f"Saved results to: {out_path}")
    if curves_dir is not None:
        print(f"Saved curves to:  {curves_dir}")


if __name__ == "__main__":
    main()
