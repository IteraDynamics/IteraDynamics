"""
calendar_equity_sim.py

Calendar-aware equity simulation from a trades CSV produced by offline_walkforward.py.

Assumptions:
- Trades DO NOT OVERLAP (single-position system).
- Each trade row has: entry_ts, exit_ts, net_ret (return on 1.0 notional, already includes costs).
- Position sizing is applied as a fraction of equity at entry: f in [0, max_exposure].
- Between trades, equity is held constant (cash earns 0).

Outputs:
- Equity curve CSV (timestamp, equity, drawdown)
- Summary printed to console and optionally saved to CSV

Example (PowerShell):
  python .\calendar_equity_sim.py `
    --trades ".\wf_runs\20260109_110637\Aprime_h48_c0.64_trades.csv" `
    --max_exposure 0.25 `
    --out_curve ".\wf_runs\20260109_110637\EQ_Aprime_c064_max025.csv" `
    --out_summary ".\wf_runs\20260109_110637\EQ_Aprime_c064_max025_summary.csv"
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class SimConfig:
    max_exposure: float = 0.25
    initial_equity: float = 1.0
    freq: str = "h"  # hourly curve; forward-filled between events


def _max_drawdown(equity: np.ndarray) -> float:
    eq = np.asarray(equity, dtype=float)
    peak = np.maximum.accumulate(eq)
    dd = eq / peak - 1.0
    return float(dd.min())  # negative number


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--trades", type=str, required=True, help="Trades CSV from offline_walkforward.py")
    p.add_argument("--max_exposure", type=float, default=0.25, help="Max fraction of equity allocated per trade")
    p.add_argument("--initial_equity", type=float, default=1.0, help="Starting equity")
    p.add_argument("--freq", type=str, default="H", help="Equity curve frequency (default hourly 'H')")
    p.add_argument("--out_curve", type=str, required=True, help="Output equity curve CSV path")
    p.add_argument("--out_summary", type=str, default=None, help="Optional output summary CSV path")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    trades_path = Path(args.trades)
    if not trades_path.exists():
        raise FileNotFoundError(f"Trades CSV not found: {trades_path}")

    cfg = SimConfig(
        max_exposure=float(args.max_exposure),
        initial_equity=float(args.initial_equity),
        freq=str(args.freq),
    )

    df = pd.read_csv(trades_path)

    # Required columns
    needed = {"entry_ts", "exit_ts", "net_ret"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"Trades CSV missing columns: {missing}. Found: {list(df.columns)}")

    # Parse timestamps
    df["entry_ts"] = pd.to_datetime(df["entry_ts"], errors="coerce")
    df["exit_ts"] = pd.to_datetime(df["exit_ts"], errors="coerce")
    df = df.dropna(subset=["entry_ts", "exit_ts", "net_ret"]).copy()

    if df.empty:
        raise RuntimeError("No valid trades after parsing timestamps/net_ret.")

    # Ensure chronological
    df = df.sort_values(["entry_ts", "exit_ts"]).reset_index(drop=True)

    # Sanity: no overlap check (best-effort)
    # (If there is overlap, this sim will still run, but results would be invalid for sizing.)
    overlaps = (df["entry_ts"].iloc[1:].values < df["exit_ts"].iloc[:-1].values).sum()
    if overlaps > 0:
        print(f"WARNING: Detected {overlaps} potential overlapping trades. Results may be invalid.")

    start_ts = df["entry_ts"].min()
    end_ts = df["exit_ts"].max()
    if pd.isna(start_ts) or pd.isna(end_ts) or end_ts <= start_ts:
        raise RuntimeError("Invalid start/end timestamps in trades.")

    # Build hourly (or chosen) timeline
    idx = pd.date_range(start=start_ts.floor(cfg.freq), end=end_ts.ceil(cfg.freq), freq=cfg.freq)

    # Apply trade returns at exit timestamps (equity is constant between exits)
    equity = cfg.initial_equity
    equity_events = []  # (timestamp, equity_after)

    for _, row in df.iterrows():
        r = float(row["net_ret"])
        # allocate fraction of equity (cap at max_exposure)
        f = float(np.clip(cfg.max_exposure, 0.0, 1.0))
        equity = equity * (1.0 + f * r)
        equity_events.append((row["exit_ts"], equity))

    events = pd.DataFrame(equity_events, columns=["ts", "equity"]).sort_values("ts")
    events = events.drop_duplicates(subset=["ts"], keep="last").set_index("ts")

    curve = pd.DataFrame(index=idx)
    curve["equity"] = np.nan
    curve.loc[events.index.intersection(curve.index), "equity"] = events["equity"]

    # Set initial equity at start, then forward fill
    curve.iloc[0, curve.columns.get_loc("equity")] = cfg.initial_equity
    curve["equity"] = curve["equity"].ffill()

    # Drawdown series
    curve["peak"] = curve["equity"].cummax()
    curve["drawdown"] = curve["equity"] / curve["peak"] - 1.0

    # Calendar-aware stats
    years = (curve.index[-1] - curve.index[0]).total_seconds() / (365.25 * 86400.0)
    ending_equity = float(curve["equity"].iloc[-1])
    total_return = ending_equity / cfg.initial_equity - 1.0
    cagr = (ending_equity / cfg.initial_equity) ** (1.0 / years) - 1.0 if years > 0 else np.nan
    mdd = float(curve["drawdown"].min())

    # Daily stats (resample to daily close)
    daily = curve["equity"].resample("D").last().dropna()
    daily_rets = daily.pct_change().dropna()

    if len(daily_rets) >= 2:
        vol = float(daily_rets.std(ddof=0) * np.sqrt(365.25))
        mean = float(daily_rets.mean() * 365.25)
        sharpe = float(mean / vol) if vol > 0 else np.nan
    else:
        vol, sharpe = np.nan, np.nan

    # Output curve
    out_curve = Path(args.out_curve)
    out_curve.parent.mkdir(parents=True, exist_ok=True)
    curve_out = curve.reset_index().rename(columns={"index": "timestamp"})
    curve_out[["timestamp", "equity", "drawdown"]].to_csv(out_curve, index=False)

    # Summary
    summary = {
        "trades_file": str(trades_path),
        "max_exposure": cfg.max_exposure,
        "initial_equity": cfg.initial_equity,
        "start": str(curve.index[0]),
        "end": str(curve.index[-1]),
        "years": float(years),
        "ending_equity": ending_equity,
        "total_return": float(total_return),
        "cagr": float(cagr),
        "max_drawdown": float(mdd),
        "ann_vol_from_daily": float(vol) if np.isfinite(vol) else None,
        "ann_sharpe_from_daily": float(sharpe) if np.isfinite(sharpe) else None,
        "num_trades": int(len(df)),
    }

    print("\n================ CALENDAR EQUITY SIM SUMMARY ================")
    print(f"TRADES:        {trades_path} | n={summary['num_trades']}")
    print(f"max_exposure:  {summary['max_exposure']}")
    print(f"start:         {summary['start']}")
    print(f"end:           {summary['end']}")
    print(f"years:         {summary['years']:.4f}")
    print(f"ending_equity: {summary['ending_equity']:.6f}")
    print(f"total_return:  {summary['total_return']:.4f}")
    print(f"CAGR:          {summary['cagr']:.4f}")
    print(f"max_drawdown:  {summary['max_drawdown']:.4f}")
    print(f"ann_vol:       {summary['ann_vol_from_daily']}")
    print(f"ann_sharpe:    {summary['ann_sharpe_from_daily']}")
    print(f"Saved curve -> {out_curve}")
    print("=============================================================\n")

    if args.out_summary:
        out_summary = Path(args.out_summary)
        out_summary.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([summary]).to_csv(out_summary, index=False)
        print(f"Saved summary -> {out_summary}")


if __name__ == "__main__":
    main()
