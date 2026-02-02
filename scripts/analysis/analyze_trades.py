#!/usr/bin/env python3
"""
Offline trade analysis for Argus / Itera.

Supports two input paths:

A) Round-trips CSV (recommended to start):
   A CSV where each row is a completed trade with entry/exit.
   Required columns:
     entry_ts, exit_ts, side, entry_price, exit_price
   Optional:
     qty, regime_entry, severity_entry, exit_reason

B) Fills file (more realistic, if you can export):
   - CSV with columns: ts, side, price, qty
   - JSON list of objects with keys: ts|timestamp|created_at, side, price, qty|size
   Script will pair fills FIFO into round-trips (simple long-only or alternating buy/sell).
   (If your execution is more complex, we’ll refine the pairing logic later.)

Outputs:
  artifacts/trades.parquet
  artifacts/summary.json
  artifacts/equity_curve.png
  artifacts/r_hist.png
  artifacts/hold_vs_r.png

Usage examples:
  python analyze_trades.py --trades-csv my_trades.csv
  python analyze_trades.py --fills-csv fills.csv
  python analyze_trades.py --fills-json fills.json --stop-loss-pct 0.02
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


ARTIFACTS = Path("artifacts")


def _dt(x) -> pd.Timestamp:
    # tolerant timestamp parsing
    t = pd.to_datetime(x, utc=True, errors="coerce")
    if pd.isna(t):
        raise ValueError(f"Could not parse timestamp: {x!r}")
    return t


def _ensure_artifacts_dir() -> None:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)


def load_roundtrips_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    need = {"entry_ts", "exit_ts", "side", "entry_price", "exit_price"}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"round-trips CSV missing columns: {sorted(missing)}")

    df = df.copy()
    df["entry_ts"] = pd.to_datetime(df["entry_ts"], utc=True, errors="coerce")
    df["exit_ts"] = pd.to_datetime(df["exit_ts"], utc=True, errors="coerce")
    if df["entry_ts"].isna().any() or df["exit_ts"].isna().any():
        bad = df[df["entry_ts"].isna() | df["exit_ts"].isna()].head(5)
        raise ValueError(f"Bad timestamps in round-trips CSV. Examples:\n{bad}")

    df["side"] = df["side"].astype(str).str.upper().str.strip()
    if not set(df["side"]).issubset({"LONG", "SHORT"}):
        raise ValueError("side must be LONG or SHORT in round-trips CSV.")

    for c in ["entry_price", "exit_price"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    if df[["entry_price", "exit_price"]].isna().any().any():
        bad = df[df["entry_price"].isna() | df["exit_price"].isna()].head(5)
        raise ValueError(f"Bad prices in round-trips CSV. Examples:\n{bad}")

    if "qty" in df.columns:
        df["qty"] = pd.to_numeric(df["qty"], errors="coerce")
    else:
        df["qty"] = np.nan

    # optional metadata columns
    for col in ["regime_entry", "severity_entry", "exit_reason"]:
        if col not in df.columns:
            df[col] = None

    return df


def load_fills_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    need = {"ts", "side", "price", "qty"}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"fills CSV missing columns: {sorted(missing)}. Need: {sorted(need)}")

    df = df.copy()
    df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")
    if df["ts"].isna().any():
        bad = df[df["ts"].isna()].head(5)
        raise ValueError(f"Bad timestamps in fills CSV. Examples:\n{bad}")

    df["side"] = df["side"].astype(str).str.upper().str.strip()
    # accept BUY/SELL (preferred)
    if not set(df["side"]).issubset({"BUY", "SELL"}):
        raise ValueError("fills CSV side must be BUY or SELL.")

    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df["qty"] = pd.to_numeric(df["qty"], errors="coerce")
    if df[["price", "qty"]].isna().any().any():
        bad = df[df["price"].isna() | df["qty"].isna()].head(5)
        raise ValueError(f"Bad price/qty in fills CSV. Examples:\n{bad}")

    return df.sort_values("ts").reset_index(drop=True)


def load_fills_json(path: Path) -> pd.DataFrame:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("fills JSON must be a list of fill objects.")

    rows = []
    for r in raw:
        if not isinstance(r, dict):
            continue
        ts = r.get("ts") or r.get("timestamp") or r.get("created_at")
        side = r.get("side")
        price = r.get("price")
        qty = r.get("qty") or r.get("size")
        if ts is None or side is None or price is None or qty is None:
            continue
        rows.append({"ts": ts, "side": side, "price": price, "qty": qty})

    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError("No usable fill rows found in JSON (need ts/side/price/qty).")

    df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")
    df["side"] = df["side"].astype(str).str.upper().str.strip()
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df["qty"] = pd.to_numeric(df["qty"], errors="coerce")

    df = df.dropna(subset=["ts", "side", "price", "qty"])
    if not set(df["side"]).issubset({"BUY", "SELL"}):
        raise ValueError("fills JSON side must be BUY/SELL after normalization.")

    return df.sort_values("ts").reset_index(drop=True)


def pair_fills_fifo(fills: pd.DataFrame) -> pd.DataFrame:
    """
    Very simple FIFO pairing for a single-asset strategy:
      - accumulate BUY lots
      - SELL closes lots FIFO
    Produces LONG round-trips only.
    (If you do shorts or partials heavily, we’ll enhance later.)
    """
    lots: List[Dict[str, Any]] = []
    trades: List[Dict[str, Any]] = []

    for _, row in fills.iterrows():
        side = row["side"]
        ts = row["ts"]
        price = float(row["price"])
        qty = float(row["qty"])

        if side == "BUY":
            lots.append({"entry_ts": ts, "entry_price": price, "qty": qty})
            continue

        # SELL: close FIFO
        remaining = qty
        while remaining > 1e-12 and lots:
            lot = lots[0]
            take = min(remaining, lot["qty"])
            trades.append(
                {
                    "entry_ts": lot["entry_ts"],
                    "exit_ts": ts,
                    "side": "LONG",
                    "entry_price": float(lot["entry_price"]),
                    "exit_price": price,
                    "qty": float(take),
                    "regime_entry": None,
                    "severity_entry": None,
                    "exit_reason": None,
                }
            )
            lot["qty"] -= take
            remaining -= take
            if lot["qty"] <= 1e-12:
                lots.pop(0)

        # If remaining > 0 and no lots, it means you sold more than you had (short or data mismatch).
        # We’ll ignore remainder for now but flag it later.
    return pd.DataFrame(trades)


def compute_trade_metrics(trades: pd.DataFrame, stop_loss_pct: float) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    df = trades.copy()

    # derived fields
    df["hold_hours"] = (df["exit_ts"] - df["entry_ts"]).dt.total_seconds() / 3600.0

    # PnL per unit
    is_long = df["side"].eq("LONG")
    is_short = df["side"].eq("SHORT")

    df["pnl_per_unit"] = np.where(
        is_long, df["exit_price"] - df["entry_price"],
        np.where(is_short, df["entry_price"] - df["exit_price"], np.nan)
    )

    # percent return per unit
    df["ret_pct"] = np.where(
        is_long, (df["exit_price"] / df["entry_price"] - 1.0),
        np.where(is_short, (df["entry_price"] / df["exit_price"] - 1.0), np.nan)
    )

    # dollar pnl if qty present
    df["pnl_usd"] = df["pnl_per_unit"] * df["qty"]

    # R-multiple: define entry risk = entry_price * stop_loss_pct (per unit)
    # For SHORT, still use entry_price * stop_loss_pct as unit risk
    unit_risk = df["entry_price"] * float(stop_loss_pct)
    unit_risk = unit_risk.replace(0, np.nan)
    df["R"] = df["pnl_per_unit"] / unit_risk

    # equity curve from per-trade USD pnl (if qty missing, use per_unit as proxy)
    pnl_series = df["pnl_usd"]
    if pnl_series.isna().all():
        pnl_series = df["pnl_per_unit"]

    df = df.sort_values("exit_ts").reset_index(drop=True)
    df["equity"] = pnl_series.fillna(0.0).cumsum()

    # summary stats (use R as primary)
    R = df["R"].replace([np.inf, -np.inf], np.nan).dropna()
    wins = R[R > 0]
    losses = R[R < 0]

    expectancy = float(R.mean()) if len(R) else float("nan")
    win_rate = float((R > 0).mean()) if len(R) else float("nan")

    profit_factor = float(wins.sum() / abs(losses.sum())) if len(losses) and len(wins) else (
        float("inf") if len(wins) and not len(losses) else float("nan")
    )

    # max drawdown on equity
    eq = df["equity"].astype(float).values
    peak = np.maximum.accumulate(eq) if len(eq) else np.array([])
    dd = (eq - peak) if len(eq) else np.array([])
    max_dd = float(dd.min()) if len(dd) else 0.0

    summary = {
        "n_trades": int(len(df)),
        "stop_loss_pct": float(stop_loss_pct),
        "expectancy_R": expectancy,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "R_mean": float(R.mean()) if len(R) else None,
        "R_median": float(R.median()) if len(R) else None,
        "R_std": float(R.std(ddof=1)) if len(R) > 1 else None,
        "R_p05": float(R.quantile(0.05)) if len(R) else None,
        "R_p95": float(R.quantile(0.95)) if len(R) else None,
        "max_drawdown_equity_units": max_dd,
        "avg_hold_hours": float(df["hold_hours"].mean()) if len(df) else None,
    }
    return df, summary


def plot_outputs(df: pd.DataFrame) -> None:
    _ensure_artifacts_dir()

    # R histogram
    plt.figure()
    r = df["R"].replace([np.inf, -np.inf], np.nan).dropna()
    if len(r):
        plt.hist(r.values, bins=40)
    plt.title("R-multiple Histogram")
    plt.xlabel("R")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(ARTIFACTS / "r_hist.png")
    plt.close()

    # equity curve
    plt.figure()
    plt.plot(df["exit_ts"].values, df["equity"].values)
    plt.title("Equity Curve (units)")
    plt.xlabel("Exit time")
    plt.ylabel("Cumulative PnL (USD if qty provided, else per-unit)")
    plt.tight_layout()
    plt.savefig(ARTIFACTS / "equity_curve.png")
    plt.close()

    # hold vs R scatter
    plt.figure()
    hh = df["hold_hours"].astype(float)
    rr = df["R"].astype(float)
    m = np.isfinite(hh.values) & np.isfinite(rr.values)
    if m.any():
        plt.scatter(hh.values[m], rr.values[m], s=10)
    plt.title("Hold Duration vs R")
    plt.xlabel("Hold hours")
    plt.ylabel("R")
    plt.tight_layout()
    plt.savefig(ARTIFACTS / "hold_vs_r.png")
    plt.close()


def bucket_slice(df: pd.DataFrame, col: str, bins: List[float]) -> Dict[str, Any]:
    x = pd.to_numeric(df[col], errors="coerce")
    r = pd.to_numeric(df["R"], errors="coerce")
    out: Dict[str, Any] = {"col": col, "bins": bins, "buckets": []}
    if x.isna().all():
        return out

    cats = pd.cut(x, bins=bins, include_lowest=True)
    for k, g in df.groupby(cats, dropna=True):
        rr = pd.to_numeric(g["R"], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        out["buckets"].append(
            {
                "bucket": str(k),
                "n": int(len(rr)),
                "expectancy_R": float(rr.mean()) if len(rr) else None,
                "win_rate": float((rr > 0).mean()) if len(rr) else None,
                "R_median": float(rr.median()) if len(rr) else None,
            }
        )
    return out


def main():
    global ARTIFACTS
    
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--trades-csv", type=Path, help="Round-trips CSV with entry/exit per trade")
    g.add_argument("--fills-csv", type=Path, help="Fills CSV with columns: ts,side,price,qty")
    g.add_argument("--fills-json", type=Path, help="Fills JSON list with ts/timestamp/created_at,side,price,qty/size")

    ap.add_argument("--stop-loss-pct", type=float, default=float(os.getenv("ARGUS_STOP_LOSS_PCT", "0.02")))
    ap.add_argument("--out", type=Path, default=ARTIFACTS)

    args = ap.parse_args()
    ARTIFACTS = args.out

    _ensure_artifacts_dir()

    if args.trades_csv:
        trades = load_roundtrips_csv(args.trades_csv)
    elif args.fills_csv:
        fills = load_fills_csv(args.fills_csv)
        trades = pair_fills_fifo(fills)
        if trades.empty:
            raise RuntimeError("No trades formed from fills (did you include both BUY and SELL fills?).")
    else:
        fills = load_fills_json(args.fills_json)
        trades = pair_fills_fifo(fills)
        if trades.empty:
            raise RuntimeError("No trades formed from fills JSON (did you include both BUY and SELL fills?).")

    trades, summary = compute_trade_metrics(trades, stop_loss_pct=args.stop_loss_pct)

    # Write artifacts
    trades_path = ARTIFACTS / "trades.parquet"
    trades.to_parquet(trades_path, index=False)

    # Add a couple practical slices if we have fields
    slices: Dict[str, Any] = {}
    if "severity_entry" in trades.columns:
        slices["severity_buckets"] = bucket_slice(trades, "severity_entry", bins=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    if "hold_hours" in trades.columns:
        slices["hold_buckets"] = bucket_slice(trades, "hold_hours", bins=[0, 1, 2, 4, 8, 16, 24, 48, 96, 1e9])

    out = {"summary": summary, "slices": slices}
    (ARTIFACTS / "summary.json").write_text(json.dumps(out, indent=2), encoding="utf-8")

    plot_outputs(trades)

    # Print the 5 numbers that decide everything
    print("\n=== Itera Offline Summary ===")
    print("trades:", summary["n_trades"])
    print("expectancy_R:", summary["expectancy_R"])
    print("win_rate:", summary["win_rate"])
    print("profit_factor:", summary["profit_factor"])
    print("max_drawdown_equity_units:", summary["max_drawdown_equity_units"])
    print("\nWrote:")
    print(" -", trades_path)
    print(" -", ARTIFACTS / "summary.json")
    print(" -", ARTIFACTS / "equity_curve.png")
    print(" -", ARTIFACTS / "r_hist.png")
    print(" -", ARTIFACTS / "hold_vs_r.png")


if __name__ == "__main__":
    main()
