"""
research/run_regime_conditioned_eval.py

Regime-conditioned evaluation for strategy trades.

Inputs (defaults):
- Trades CSV:  research/backtests/results/regime_trend_v1_trades.csv
- Regimes CSV: output/regime_classifier_v1_labeled.csv  (per-bar labels)

Trades file expected (auto-detected):
- entry_ts / exit_ts (epoch sec/ms OR ISO)
- pnl_pct (preferred) OR pnl_usd + notional to derive %

Regimes file expected (auto-detected):
- Timestamp (epoch sec/ms OR ISO)
- regime_label

Outputs:
- output/regime_conditioned_trades_<entry|exit>.csv
- output/regime_conditioned_report_<entry|exit>.csv
- output/regime_conditioned_summary_<entry|exit>.json

Usage:
  python -m research.run_regime_conditioned_eval
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


DEFAULT_TRADES_PATH = os.path.join("research", "backtests", "results", "regime_trend_v1_trades.csv")
DEFAULT_REGIMES_PATH = os.path.join("output", "regime_classifier_v1_labeled.csv")
DEFAULT_OUT_DIR = os.path.join("output")


# ---------------------------
# Utilities
# ---------------------------

def _ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def _find_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    """
    Return first matching candidate column name. Tries:
      1) exact match
      2) case-insensitive match
    """
    cols = list(df.columns)
    for c in candidates:
        if c in cols:
            return c

    lower_map = {c.lower(): c for c in cols}
    for c in candidates:
        hit = lower_map.get(c.lower())
        if hit is not None:
            return hit

    return None


def _coerce_dt(s: pd.Series) -> pd.Series:
    """
    Coerce series to timezone-aware UTC timestamps.
    Supports:
      - datetime dtype
      - numeric epoch seconds/ms
      - ISO strings
    """
    if pd.api.types.is_datetime64_any_dtype(s):
        return pd.to_datetime(s, utc=True, errors="coerce")

    if pd.api.types.is_numeric_dtype(s):
        x = pd.to_numeric(s, errors="coerce")
        # Heuristic: >1e12 => ms, else seconds
        maxv = float(np.nanmax(x.values)) if np.isfinite(np.nanmax(x.values)) else float("nan")
        unit = "ms" if (np.isfinite(maxv) and maxv > 1e12) else "s"
        return pd.to_datetime(x, unit=unit, utc=True, errors="coerce")

    # object/strings
    return pd.to_datetime(s.astype(str), utc=True, errors="coerce")


def _compound_from_pct_returns(pcts: np.ndarray) -> float:
    pcts = np.array(pcts, dtype=float)
    pcts = pcts[~np.isnan(pcts)]
    if pcts.size == 0:
        return float("nan")
    gross = np.prod(1.0 + (pcts / 100.0))
    return (gross - 1.0) * 100.0


# ---------------------------
# Trades
# ---------------------------

def load_trades(path: str, entry_col_override: Optional[str] = None, exit_col_override: Optional[str] = None) -> pd.DataFrame:
    df = pd.read_csv(path)

    # include YOUR actual schema first
    entry_candidates = [
        "entry_ts", "entry_time", "entry_timestamp", "entry_dt",
        "open_ts", "open_time", "open_timestamp", "time_in", "ts_in",
        "Timestamp", "timestamp", "time", "datetime", "date",
    ]
    exit_candidates = [
        "exit_ts", "exit_time", "exit_timestamp", "exit_dt",
        "close_ts", "close_time", "close_timestamp", "time_out", "ts_out",
    ]

    entry_col = entry_col_override if entry_col_override else _find_col(df, entry_candidates)
    exit_col = exit_col_override if exit_col_override else _find_col(df, exit_candidates)

    if entry_col_override and entry_col_override not in df.columns:
        raise ValueError(f"--entry-col '{entry_col_override}' not found. Columns: {list(df.columns)}")
    if exit_col_override and exit_col_override not in df.columns:
        raise ValueError(f"--exit-col '{exit_col_override}' not found. Columns: {list(df.columns)}")

    if not entry_col:
        raise ValueError(
            f"Could not find an entry timestamp column in trades file: {path}\n"
            f"Columns: {list(df.columns)}\n"
            f"Try: python -m research.run_regime_conditioned_eval --entry-col entry_ts"
        )

    # Return column detection (your schema uses pnl_pct)
    ret_col = _find_col(df, [
        "pnl_pct", "return_pct", "trade_return_pct", "pct_return", "ret_pct",
        "pnl_percent", "return_percent",
    ])
    pnl_usd_col = _find_col(df, ["pnl_usd", "pnl", "profit_usd", "profit"])
    # your file uses entry_notional_net; keep broad
    notional_col = _find_col(df, ["entry_notional_net", "entry_notional_usd", "notional_usd", "entry_usd"])

    out = df.copy()
    out["entry_time"] = _coerce_dt(out[entry_col])

    if exit_col:
        out["exit_time"] = _coerce_dt(out[exit_col])
    else:
        out["exit_time"] = pd.NaT

    if ret_col:
        out["trade_return_pct"] = pd.to_numeric(out[ret_col], errors="coerce")
    elif pnl_usd_col and notional_col:
        pnl = pd.to_numeric(out[pnl_usd_col], errors="coerce")
        notional = pd.to_numeric(out[notional_col], errors="coerce")
        out["trade_return_pct"] = (pnl / notional) * 100.0
    else:
        out["trade_return_pct"] = np.nan

    out["is_win"] = out["trade_return_pct"] > 0

    out = out.dropna(subset=["entry_time"]).sort_values("entry_time").reset_index(drop=True)
    return out


# ---------------------------
# Regimes
# ---------------------------

def load_regimes(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    ts_col = _find_col(df, ["Timestamp", "timestamp", "time", "datetime", "date"])
    reg_col = _find_col(df, ["regime_label", "regime", "label", "state"])

    if not ts_col:
        raise ValueError(f"Could not find timestamp column in regimes file: {path}. Columns: {list(df.columns)}")
    if not reg_col:
        raise ValueError(f"Could not find regime label column in regimes file: {path}. Columns: {list(df.columns)}")

    out = df.copy()
    out["Timestamp"] = _coerce_dt(out[ts_col])
    out["regime_label"] = out[reg_col].astype(str)

    out = out.dropna(subset=["Timestamp"]).sort_values("Timestamp").reset_index(drop=True)
    return out[["Timestamp", "regime_label"]]


# ---------------------------
# Join
# ---------------------------

def assign_trade_regimes(trades: pd.DataFrame, regimes: pd.DataFrame, label_at: str) -> pd.DataFrame:
    if label_at not in ("entry", "exit"):
        raise ValueError("--label-at must be 'entry' or 'exit'")

    tcol = "entry_time" if label_at == "entry" else "exit_time"
    df_tr = trades.copy()

    if label_at == "exit":
        df_tr = df_tr.dropna(subset=[tcol]).copy()

    df_tr = df_tr.sort_values(tcol).reset_index(drop=True)
    reg = regimes.sort_values("Timestamp").reset_index(drop=True)

    joined = pd.merge_asof(
        df_tr,
        reg,
        left_on=tcol,
        right_on="Timestamp",
        direction="backward",
        allow_exact_matches=True,
    )

    joined = joined.rename(columns={"regime_label": f"{label_at}_regime"})
    joined = joined.drop(columns=["Timestamp"], errors="ignore")
    return joined


# ---------------------------
# Reporting
# ---------------------------

@dataclass
class RegimeReportRow:
    regime: str
    n_trades: int
    win_rate_pct: float
    avg_trade_pct: float
    median_trade_pct: float
    compounded_return_pct: float
    best_trade_pct: float
    worst_trade_pct: float


def summarize_by_regime(df: pd.DataFrame, regime_col: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    d = df.dropna(subset=[regime_col]).copy()

    rows: List[RegimeReportRow] = []
    for reg, g in d.groupby(regime_col):
        r = g["trade_return_pct"].astype(float).values
        wins = g["is_win"].astype(bool).values
        rows.append(
            RegimeReportRow(
                regime=str(reg),
                n_trades=int(len(g)),
                win_rate_pct=float(np.mean(wins) * 100.0) if len(g) else float("nan"),
                avg_trade_pct=float(np.nanmean(r)) if len(g) else float("nan"),
                median_trade_pct=float(np.nanmedian(r)) if len(g) else float("nan"),
                compounded_return_pct=float(_compound_from_pct_returns(r)),
                best_trade_pct=float(np.nanmax(r)) if len(g) else float("nan"),
                worst_trade_pct=float(np.nanmin(r)) if len(g) else float("nan"),
            )
        )

    report = pd.DataFrame([asdict(x) for x in rows]).sort_values("n_trades", ascending=False).reset_index(drop=True)

    overall = {
        "n_trades_total": int(len(d)),
        "win_rate_pct_total": float(d["is_win"].mean() * 100.0) if len(d) else float("nan"),
        "avg_trade_pct_total": float(d["trade_return_pct"].mean()) if len(d) else float("nan"),
        "median_trade_pct_total": float(d["trade_return_pct"].median()) if len(d) else float("nan"),
        "compounded_return_pct_total": float(_compound_from_pct_returns(d["trade_return_pct"].values)),
    }
    return report, overall


# ---------------------------
# CLI / Main
# ---------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Regime-conditioned evaluation for trades.")
    p.add_argument("--trades", type=str, default=DEFAULT_TRADES_PATH)
    p.add_argument("--regimes", type=str, default=DEFAULT_REGIMES_PATH)
    p.add_argument("--entry-col", type=str, default="", help="Override entry timestamp column name")
    p.add_argument("--exit-col", type=str, default="", help="Override exit timestamp column name")
    p.add_argument("--label-at", type=str, default="entry", choices=["entry", "exit"])
    p.add_argument("--out-dir", type=str, default=DEFAULT_OUT_DIR)
    p.add_argument("--tag", type=str, default="", help="Optional tag in output filenames")
    return p


def main() -> int:
    args = build_parser().parse_args()
    _ensure_dir(args.out_dir)

    print("Loading trades...")
    trades = load_trades(
        args.trades,
        entry_col_override=args.entry_col.strip() or None,
        exit_col_override=args.exit_col.strip() or None,
    )

    print("Loading regimes...")
    regimes = load_regimes(args.regimes)

    print(f"Assigning regimes at {args.label_at}-time...")
    joined = assign_trade_regimes(trades, regimes, label_at=args.label_at)
    regime_col = f"{args.label_at}_regime"

    report_df, overall = summarize_by_regime(joined, regime_col=regime_col)

    tag = f"_{args.tag}" if args.tag else ""
    joined_out = os.path.join(args.out_dir, f"regime_conditioned_trades_{args.label_at}{tag}.csv")
    report_out = os.path.join(args.out_dir, f"regime_conditioned_report_{args.label_at}{tag}.csv")
    summary_out = os.path.join(args.out_dir, f"regime_conditioned_summary_{args.label_at}{tag}.json")

    joined.to_csv(joined_out, index=False)
    report_df.to_csv(report_out, index=False)
    with open(summary_out, "w", encoding="utf-8") as f:
        json.dump(
            {
                "inputs": {"trades": args.trades, "regimes": args.regimes, "label_at": args.label_at},
                "overall": overall,
                "by_regime": report_df.to_dict(orient="records"),
            },
            f,
            indent=2,
        )

    print("\nSaved:")
    print(f" - {joined_out}")
    print(f" - {report_out}")
    print(f" - {summary_out}")

    print("\n=== OVERALL ===")
    for k, v in overall.items():
        print(f"{k:>28}: {v}")

    print("\n=== BY REGIME ===")
    if len(report_df) == 0:
        print("No regime assignments (check timestamp alignment).")
    else:
        print(report_df.to_string(index=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
