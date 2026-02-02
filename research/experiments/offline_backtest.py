"""
offline_backtest.py

Offline evaluation for Argus' RandomForest model using the same features as signal_generator.py:
  - RSI (14)
  - BB_Pos (20, 2)
  - Vol_Z (20)

Key properties:
  - explicit paths (no systemd/env reliance)
  - offline only (no API calls)
  - single-position, no-overlap execution (realistic)
  - emits tradability metrics + equity curve stats
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import csv
import math

import numpy as np
import pandas as pd
import joblib


# -----------------------------
# Config / Defaults
# -----------------------------

@dataclass
class BacktestConfig:
    horizon: int = 24              # holding period in hours/bars
    conf: float = 0.55             # probability threshold to enter
    fee_bps: float = 6.0           # fee per side in bps
    slip_bps: float = 5.0          # slippage per side in bps
    long_only: bool = True         # if False, allow shorts when p <= (1-conf)
    min_rows: int = 300            # sanity gate


# -----------------------------
# Feature Engineering
# -----------------------------

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Pure pandas RSI (Wilder-style EMA approximation)."""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)

    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    out = 100 - (100 / (1 + rs))
    return out


def bb_pos(close: pd.Series, length: int = 20, std: float = 2.0) -> pd.Series:
    """Bollinger Band position: (close - lower) / (upper - lower)."""
    mid = close.rolling(length).mean()
    sd = close.rolling(length).std(ddof=0)
    lower = mid - std * sd
    upper = mid + std * sd
    denom = (upper - lower).replace(0, np.nan)
    return (close - lower) / denom


def vol_z(volume: pd.Series, length: int = 20) -> pd.Series:
    """Volume z-score over rolling window."""
    m = volume.rolling(length).mean()
    s = volume.rolling(length).std(ddof=0).replace(0, np.nan)
    return (volume - m) / s


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce", utc=False)
    df = df.dropna(subset=["Timestamp"]).sort_values("Timestamp").reset_index(drop=True)

    df["RSI"] = rsi(df["Close"], period=14)
    df["BB_Pos"] = bb_pos(df["Close"], length=20, std=2.0)
    df["Vol_Z"] = vol_z(df["Volume"], length=20)
    return df


def build_labels(df: pd.DataFrame, horizon: int = 24) -> pd.DataFrame:
    """Binary label: 1 if Future_Close > Close else 0 (for diagnostics only)."""
    df = df.copy()
    df["Future_Close"] = df["Close"].shift(-horizon)
    df["Target"] = (df["Future_Close"] > df["Close"]).astype(int)
    return df


# -----------------------------
# Backtest Helpers
# -----------------------------

def apply_round_trip_cost(ret: float, fee_bps: float, slip_bps: float) -> float:
    """
    Apply per-trade costs (round-trip):
      entry fee + exit fee + entry slip + exit slip
    Approx as simple bps deducted from return.
    """
    round_trip_bps = 2.0 * (fee_bps + slip_bps)
    cost = round_trip_bps / 10_000.0
    return float(ret - cost)


def equity_curve_stats(equity: np.ndarray) -> dict:
    """
    equity: array of equity values (starting at 1.0)
    Returns max drawdown and a simple Sharpe-ish using per-bar returns.
    """
    equity = np.asarray(equity, dtype=float)
    if equity.size < 2:
        return {"max_dd": 0.0, "sharpe": 0.0}

    peaks = np.maximum.accumulate(equity)
    dd = (equity / peaks) - 1.0
    max_dd = float(np.min(dd))

    rets = np.diff(equity) / equity[:-1]
    mu = float(np.mean(rets))
    sd = float(np.std(rets, ddof=0))
    sharpe = float(mu / sd) * math.sqrt(24 * 365) if sd > 0 else 0.0  # annualized-ish for hourly bars
    return {"max_dd": max_dd, "sharpe": sharpe}


def trade_stats(trade_rets: np.ndarray) -> dict:
    trade_rets = np.asarray(trade_rets, dtype=float)
    trade_rets = trade_rets[np.isfinite(trade_rets)]

    if trade_rets.size == 0:
        return {
            "trades": 0,
            "win_rate": None,
            "avg_return": None,
            "median_return": None,
            "profit_factor": None,
            "sum_return": 0.0,
        }

    wins = trade_rets[trade_rets > 0]
    losses = trade_rets[trade_rets < 0]
    win_rate = float((trade_rets > 0).mean())

    if losses.size > 0:
        profit_factor = float(wins.sum() / abs(losses.sum())) if wins.size > 0 else 0.0
    else:
        profit_factor = float("inf") if wins.size > 0 else 0.0

    return {
        "trades": int(trade_rets.size),
        "win_rate": win_rate,
        "avg_return": float(trade_rets.mean()),
        "median_return": float(np.median(trade_rets)),
        "profit_factor": profit_factor,
        "sum_return": float(trade_rets.sum()),
    }


# -----------------------------
# Single-Position Backtest
# -----------------------------

def run_single_position_backtest(df: pd.DataFrame, model, cfg: BacktestConfig) -> dict:
    feature_cols = ["RSI", "BB_Pos", "Vol_Z"]
    expected_cols = {"Timestamp", "Open", "High", "Low", "Close", "Volume"}
    missing_base = expected_cols - set(df.columns)
    if missing_base:
        raise ValueError(f"flight_recorder.csv missing columns: {missing_base}")

    df = df.copy()
    df = df.dropna(subset=feature_cols).reset_index(drop=True)

    if len(df) < cfg.min_rows:
        raise ValueError(f"Not enough rows after feature prep. Have={len(df)}, need>={cfg.min_rows}")

    # Need enough bars to exit trades
    if len(df) <= cfg.horizon + 5:
        raise ValueError(f"Not enough rows to run horizon={cfg.horizon} backtest. rows={len(df)}")

    # Predict with DataFrame to preserve feature names (silences sklearn warning)
    X = df[feature_cols].astype(float)
    if hasattr(model, "predict_proba"):
        p_long = model.predict_proba(X)[:, 1]
    else:
        p_long = model.predict(X).astype(float)

    # Labels for diagnostics only (not used for trading)
    y = df["Target"].astype(int).values if "Target" in df.columns else None

    # --- Execution model ---
    # - At most one position open at any time.
    # - Enter at bar i close (approx).
    # - Exit at bar i+horizon close (approx).
    # - While in a trade, we do NOT open new ones.
    #
    # Position side:
    #   long-only: enter long if p>=conf
    #   long/short: enter long if p>=conf, enter short if p<=1-conf

    n = len(df)
    equity = [1.0]
    trade_rets = []
    trades_detail = []  # optional debug

    i = 0
    while i < n - cfg.horizon:
        enter = False
        side = 0  # +1 long, -1 short

        if cfg.long_only:
            if p_long[i] >= cfg.conf:
                enter = True
                side = +1
        else:
            if p_long[i] >= cfg.conf:
                enter = True
                side = +1
            elif p_long[i] <= (1.0 - cfg.conf):
                enter = True
                side = -1

        if not enter:
            # no position; equity flat for this step
            equity.append(equity[-1])
            i += 1
            continue

        entry_px = float(df["Close"].iloc[i])
        exit_ix = i + cfg.horizon
        exit_px = float(df["Close"].iloc[exit_ix])

        raw_ret = (exit_px / entry_px) - 1.0
        signed_ret = raw_ret if side == +1 else -raw_ret
        net_ret = apply_round_trip_cost(signed_ret, cfg.fee_bps, cfg.slip_bps)

        # Update equity at exit; during holding we keep it flat (conservative mark-to-close model)
        # (You can later mark-to-market each bar, but this keeps it simple and non-overlapping.)
        eq0 = equity[-1]
        eq1 = eq0 * (1.0 + net_ret)

        # Fill flat bars during holding period (optional; keeps equity aligned to time)
        # We add cfg.horizon steps of flat equity, then the jump at exit.
        for _ in range(cfg.horizon - 1):
            equity.append(equity[-1])
        equity.append(eq1)

        trade_rets.append(net_ret)
        trades_detail.append(
            {
                "entry_i": i,
                "exit_i": exit_ix,
                "entry_ts": str(df["Timestamp"].iloc[i]),
                "exit_ts": str(df["Timestamp"].iloc[exit_ix]),
                "side": "LONG" if side == 1 else "SHORT",
                "p": float(p_long[i]),
                "ret": float(net_ret),
            }
        )

        # Jump index to first bar AFTER exit (no overlap)
        i = exit_ix + 1

    equity_arr = np.asarray(equity, dtype=float)
    stats_eq = equity_curve_stats(equity_arr)
    stats_tr = trade_stats(np.asarray(trade_rets, dtype=float))

    # Diagnostics: majority baseline + model 0.5 accuracy (only if labels exist)
    majority_baseline_accuracy = None
    model_accuracy = None
    label_up_rate = None
    if y is not None and len(y) == len(df):
        # only compare on rows where label is defined
        y2 = df["Target"].astype(int).values
        mask = np.isfinite(y2)
        y2 = y2[mask]
        p2 = p_long[mask]
        if y2.size:
            majority = int(y2.mean() >= 0.5)
            majority_baseline_accuracy = float((y2 == majority).mean())
            model_accuracy = float(((p2 >= 0.5).astype(int) == y2).mean())
            label_up_rate = float(np.mean(y2))

    return {
        "rows_used": int(len(df)),
        "horizon_h": int(cfg.horizon),
        "conf_threshold": float(cfg.conf),
        "fee_bps": float(cfg.fee_bps),
        "slip_bps": float(cfg.slip_bps),
        "long_only": bool(cfg.long_only),
        "p_up_mean": float(np.mean(p_long)),
        "label_up_rate": label_up_rate,
        "majority_baseline_accuracy": majority_baseline_accuracy,
        "model_accuracy": model_accuracy,
        "trade_stats": stats_tr,
        "equity": {
            "final_equity": float(equity_arr[-1]) if equity_arr.size else 1.0,
            "max_dd": float(stats_eq["max_dd"]),
            "sharpe": float(stats_eq["sharpe"]),
        },
        "trades_detail": trades_detail,  # keep for debugging; you can omit if you want
    }


# -----------------------------
# CSV Emit
# -----------------------------

def emit_row(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    # Flatten row
    ts = row.get("trade_stats", {}) or {}
    eq = row.get("equity", {}) or {}

    out = {
        "horizon_h": row.get("horizon_h"),
        "conf": row.get("conf_threshold"),
        "long_only": row.get("long_only"),
        "fee_bps": row.get("fee_bps"),
        "slip_bps": row.get("slip_bps"),
        "rows_used": row.get("rows_used"),
        "p_up_mean": row.get("p_up_mean"),
        "label_up_rate": row.get("label_up_rate"),
        "majority_baseline_accuracy": row.get("majority_baseline_accuracy"),
        "model_accuracy": row.get("model_accuracy"),
        "trades": ts.get("trades"),
        "win_rate": ts.get("win_rate"),
        "avg_return": ts.get("avg_return"),
        "median_return": ts.get("median_return"),
        "profit_factor": ts.get("profit_factor"),
        "sum_return": ts.get("sum_return"),
        "final_equity": eq.get("final_equity"),
        "max_dd": eq.get("max_dd"),
        "sharpe": eq.get("sharpe"),
    }

    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(out.keys()))
        if write_header:
            w.writeheader()
        w.writerow(out)


# -----------------------------
# CLI / Main
# -----------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=str, default=None, help="Path to flight_recorder.csv")
    p.add_argument("--model", type=str, default=None, help="Path to random_forest.pkl")
    p.add_argument("--horizon", type=int, default=24, help="Holding horizon in hours (default 24)")
    p.add_argument("--conf", type=float, default=0.55, help="Entry threshold (default 0.55)")
    p.add_argument("--fee_bps", type=float, default=6.0, help="Fee per side in bps (default 6)")
    p.add_argument("--slip_bps", type=float, default=5.0, help="Slippage per side in bps (default 5)")
    p.add_argument("--long_short", action="store_true", help="Enable long/short (default long-only)")
    p.add_argument("--emit", type=str, default=None, help="Append one summary row to this CSV path")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    ROOT = Path(__file__).resolve().parent

    # Default paths (your local layout)
    DATA = Path(args.data) if args.data else (ROOT / "runtime" / "argus" / "flight_recorder.csv")
    MODEL = Path(args.model) if args.model else (ROOT / "runtime" / "argus" / "models" / "random_forest.pkl")

    print("ROOT :", ROOT)
    print("DATA :", DATA, "| exists:", DATA.exists())
    print("MODEL:", MODEL, "| exists:", MODEL.exists())

    if not DATA.exists():
        raise FileNotFoundError(f"Data CSV not found at: {DATA}")
    if not MODEL.exists():
        raise FileNotFoundError(f"Model file not found at: {MODEL}")

    cfg = BacktestConfig(
        horizon=int(args.horizon),
        conf=float(args.conf),
        fee_bps=float(args.fee_bps),
        slip_bps=float(args.slip_bps),
        long_only=(not args.long_short),
    )

    model = joblib.load(MODEL)
    df = pd.read_csv(DATA)

    # Basic schema check
    expected_cols = {"Timestamp", "Open", "High", "Low", "Close", "Volume"}
    missing = expected_cols - set(df.columns)
    if missing:
        raise ValueError(
            f"flight_recorder.csv is missing columns: {missing}\n"
            f"Found columns: {list(df.columns)}"
        )

    df = build_features(df)
    df = build_labels(df, horizon=cfg.horizon)

    result = run_single_position_backtest(df, model, cfg)

    print("\n================ OFFLINE BACKTEST RESULT (SINGLE POSITION) ================")
    print(f"rows_used: {result['rows_used']}")
    print(f"p_up_mean (model avg prob): {result['p_up_mean']:.4f}")
    if result["label_up_rate"] is not None:
        print(f"label_up_rate (p(up) true): {result['label_up_rate']:.4f}")
        print(f"majority_baseline_accuracy: {result['majority_baseline_accuracy']:.4f}")
        print(f"model_accuracy:            {result['model_accuracy']:.4f}")

    print(f"horizon_h: {result['horizon_h']} | conf: {result['conf_threshold']:.2f} | long_only: {result['long_only']}")
    print(f"fees/slip (bps per side): fee={result['fee_bps']} slip={result['slip_bps']}")

    ts = result["trade_stats"]
    eq = result["equity"]

    print("\n---------------- Trades (NON-OVERLAP) ----------------")
    print(f"trades:        {ts['trades']}")
    print(f"win_rate:      {ts['win_rate'] if ts['win_rate'] is not None else None}")
    print(f"avg_return:    {ts['avg_return'] if ts['avg_return'] is not None else None}")
    print(f"median_return: {ts['median_return'] if ts['median_return'] is not None else None}")
    print(f"profit_factor: {ts['profit_factor'] if ts['profit_factor'] is not None else None}")
    print(f"sum_return:    {ts['sum_return']:.6f}   (unitless, additive over trades)")

    print("\n---------------- Equity ----------------")
    print(f"final_equity: {eq['final_equity']:.6f}  (start=1.0)")
    print(f"max_dd:       {eq['max_dd']:.4f}  (e.g. -0.25 = -25%)")
    print(f"sharpe-ish:   {eq['sharpe']:.3f}  (rough, hourly annualization)")

    if args.emit:
        out_csv = Path(args.emit)
        emit_row(out_csv, result)
        print(f"\n[emit] appended summary row -> {out_csv}")

    print("=========================================================================\n")


if __name__ == "__main__":
    main()
