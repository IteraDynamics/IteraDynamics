r"""
offline_walkforward.py

Walk-forward backtest (offline) using historical 1h BTC-USD candles from Coinbase.

- Reads CSV produced by download_coinbase_history.py (expected columns: Timestamp/Open/High/Low/Close/Volume)
- Builds features: RSI(14), BB_Pos(20,2), Vol_Z(20)
- Builds label: FwdRet > target_ret at horizon H   (tradable magnitude label)
- Walk-forward training: train_days -> test_days, stepping by step_days
- Trains a new model each fold (RF default)
- Enforces NO OVERLAP / SINGLE POSITION:
    * enter only if flat
    * hold exactly horizon bars (or until end of test window)
    * no pyramiding, no overlapping trades

Outputs:
  - fold metrics CSV (default: wf_folds.csv)
  - trades CSV (default: wf_trades.csv)

Example:
  python .\offline_walkforward.py --data .\data\historical\btc_usd_1h.csv --horizon 24 --conf 0.54 --target_ret 0.0035 --train_days 180 --test_days 30 --step_days 30
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier


# -----------------------------
# Config
# -----------------------------

@dataclass
class WFConfig:
    horizon: int = 24
    conf: float = 0.54
    fee_bps: float = 6.0
    slip_bps: float = 5.0

    # IMPORTANT: tradable-magnitude label threshold
    # e.g. 0.0035 = +0.35% over horizon
    target_ret: float = 0.0035

    train_days: int = 180
    test_days: int = 30
    step_days: int = 30

    long_only: bool = True

    # row gates
    min_train_rows: int = 24 * 30   # at least ~30 days of hourly rows
    min_test_rows: int = 24 * 7     # at least ~1 week

    # model params
    n_estimators: int = 300
    max_depth: int = 6
    random_state: int = 42


# -----------------------------
# Features (pure pandas)
# -----------------------------

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def bb_pos(close: pd.Series, length: int = 20, std: float = 2.0) -> pd.Series:
    mid = close.rolling(length).mean()
    sd = close.rolling(length).std(ddof=0)
    lower = mid - std * sd
    upper = mid + std * sd
    denom = (upper - lower).replace(0, np.nan)
    return (close - lower) / denom


def vol_z(volume: pd.Series, length: int = 20) -> pd.Series:
    m = volume.rolling(length).mean()
    s = volume.rolling(length).std(ddof=0).replace(0, np.nan)
    return (volume - m) / s


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce", utc=False)
    df = df.dropna(subset=["Timestamp"]).sort_values("Timestamp").reset_index(drop=True)

    df["RSI"] = rsi(df["Close"], 14)
    df["BB_Pos"] = bb_pos(df["Close"], 20, 2.0)
    df["Vol_Z"] = vol_z(df["Volume"], 20)
    return df


def build_labels(df: pd.DataFrame, horizon: int, target_ret: float) -> pd.DataFrame:
    """
    FwdRet = Close[t+h]/Close[t] - 1
    Target = 1 if FwdRet > target_ret else 0
    """
    df = df.copy()
    df["Future_Close"] = df["Close"].shift(-horizon)
    df["FwdRet"] = df["Future_Close"] / df["Close"] - 1.0
    df["Target"] = (df["FwdRet"] > float(target_ret)).astype(int)
    return df


# -----------------------------
# Walk-forward splitter
# -----------------------------

def make_folds(
    df: pd.DataFrame,
    train_days: int,
    test_days: int,
    step_days: int,
) -> List[Tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
    """
    Returns list of folds:
      (train_start, train_end, test_start, test_end)
    All ranges are time-based and inclusive on start, exclusive on end.
    """
    t0 = df["Timestamp"].min()
    t1 = df["Timestamp"].max()

    train = pd.Timedelta(days=train_days)
    test = pd.Timedelta(days=test_days)
    step = pd.Timedelta(days=step_days)

    folds = []
    test_start = t0 + train
    while True:
        train_start = test_start - train
        train_end = test_start
        test_end = test_start + test

        if test_end > t1:
            break

        folds.append((train_start, train_end, test_start, test_end))
        test_start = test_start + step

    return folds


# -----------------------------
# Trading mechanics (NO OVERLAP)
# -----------------------------

def round_trip_cost(fee_bps: float, slip_bps: float) -> float:
    return 2.0 * (fee_bps + slip_bps) / 10_000.0


def simulate_single_position(
    test_df: pd.DataFrame,
    p_long: np.ndarray,
    cfg: WFConfig,
) -> Tuple[List[Dict], Dict]:
    """
    Single position, no overlap:
      - if flat and signal passes threshold -> enter at close[t]
      - exit after horizon bars at close[t+h]
    """
    cost = round_trip_cost(cfg.fee_bps, cfg.slip_bps)
    trades: List[Dict] = []

    i = 0
    n = len(test_df)
    while i < n:
        if i + cfg.horizon >= n:
            break

        prob = float(p_long[i])
        enter = False
        side = "LONG"

        if cfg.long_only:
            if prob >= cfg.conf:
                enter = True
        else:
            if prob >= cfg.conf:
                enter = True
                side = "LONG"
            elif prob <= (1.0 - cfg.conf):
                enter = True
                side = "SHORT"

        if not enter:
            i += 1
            continue

        entry_ts = test_df["Timestamp"].iloc[i]
        entry_px = float(test_df["Close"].iloc[i])
        exit_i = i + cfg.horizon
        exit_ts = test_df["Timestamp"].iloc[exit_i]
        exit_px = float(test_df["Close"].iloc[exit_i])

        raw_ret = (exit_px / entry_px - 1.0)
        if side == "SHORT":
            raw_ret = -raw_ret

        net_ret = raw_ret - cost

        trades.append(
            {
                "entry_ts": entry_ts,
                "exit_ts": exit_ts,
                "side": side,
                "p": prob,
                "entry_px": entry_px,
                "exit_px": exit_px,
                "raw_ret": raw_ret,
                "net_ret": net_ret,
                "bars_held": cfg.horizon,
            }
        )

        # jump to exit (no overlap)
        i = exit_i

    net = np.array([t["net_ret"] for t in trades], dtype=float)
    if len(net) == 0:
        stats = {
            "trades": 0,
            "win_rate": None,
            "avg_net": None,
            "median_net": None,
            "profit_factor": None,
            "sum_net": 0.0,
        }
    else:
        wins = net[net > 0]
        losses = net[net < 0]
        pf = float(wins.sum() / abs(losses.sum())) if losses.size else (float("inf") if wins.size else 0.0)
        stats = {
            "trades": int(len(net)),
            "win_rate": float((net > 0).mean()),
            "avg_net": float(net.mean()),
            "median_net": float(np.median(net)),
            "profit_factor": pf,
            "sum_net": float(net.sum()),
        }

    return trades, stats


# -----------------------------
# Main
# -----------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=str, default=None, help="Path to historical 1h CSV")
    p.add_argument("--out", type=str, default="wf_folds.csv", help="Fold metrics output CSV")
    p.add_argument("--trades_out", type=str, default="wf_trades.csv", help="Trades output CSV")

    p.add_argument("--horizon", type=int, default=24)
    p.add_argument("--conf", type=float, default=0.54)
    p.add_argument("--fee_bps", type=float, default=6.0)
    p.add_argument("--slip_bps", type=float, default=5.0)
    p.add_argument("--target_ret", type=float, default=0.0035, help="Label threshold (e.g. 0.0035 = +0.35%)")
    p.add_argument("--long_short", action="store_true")

    p.add_argument("--train_days", type=int, default=180)
    p.add_argument("--test_days", type=int, default=30)
    p.add_argument("--step_days", type=int, default=30)

    # model knobs
    p.add_argument("--n_estimators", type=int, default=300)
    p.add_argument("--max_depth", type=int, default=6)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parent
    data_path = Path(args.data) if args.data else (root / "data" / "historical" / "btc_usd_1h.csv")

    print(f"ROOT: {root}")
    print(f"DATA: {data_path} | exists: {data_path.exists()}")

    if not data_path.exists():
        raise FileNotFoundError(f"Missing data file: {data_path}")

    cfg = WFConfig(
        horizon=int(args.horizon),
        conf=float(args.conf),
        fee_bps=float(args.fee_bps),
        slip_bps=float(args.slip_bps),
        target_ret=float(args.target_ret),
        train_days=int(args.train_days),
        test_days=int(args.test_days),
        step_days=int(args.step_days),
        long_only=(not args.long_short),
        n_estimators=int(args.n_estimators),
        max_depth=int(args.max_depth),
    )

    df = pd.read_csv(data_path)

    expected = {"Timestamp", "Open", "High", "Low", "Close", "Volume"}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing columns: {missing} | found: {list(df.columns)}")

    df = build_features(df)
    df = build_labels(df, cfg.horizon, cfg.target_ret)

    feature_cols = ["RSI", "BB_Pos", "Vol_Z"]
    df = df.dropna(subset=feature_cols + ["Target", "FwdRet"]).reset_index(drop=True)

    folds = make_folds(df, cfg.train_days, cfg.test_days, cfg.step_days)
    if not folds:
        span_days = (df["Timestamp"].max() - df["Timestamp"].min()).total_seconds() / 86400.0
        raise RuntimeError(
            f"No folds produced.\n"
            f"- data span days: {span_days:.2f}\n"
            f"- try smaller windows, e.g. --train_days 120 --test_days 14 --step_days 14\n"
        )

    fold_rows = []
    all_trades = []

    for k, (tr0, tr1, te0, te1) in enumerate(folds, start=1):
        train_df = df[(df["Timestamp"] >= tr0) & (df["Timestamp"] < tr1)].copy()
        test_df = df[(df["Timestamp"] >= te0) & (df["Timestamp"] < te1)].copy()

        if len(train_df) < cfg.min_train_rows or len(test_df) < cfg.min_test_rows:
            continue

        Xtr = train_df[feature_cols].astype(float).values
        ytr = train_df["Target"].astype(int).values
        Xte = test_df[feature_cols].astype(float).values
        yte = test_df["Target"].astype(int).values

        model = RandomForestClassifier(
            n_estimators=cfg.n_estimators,
            max_depth=cfg.max_depth,
            random_state=cfg.random_state,
            class_weight="balanced",
            n_jobs=-1,
        )
        model.fit(Xtr, ytr)
        p_long = model.predict_proba(Xte)[:, 1]

        # classification metrics (sanity only)
        yhat = (p_long >= 0.5).astype(int)
        majority = int(np.round(yte.mean()))
        base_acc = float((yte == majority).mean())
        acc = float((yhat == yte).mean())

        trades, tstats = simulate_single_position(test_df, p_long, cfg)

        for t in trades:
            t["fold"] = k
            t["test_start"] = te0
            t["test_end"] = te1
        all_trades.extend(trades)

        fold_rows.append(
            {
                "fold": k,
                "train_start": tr0,
                "train_end": tr1,
                "test_start": te0,
                "test_end": te1,
                "train_rows": int(len(train_df)),
                "test_rows": int(len(test_df)),
                "label_up_rate_test": float(yte.mean()),
                "baseline_acc_test": base_acc,
                "model_acc_test": acc,
                "conf": cfg.conf,
                "horizon_h": cfg.horizon,
                "target_ret": cfg.target_ret,
                "fee_bps": cfg.fee_bps,
                "slip_bps": cfg.slip_bps,
                "long_only": cfg.long_only,
                **{f"trade_{k2}": v for k2, v in tstats.items()},
            }
        )

    if not fold_rows:
        raise RuntimeError(
            "No folds survived row gates. Lower --min_* gates in code, or use smaller windows.\n"
            "Try: --train_days 120 --test_days 14 --step_days 14"
        )

    out_path = Path(args.out)
    trades_path = Path(args.trades_out)

    pd.DataFrame(fold_rows).to_csv(out_path, index=False)
    pd.DataFrame(all_trades).to_csv(trades_path, index=False)

    folds_df = pd.DataFrame(fold_rows)
    trades_df = pd.DataFrame(all_trades)

    print("\n================ WALKFORWARD SUMMARY ================")
    print(f"folds_written: {len(folds_df)} -> {out_path}")
    print(f"trades_written: {len(trades_df)} -> {trades_path}")
    print(f"avg_fold_trades: {folds_df['trade_trades'].mean():.2f}")

    if len(trades_df):
        net = trades_df["net_ret"].astype(float).values
        print(f"trades_total: {len(net)}")
        print(f"win_rate:     {(net > 0).mean():.3f}")
        print(f"avg_net:      {net.mean():.5f}")
        print(f"median_net:   {np.median(net):.5f}")
        wins = net[net > 0]
        losses = net[net < 0]
        pf = wins.sum() / abs(losses.sum()) if losses.size else float("inf")
        print(f"profit_factor:{pf:.3f}")
        print(f"sum_net:      {net.sum():.5f}")
    print("=====================================================\n")


if __name__ == "__main__":
    main()
