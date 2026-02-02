# sniper/sniper_backtest.py
# 🔫 BTC SNIPER BACKTEST – REV 2
#
# CLI wrapper for Sniper core:
#   - loads BTC OHLCV CSV
#   - wires SniperConfig
#   - runs backtest & prints stats
#   - writes:
#       data/sniper_equity_curve.csv
#       data/sniper_trades.csv

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .sniper_core import SniperConfig, add_features, run_sniper_backtest


def _load_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    df = pd.read_csv(path)

    if "Timestamp" not in df.columns:
        raise ValueError("Expected a 'Timestamp' column in the CSV data.")

    df["Timestamp"] = pd.to_datetime(df["Timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["Timestamp"])
    df = df.set_index("Timestamp").sort_index()

    required_cols = {"Open", "High", "Low", "Close"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in data: {missing}")

    return df


def main():
    parser = argparse.ArgumentParser(description="BTC Sniper backtest (Rev-2).")

    parser.add_argument("--data-file", required=True, help="Path to BTC OHLCV CSV.")
    parser.add_argument("--initial-equity", type=float, default=100.0, help="Starting equity in USD.")
    parser.add_argument("--fee-bps", type=float, default=10.0, help="Fee in basis points per side.")
    parser.add_argument("--risk-fraction", type=float, default=0.5, help="Fraction of equity to risk per trade.")

    # Strategy knobs
    parser.add_argument("--fast-sma", type=int, default=20, help="Fast SMA length for trend filter.")
    parser.add_argument("--slow-sma", type=int, default=100, help="Slow SMA length for trend filter.")
    parser.add_argument("--breakout-zscore", type=float, default=2.0, help="Return z-score threshold for entry.")
    parser.add_argument("--lookback-vol", type=int, default=48, help="Lookback window (bars) for volatility calc.")
    parser.add_argument("--atr-length", type=int, default=14, help="ATR length in bars.")
    parser.add_argument("--stop-atr-mult", type=float, default=1.5, help="Stop distance in ATR multiples.")
    parser.add_argument("--target-atr-mult", type=float, default=3.0, help="Target distance in ATR multiples.")
    parser.add_argument("--max-hold-hours", type=int, default=72, help="Maximum holding period in hours.")

    args = parser.parse_args()

    data_path = Path(args.data_file)
    initial_equity = float(args.initial_equity)

    print("🔫 Running BTC Sniper backtest (Rev-2)...")
    print(f"   Data file:      {data_path}")
    print(f"   Initial equity: ${initial_equity:,.2f}")
    print(f"   Fee:            {args.fee_bps:.2f} bps per side")
    print(f"   Risk fraction:  {args.risk_fraction:.2f}")
    print(f"   Fast SMA:       {args.fast_sma}")
    print(f"   Slow SMA:       {args.slow_sma}")
    print(f"   Breakout z:     {args.breakout_zscore:.2f}")
    print(f"   Vol lookback:   {args.lookback_vol} bars")
    print(f"   ATR length:     {args.atr_length}")
    print(f"   Stop ATR mult:  {args.stop_atr_mult:.2f}")
    print(f"   Target ATR mult:{args.target_atr_mult:.2f}")
    print(f"   Max hold:       {args.max_hold_hours}h")

    df = _load_data(data_path)

    cfg = SniperConfig(
        fast_sma=args.fast_sma,
        slow_sma=args.slow_sma,
        breakout_zscore=args.breakout_zscore,
        lookback_vol=args.lookback_vol,
        atr_length=args.atr_length,
        stop_atr_mult=args.stop_atr_mult,
        target_atr_mult=args.target_atr_mult,
        max_hold_hours=args.max_hold_hours,
        risk_fraction=args.risk_fraction,
        fee_bps=args.fee_bps,
    )

    df_feat = add_features(df, cfg)
    equity_df, trades = run_sniper_backtest(df_feat, cfg, initial_equity=initial_equity)

    final_equity = float(equity_df["equity"].iloc[-1])
    total_return = final_equity / initial_equity - 1.0

    rolling_max = equity_df["equity"].cummax()
    drawdown = equity_df["equity"] / rolling_max - 1.0
    max_dd = float(drawdown.min()) if not drawdown.empty else 0.0

    n_trades = len(trades)
    wins = sum(1 for t in trades if t.net_pnl > 0)
    win_rate = wins / n_trades if n_trades > 0 else 0.0
    total_pnl = sum(t.net_pnl for t in trades)

    eq_out_path = data_path.parent / "sniper_equity_curve.csv"
    trades_out_path = data_path.parent / "sniper_trades.csv"

    equity_df.to_csv(eq_out_path, index=True)

    if trades:
        trades_df = pd.DataFrame(
            [
                {
                    "entry_time": t.entry_time,
                    "exit_time": t.exit_time,
                    "side": t.side,
                    "qty": t.qty,
                    "entry_price": t.entry_price,
                    "exit_price": t.exit_price,
                    "gross_pnl": t.gross_pnl,
                    "net_pnl": t.net_pnl,
                    "return_pct": t.return_pct,
                    "bars_held": t.bars_held,
                    "exit_reason": t.exit_reason,
                }
                for t in trades
            ]
        )
        trades_df.to_csv(trades_out_path, index=False)
    else:
        trades_out_path.write_text("no trades\n")

    print("\n📊 Backtest Results (Rev-2)")
    print("---------------------------")
    print(f"Final equity:     ${final_equity:,.2f}")
    print(f"Total return:     {total_return * 100:,.2f}%")
    print(f"Max drawdown:     {max_dd * 100:,.2f}%")
    print(f"Trades:           {n_trades}")
    print(f"Win rate:         {win_rate * 100:,.2f}%")
    print(f"Total net P&L:    ${total_pnl:,.2f}")
    print(f"\n💾 Equity curve written to: {eq_out_path}")
    print(f"💾 Trade log written to:   {trades_out_path}")


if __name__ == "__main__":
    main()
