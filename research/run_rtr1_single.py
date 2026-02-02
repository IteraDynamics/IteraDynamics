from __future__ import annotations

from pathlib import Path

from engine.backtest_core import (
    BacktestConfig,
    load_flight_recorder,
    run_backtest_long_only,
    buy_and_hold_baseline,
)
from strategies.rtr1 import RTR1Params, build_signals_rtr1


def main():
    data_path = Path("research/backtests/data/flight_recorder.csv")
    df = load_flight_recorder(str(data_path))

    cfg = BacktestConfig(
        fee_bps=0.0,
        slippage_bps=0.0,
        initial_cash=100.0,
        trade_on_close=True,
        min_notional_usd=5.0,
    )

    params = RTR1Params(
        sma_fast=50,
        sma_regime=200,
        use_atr_filter=True,
        atr_filter_lookback=200,
        max_hold_bars=None,     # try 48 later if you want time-boxing
        target_frac_equity=1.0,
    )

    sig = build_signals_rtr1(df, params)

    # For now we deploy all cash; later we can implement vol targeting sizing.
    res = run_backtest_long_only(df, sig, cfg)
    bh = buy_and_hold_baseline(df, cfg)

    print("\n=== RTR-1 SUMMARY ===")
    for k, v in res.summary.items():
        print(f"{k:>18}: {v}")

    print("\n=== BUY & HOLD (costed) ===")
    for k, v in bh.items():
        print(f"{k:>18}: {v}")

    # Save
    out_dir = Path("research/backtests/results")
    out_dir.mkdir(parents=True, exist_ok=True)
    res.trades.to_csv(out_dir / "rtr1_trades.csv", index=False)
    res.equity_curve.to_csv(out_dir / "rtr1_equity.csv", index=False)
    print("\nSaved:")
    print(" - research/backtests/results/rtr1_trades.csv")
    print(" - research/backtests/results/rtr1_equity.csv")


if __name__ == "__main__":
    main()
