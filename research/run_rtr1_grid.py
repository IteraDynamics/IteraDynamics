from __future__ import annotations

from pathlib import Path
import itertools
import pandas as pd

from engine.backtest_core import BacktestConfig, load_flight_recorder, run_backtest_long_only
from strategies.rtr1 import RTR1Params, build_signals_rtr1


def main():
    data_path = Path("research/backtests/data/flight_recorder.csv")
    df = load_flight_recorder(str(data_path))

    cfg = BacktestConfig(
        fee_bps=6.0,
        slippage_bps=10.0,
        initial_cash=100.0,
        trade_on_close=True,
        min_notional_usd=5.0,
    )

    sma_fast_list = [20, 50, 100]
    sma_regime_list = [150, 200, 300]
    atr_filter_list = [False, True]
    max_hold_list = [None, 24, 48, 72]  # in bars (hours)

    rows = []

    for sma_fast, sma_regime, use_atr, max_hold in itertools.product(
        sma_fast_list, sma_regime_list, atr_filter_list, max_hold_list
    ):
        # Need regime SMA longer than fast SMA for sanity
        if sma_regime <= sma_fast:
            continue

        params = RTR1Params(
            sma_fast=sma_fast,
            sma_regime=sma_regime,
            use_atr_filter=use_atr,
            atr_filter_lookback=200,
            max_hold_bars=max_hold,
            target_frac_equity=1.0,
        )

        sig = build_signals_rtr1(df, params)
        res = run_backtest_long_only(df, sig, cfg)

        s = res.summary
        rows.append(
            {
                "sma_fast": sma_fast,
                "sma_regime": sma_regime,
                "atr_filter": use_atr,
                "max_hold": max_hold if max_hold is not None else "None",
                "trades": s["trades"],
                "win_rate_pct": round(s["win_rate_pct"], 2),
                "total_return_pct": round(s["total_return_pct"], 2),
                "max_drawdown_pct": round(s["max_drawdown_pct"], 2),
                "avg_trade_pct": round(s["avg_trade_pct"], 3),
                "sharpeish": round(s["sharpeish"], 4),
                "final_equity": round(s["final_equity"], 2),
            }
        )

    out = pd.DataFrame(rows)
    out = out.sort_values(["total_return_pct", "sharpeish"], ascending=False).reset_index(drop=True)

    print("\nTop 20 configs:")
    print(out.head(20).to_string(index=False))

    out_dir = Path("research/backtests/results")
    out_dir.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_dir / "rtr1_grid.csv", index=False)
    print("\nSaved: research/backtests/results/rtr1_grid.csv")


if __name__ == "__main__":
    main()
