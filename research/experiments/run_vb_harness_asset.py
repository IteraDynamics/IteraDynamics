"""
Run a Layer-2 strategy via the Argus research harness on an asset OHLCV CSV
(applies the same walk-forward harness logic as BTC).

Notes:
  - Does NOT modify the harness/backtest engine.
  - Computes extra metrics (profit factor, # trades) from the harness equity/exposure
    trace by treating exposure>1% as "in trade" segments.
  - Optional --start_date / --end_date slice the loaded OHLCV before any strategy/backtest
    (no lookahead: only historical bars in the slice are visible).
  - Optional --regime_mode runs predefined time windows and writes a combined metrics CSV.
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
ARGUS_DIR = REPO_ROOT / "runtime" / "argus"

if str(ARGUS_DIR) not in sys.path:
    sys.path.insert(0, str(ARGUS_DIR))

from research.harness.backtest_runner import (  # noqa: E402
    load_flight_recorder,
    load_strategy_func,
    run_backtest,
)

# Predefined regime windows for --regime_mode (orchestration only; no strategy changes).
REGIME_WINDOWS: List[Tuple[str, Optional[str], Optional[str]]] = [
    ("full_cycle", None, None),
    ("crash_window", "2021-07-01", "2022-12-31"),
    ("bull_window", "2023-01-01", "2025-12-30"),
]


def _is_entry_action(action: str) -> bool:
    a = str(action or "").strip().upper()
    return a in {"BUY", "ENTER", "ENTER_LONG", "LONG"}


def _is_exit_action(action: str) -> bool:
    a = str(action or "").strip().upper()
    return a in {"EXIT", "EXIT_LONG", "SELL", "CLOSE"}


def _regime_trend_up(df_slice: pd.DataFrame, ema_len: int = 2000) -> tuple[bool, float | None, float | None]:
    close_col = "Close" if "Close" in df_slice.columns else ("close" if "close" in df_slice.columns else None)
    if close_col is None:
        return False, None, None
    close = pd.to_numeric(df_slice[close_col], errors="coerce").dropna()
    if close.empty or len(close) < ema_len:
        last_close = float(close.iloc[-1]) if not close.empty else None
        return False, last_close, None
    ema = close.ewm(span=ema_len, adjust=False).mean()
    close_last = float(close.iloc[-1])
    ema_last = float(ema.iloc[-1])
    return bool(close_last > ema_last), close_last, ema_last


def _make_regime_gated_strategy(
    strategy_func: Callable,
    *,
    gate_mode: str,
    gate_ema_len: int,
) -> Callable:
    """
    Wrapper-level gating experiment:
      - keep strategy internals unchanged
      - block ENTRY actions outside favorable regime
      - always honor EXIT actions
    """
    gate_mode = (gate_mode or "none").strip().lower()
    if gate_mode == "none":
        return strategy_func

    def _wrapped(df_slice: pd.DataFrame, ctx: Any, *, closed_only: bool = True) -> Dict[str, Any]:
        intent = strategy_func(df_slice, ctx, closed_only=closed_only)
        action = str(intent.get("action", "HOLD"))

        # Exits are never blocked.
        if _is_exit_action(action):
            return intent

        if gate_mode == "trend_up_ema":
            regime_ok, close_last, ema_last = _regime_trend_up(df_slice, ema_len=gate_ema_len)
        else:
            regime_ok, close_last, ema_last = True, None, None

        if _is_entry_action(action) and not regime_ok:
            meta = intent.get("meta") if isinstance(intent.get("meta"), dict) else {}
            new_meta = dict(meta)
            new_meta.update(
                {
                    "regime_gate_mode": gate_mode,
                    "regime_gate_pass": False,
                    "regime_gate_reason": f"entry blocked: {gate_mode}",
                    "regime_close": close_last,
                    "regime_ema": ema_last,
                    "regime_ema_len": gate_ema_len,
                }
            )
            return {
                "action": "HOLD",
                "confidence": float(intent.get("confidence", 0.0)),
                "desired_exposure_frac": 0.0,
                "horizon_hours": int(intent.get("horizon_hours", 0) or 0),
                "reason": f"Regime gate blocked entry ({gate_mode})",
                "meta": new_meta,
            }

        if isinstance(intent.get("meta"), dict):
            out = dict(intent)
            out_meta = dict(intent["meta"])
            out_meta.update(
                {
                    "regime_gate_mode": gate_mode,
                    "regime_gate_pass": bool(regime_ok),
                    "regime_close": close_last,
                    "regime_ema": ema_last,
                    "regime_ema_len": gate_ema_len,
                }
            )
            out["meta"] = out_meta
            return out
        return intent

    return _wrapped


def _infer_data_path(asset: str) -> Path:
    asset = asset.strip().lower()
    p = REPO_ROOT / "data" / f"{asset}usd_3600s_2019-01-01_to_2025-12-30.csv"
    return p


def _slice_ohlcv_by_dates(
    df: pd.DataFrame,
    start_date: Optional[str],
    end_date: Optional[str],
) -> pd.DataFrame:
    """
    Filter rows by Timestamp before harness/strategy run.
    Dates are interpreted as UTC calendar days:
      - start_date: inclusive from 00:00:00 UTC
      - end_date: inclusive through end of that UTC day
    """
    if not start_date and not end_date:
        return df

    out = df.copy()
    ts = pd.to_datetime(out["Timestamp"], utc=True, errors="coerce")
    mask = pd.Series(True, index=out.index)

    if start_date:
        t0 = pd.Timestamp(start_date, tz="UTC")
        mask &= ts >= t0
    if end_date:
        t1_excl = pd.Timestamp(end_date, tz="UTC") + pd.Timedelta(days=1)
        mask &= ts < t1_excl

    return out.loc[mask].reset_index(drop=True)


def _compute_trade_segments(
    equity_df: pd.DataFrame,
    *,
    exposure_threshold: float = 0.01,
) -> List[Tuple[int, int]]:
    expo = equity_df["exposure"].astype(float).to_numpy()
    in_trade = expo > exposure_threshold

    segs: List[Tuple[int, int]] = []
    start: int | None = None
    for i in range(len(in_trade)):
        if start is None and in_trade[i]:
            start = i
        elif start is not None and not in_trade[i]:
            end = i
            if end > start:
                segs.append((start, end))
            start = None

    if start is not None:
        segs.append((start, len(in_trade) - 1))

    return segs


def _compute_profit_factor(trade_returns: np.ndarray) -> float:
    wins = trade_returns[trade_returns > 0]
    losses = trade_returns[trade_returns < 0]

    if wins.size == 0 and losses.size == 0:
        return 0.0
    if wins.size == 0:
        return 0.0
    if losses.size == 0:
        return float("inf")

    gross_profit = float(wins.sum())
    gross_loss = float(abs(losses.sum()))
    return gross_profit / gross_loss if gross_loss > 0 else float("inf")


def _run_harness_once(
    df: pd.DataFrame,
    strategy_func: Callable,
    *,
    lookback: int,
    initial_equity: float,
    fee_bps: float,
    slippage_bps: float,
    exposure_threshold: float,
) -> Tuple[pd.DataFrame, Dict[str, Any], float, int]:
    equity_df, metrics = run_backtest(
        df,
        strategy_func,
        lookback=lookback,
        initial_equity=initial_equity,
        fee_bps=fee_bps,
        slippage_bps=slippage_bps,
        closed_only=True,
        moonwire_feed=None,
    )

    segs = _compute_trade_segments(equity_df, exposure_threshold=exposure_threshold)
    trade_returns: List[float] = []
    for start_i, end_i in segs:
        entry_eq = float(equity_df["equity"].iloc[start_i])
        exit_eq = float(equity_df["equity"].iloc[end_i])
        if entry_eq != 0:
            trade_returns.append((exit_eq / entry_eq) - 1.0)
    trade_returns_arr = np.array(trade_returns, dtype=float)
    profit_factor = _compute_profit_factor(trade_returns_arr)
    n_trades = int(len(trade_returns_arr))

    return equity_df, metrics, profit_factor, n_trades


def _print_metrics_summary(
    metrics: Dict[str, Any],
    profit_factor: float,
    n_trades: int,
) -> None:
    print("\n" + "=" * 60)
    print("METRICS SUMMARY")
    print("=" * 60)
    print(f"Total Return:    {metrics['total_return'] * 100:>10.2f}%")
    print(f"CAGR:            {metrics['cagr'] * 100:>10.2f}%")
    print(f"Max Drawdown:    {metrics['max_drawdown'] * 100:>10.2f}%")
    print(f"Calmar:          {metrics['calmar']:>10.2f}")
    if np.isfinite(profit_factor):
        pf_str = f"{profit_factor:>10.2f}"
    else:
        pf_str = f"{'inf':>10}"
    print(f"Profit Factor:  {pf_str}")
    print(f"# of trades:    {n_trades:>10,}")
    print(f"Exposure (avg): {metrics['avg_exposure'] * 100:>10.2f}%")
    print(f"Time in Market: {metrics['time_in_market'] * 100:>10.2f}%")
    print(f"Final Equity:   ${metrics['final_equity']:>10,.2f}")
    print(f"Bars:           {metrics['bars']:>10,}")
    print(f"Years:          {metrics['years']:>10.2f}")
    print("=" * 60)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Run a strategy module via Argus harness on an asset CSV.",
    )
    ap.add_argument("--asset", type=str, default="eth", help="Asset symbol (e.g. eth, btc).")
    ap.add_argument(
        "--data",
        type=str,
        default="",
        help="Optional explicit CSV path (overrides --asset inference).",
    )
    ap.add_argument(
        "--out_csv",
        type=str,
        default="",
        help=(
            "Output CSV path. Single-run: equity+metrics. "
            "Regime mode: combined regime metrics if set (else default regime filename)."
        ),
    )
    ap.add_argument("--lookback", type=int, default=200)
    ap.add_argument("--initial_equity", type=float, default=10000.0)
    ap.add_argument("--fee_bps", type=float, default=10.0)
    ap.add_argument("--slippage_bps", type=float, default=5.0)
    ap.add_argument("--exposure_threshold", type=float, default=0.01)
    ap.add_argument(
        "--strategy_path",
        type=str,
        default="",
        help=(
            "Optional path to a strategy .py file to load directly (avoids import-path collisions). "
            "If set, overrides --strategy_module. Example: research/strategies/sg_eth_mean_reversion_v1.py"
        ),
    )
    ap.add_argument(
        "--strategy_module",
        type=str,
        default="research.strategies.sg_volatility_breakout_v1",
        help="Import path to strategy module implementing the strategy function.",
    )
    ap.add_argument(
        "--strategy_func",
        type=str,
        default="generate_intent",
        help="Strategy callable name inside --strategy_module.",
    )
    ap.add_argument(
        "--start_date",
        type=str,
        default="",
        help="Optional UTC date YYYY-MM-DD; slice rows with Timestamp >= this day start.",
    )
    ap.add_argument(
        "--end_date",
        type=str,
        default="",
        help="Optional UTC date YYYY-MM-DD; slice rows with Timestamp <= end of this day.",
    )
    ap.add_argument(
        "--regime_mode",
        action="store_true",
        help="Run predefined regime windows; ignores --start_date/--end_date.",
    )
    ap.add_argument(
        "--regime_out_csv",
        type=str,
        default="",
        help="Optional path for regime combined CSV (overrides default and --out_csv).",
    )
    ap.add_argument(
        "--regime_gate_mode",
        type=str,
        choices=["none", "trend_up_ema"],
        default="none",
        help="Optional wrapper-level entry gate mode (default none).",
    )
    ap.add_argument(
        "--gate_ema_len",
        type=int,
        default=2000,
        help="EMA length for trend_up_ema regime gate (default 2000).",
    )
    ap.add_argument(
        "--compare_ungated",
        action="store_true",
        help="In regime_mode, run and print UNGATED vs GATED side-by-side.",
    )
    args = ap.parse_args()

    asset = args.asset.strip().lower()
    strategy_module = args.strategy_module.strip()
    strategy_func_name = args.strategy_func.strip()
    strategy_slug = (Path(args.strategy_path).stem if args.strategy_path else strategy_module.split(".")[-1])
    data_path = Path(args.data).expanduser() if args.data else _infer_data_path(asset)

    if not data_path.exists():
        raise FileNotFoundError(f"Data CSV not found: {data_path}")

    df_full = load_flight_recorder(str(data_path))

    # Strategy load:
    # - Default: import by module path via Argus loader
    # - Optional: load by file path to support NEW research sleeves under repo_root/research/strategies
    if args.strategy_path:
        strat_path = Path(args.strategy_path).expanduser()
        if not strat_path.is_absolute():
            strat_path = (REPO_ROOT / strat_path).resolve()
        if not strat_path.exists():
            raise FileNotFoundError(f"Strategy file not found: {strat_path}")

        # Load under a unique module name to avoid any package shadowing.
        unique_name = f"file_strategy_{strat_path.stem}"
        spec = importlib.util.spec_from_file_location(unique_name, str(strat_path))
        if spec is None or spec.loader is None:
            raise ImportError(f"Failed to create import spec for: {strat_path}")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)  # type: ignore[attr-defined]
        fn = getattr(mod, strategy_func_name, None)
        if fn is None or not callable(fn):
            raise AttributeError(f"Strategy file '{strat_path}' has no callable '{strategy_func_name}'")
        strategy_func_base = fn
        strategy_module = f"{strat_path}"
    else:
        strategy_func_base = load_strategy_func(strategy_module, strategy_func_name)
    strategy_func_gated = _make_regime_gated_strategy(
        strategy_func_base,
        gate_mode=args.regime_gate_mode,
        gate_ema_len=args.gate_ema_len,
    )

    if args.regime_mode:
        regime_rows: List[Dict[str, Any]] = []
        print("=" * 60)
        print("HARNESS REGIME MODE")
        print("=" * 60)
        print(f"Strategy module: {strategy_module}")
        print(f"Strategy func  : {strategy_func_name}")
        print(f"Asset           : {asset}")
        print(f"Data path       : {data_path}")
        print(f"Lookback        : {args.lookback}")
        print(f"Initial equity  : ${args.initial_equity:,.2f}")
        print(f"Fee (bps)       : {args.fee_bps}")
        print(f"Slippage (bps) : {args.slippage_bps}")
        print(f"Regime gate     : {args.regime_gate_mode}")
        if args.regime_gate_mode == "trend_up_ema":
            print(f"Gate EMA len    : {args.gate_ema_len}")
        print(f"Compare ungated : {bool(args.compare_ungated)}")

        for regime_name, w_start, w_end in REGIME_WINDOWS:
            sliced = _slice_ohlcv_by_dates(df_full, w_start, w_end)
            n_bars = len(sliced)
            start_s = w_start or ""
            end_s = w_end or ""

            if n_bars <= args.lookback:
                print(
                    f"\n[{regime_name}] SKIP: insufficient bars after slice "
                    f"({n_bars} <= lookback {args.lookback})"
                )
                regime_rows.append(
                    {
                        "regime_name": regime_name,
                        "start_date": start_s,
                        "end_date": end_s,
                        "bars": n_bars,
                        "CAGR": float("nan"),
                        "MaxDD": float("nan"),
                        "Calmar": float("nan"),
                        "Profit_Factor": float("nan"),
                        "trades": 0,
                        "exposure": float("nan"),
                        "avg_exposure": float("nan"),
                        "time_in_market": float("nan"),
                        "final_equity": float("nan"),
                        "total_return": float("nan"),
                        "error": "insufficient_bars",
                    }
                )
                continue

            try:
                _, metrics_g, pf_g, n_trades_g = _run_harness_once(
                    sliced,
                    strategy_func_gated,
                    lookback=args.lookback,
                    initial_equity=args.initial_equity,
                    fee_bps=args.fee_bps,
                    slippage_bps=args.slippage_bps,
                    exposure_threshold=args.exposure_threshold,
                )
            except Exception as e:
                print(f"\n[{regime_name}] ERROR: {e}")
                regime_rows.append(
                    {
                        "regime_name": regime_name,
                        "start_date": start_s,
                        "end_date": end_s,
                        "bars": n_bars,
                        "CAGR": float("nan"),
                        "MaxDD": float("nan"),
                        "Calmar": float("nan"),
                        "Profit_Factor": float("nan"),
                        "trades": 0,
                        "exposure": float("nan"),
                        "avg_exposure": float("nan"),
                        "time_in_market": float("nan"),
                        "final_equity": float("nan"),
                        "total_return": float("nan"),
                        "gate_mode": args.regime_gate_mode,
                        "error": str(e),
                    }
                )
                continue

            if args.compare_ungated:
                _, metrics_u, pf_u, n_trades_u = _run_harness_once(
                    sliced,
                    strategy_func_base,
                    lookback=args.lookback,
                    initial_equity=args.initial_equity,
                    fee_bps=args.fee_bps,
                    slippage_bps=args.slippage_bps,
                    exposure_threshold=args.exposure_threshold,
                )
                regime_rows.append(
                    {
                        "regime_name": regime_name,
                        "start_date": start_s,
                        "end_date": end_s,
                        "bars": metrics_g["bars"],
                        "gate_mode": args.regime_gate_mode,
                        "ungated_CAGR": metrics_u["cagr"],
                        "ungated_MaxDD": metrics_u["max_drawdown"],
                        "ungated_Calmar": metrics_u["calmar"],
                        "ungated_Profit_Factor": pf_u if np.isfinite(pf_u) else float("inf"),
                        "ungated_trades": n_trades_u,
                        "ungated_exposure": metrics_u["avg_exposure"],
                        "ungated_final_equity": metrics_u["final_equity"],
                        "gated_CAGR": metrics_g["cagr"],
                        "gated_MaxDD": metrics_g["max_drawdown"],
                        "gated_Calmar": metrics_g["calmar"],
                        "gated_Profit_Factor": pf_g if np.isfinite(pf_g) else float("inf"),
                        "gated_trades": n_trades_g,
                        "gated_exposure": metrics_g["avg_exposure"],
                        "gated_final_equity": metrics_g["final_equity"],
                        "error": "",
                    }
                )
            else:
                regime_rows.append(
                    {
                        "regime_name": regime_name,
                        "start_date": start_s,
                        "end_date": end_s,
                        "bars": metrics_g["bars"],
                        "CAGR": metrics_g["cagr"],
                        "MaxDD": metrics_g["max_drawdown"],
                        "Calmar": metrics_g["calmar"],
                        "Profit_Factor": pf_g if np.isfinite(pf_g) else float("inf"),
                        "trades": n_trades_g,
                        "exposure": metrics_g["avg_exposure"],
                        "avg_exposure": metrics_g["avg_exposure"],
                        "time_in_market": metrics_g["time_in_market"],
                        "final_equity": metrics_g["final_equity"],
                        "total_return": metrics_g["total_return"],
                        "gate_mode": args.regime_gate_mode,
                        "error": "",
                    }
                )

        regime_df = pd.DataFrame(regime_rows)

        if args.regime_out_csv:
            regime_path = Path(args.regime_out_csv).expanduser()
            if not regime_path.is_absolute():
                regime_path = (REPO_ROOT / regime_path).resolve()
        elif args.out_csv:
            regime_path = Path(args.out_csv).expanduser()
            if not regime_path.is_absolute():
                regime_path = (REPO_ROOT / regime_path).resolve()
        else:
            regime_path = (
                REPO_ROOT
                / "research"
                / "experiments"
                / "output"
                / f"vb_regime_results_{asset}_{strategy_slug}.csv"
            ).resolve()

        regime_path.parent.mkdir(parents=True, exist_ok=True)
        regime_df.to_csv(regime_path, index=False)

        print("\n" + "=" * 60)
        print("REGIME SUMMARY")
        print("=" * 60)
        for _, row in regime_df.iterrows():
            print(f"\n[{row['regime_name']}]")
            print(f"  window: {row['start_date'] or '—'} .. {row['end_date'] or '—'}")
            if row.get("error") and str(row["error"]).strip():
                print(f"  (skipped/error: {row['error']})")
                continue
            if args.compare_ungated:
                print("  UNGATED:")
                print(f"    CAGR:    {float(row['ungated_CAGR']) * 100:>10.2f}%")
                print(f"    MaxDD:   {float(row['ungated_MaxDD']) * 100:>10.2f}%")
                print(f"    Calmar:  {float(row['ungated_Calmar']):>10.2f}")
                pf_u = row["ungated_Profit_Factor"]
                pf_u_str = f"{float(pf_u):>10.2f}" if np.isfinite(float(pf_u)) else f"{'inf':>10}"
                print(f"    PF:      {pf_u_str}")
                print(f"    trades:  {int(row['ungated_trades']):>10,}")
                print(f"    exposure:{float(row['ungated_exposure']) * 100:>10.2f}%")
                print(f"    final $: ${float(row['ungated_final_equity']):>10,.2f}")

                print("  GATED:")
                print(f"    CAGR:    {float(row['gated_CAGR']) * 100:>10.2f}%")
                print(f"    MaxDD:   {float(row['gated_MaxDD']) * 100:>10.2f}%")
                print(f"    Calmar:  {float(row['gated_Calmar']):>10.2f}")
                pf_g = row["gated_Profit_Factor"]
                pf_g_str = f"{float(pf_g):>10.2f}" if np.isfinite(float(pf_g)) else f"{'inf':>10}"
                print(f"    PF:      {pf_g_str}")
                print(f"    trades:  {int(row['gated_trades']):>10,}")
                print(f"    exposure:{float(row['gated_exposure']) * 100:>10.2f}%")
                print(f"    final $: ${float(row['gated_final_equity']):>10,.2f}")
                continue

            cagr_v = row["CAGR"]
            mdd_v = row["MaxDD"]
            cal_v = row["Calmar"]
            if pd.isna(cagr_v) or pd.isna(mdd_v) or pd.isna(cal_v):
                print("  (metrics unavailable)")
                continue
            print(f"  CAGR:    {float(cagr_v) * 100:>10.2f}%")
            print(f"  MaxDD:   {float(mdd_v) * 100:>10.2f}%")
            print(f"  Calmar:  {float(cal_v):>10.2f}")
            pf = row["Profit_Factor"]
            if isinstance(pf, (int, float)) and np.isfinite(float(pf)):
                print(f"  PF:      {float(pf):>10.2f}")
            else:
                print("  PF:      inf")
            print(f"  trades:  {int(row['trades']):>10,}")
            ae = row["exposure"]
            fe = row["final_equity"]
            if pd.isna(ae) or pd.isna(fe):
                print("  exposure:       n/a")
                print("  final $:        n/a")
            else:
                print(f"  exposure:{float(ae) * 100:>10.2f}%")
                print(f"  final $: ${float(fe):>10,.2f}")
        print("=" * 60)
        print(f"Wrote regime summary: {regime_path}")
        return

    # Single-window run (existing behavior + optional date slice)
    start_d = args.start_date.strip() or None
    end_d = args.end_date.strip() or None
    df = _slice_ohlcv_by_dates(df_full, start_d, end_d)

    if args.out_csv:
        out_path = Path(args.out_csv).expanduser()
        if not out_path.is_absolute():
            out_path = (REPO_ROOT / out_path).resolve()
    else:
        out_path = (
            REPO_ROOT / "research" / "experiments" / "output" / f"{strategy_slug}_{asset}_results.csv"
        ).resolve()

    print("=" * 60)
    print("HARNESS STRATEGY RUN (asset swap)")
    print("=" * 60)
    print(f"Strategy module: {strategy_module}")
    print(f"Strategy func  : {strategy_func_name}")
    print(f"Asset           : {asset}")
    print(f"Data path       : {data_path}")
    if start_d or end_d:
        print(f"Date slice      : {start_d or '—'} .. {end_d or '—'} (UTC, inclusive days)")
    print(f"Bars (sliced)   : {len(df):,}")
    print(f"Lookback        : {args.lookback}")
    print(f"Initial equity  : ${args.initial_equity:,.2f}")
    print(f"Fee (bps)       : {args.fee_bps}")
    print(f"Slippage (bps) : {args.slippage_bps}")

    strategy_func_single = strategy_func_gated
    equity_df, metrics, profit_factor, n_trades = _run_harness_once(
        df,
        strategy_func_single,
        lookback=args.lookback,
        initial_equity=args.initial_equity,
        fee_bps=args.fee_bps,
        slippage_bps=args.slippage_bps,
        exposure_threshold=args.exposure_threshold,
    )

    _print_metrics_summary(metrics, profit_factor, n_trades)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df = equity_df.copy()
    out_df["asset"] = asset
    out_df["strategy_module"] = strategy_module
    out_df["strategy_func"] = strategy_func_name
    out_df["gate_mode"] = args.regime_gate_mode
    out_df["lookback"] = args.lookback
    out_df["fee_bps"] = args.fee_bps
    out_df["slippage_bps"] = args.slippage_bps
    out_df["initial_equity"] = args.initial_equity
    if start_d:
        out_df["slice_start_date"] = start_d
    if end_d:
        out_df["slice_end_date"] = end_d

    out_df["total_return"] = metrics["total_return"]
    out_df["CAGR"] = metrics["cagr"]
    out_df["Max Drawdown"] = metrics["max_drawdown"]
    out_df["Calmar"] = metrics["calmar"]
    out_df["Profit Factor"] = profit_factor
    out_df["n_trades"] = n_trades
    out_df["avg_exposure"] = metrics["avg_exposure"]
    out_df["time_in_market"] = metrics["time_in_market"]

    out_df.to_csv(out_path, index=False)
    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()
