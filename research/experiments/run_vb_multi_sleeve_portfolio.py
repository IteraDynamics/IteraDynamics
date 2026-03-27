"""
Deterministic multi-sleeve portfolio simulation (no optimization).

Goal:
  Combine BTC VB and ETH VB-v3 sleeves under a simple allocator:
    - Let desired exposures be e_btc, e_eth
    - If e_btc + e_eth <= 1: use as-is
    - Else normalize: w_btc=e_btc/total, w_eth=e_eth/total

Notes:
  - Reuses existing harness backtest for each sleeve independently.
  - Does NOT modify strategy or backtest engine internals.
  - Produces comparable metrics for BTC-only, ETH-only, and Combined.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
ARGUS_DIR = REPO_ROOT / "runtime" / "argus"

if str(ARGUS_DIR) not in sys.path:
    sys.path.insert(0, str(ARGUS_DIR))

from research.harness.backtest_runner import (  # noqa: E402
    load_strategy_func,
    run_backtest,
)


@dataclass
class SleeveConfig:
    name: str
    asset: str
    data: Path
    strategy_module: str
    strategy_func: str


def _infer_data_path(asset: str) -> Path:
    return REPO_ROOT / "data" / f"{asset.strip().lower()}usd_3600s_2019-01-01_to_2025-12-30.csv"


def _load_ohlcv_flexible(csv_path: Path) -> pd.DataFrame:
    """
    Flexible CSV loader for case variants in column names.
    Normalizes to harness-expected columns:
      Timestamp, Open, High, Low, Close, Volume
    """
    raw = pd.read_csv(csv_path)
    if raw.empty:
        raise ValueError(f"CSV is empty: {csv_path}")

    # Case-insensitive column map
    lower_to_actual = {str(c).strip().lower(): c for c in raw.columns}

    def _pick(*names: str) -> str | None:
        for n in names:
            c = lower_to_actual.get(n.lower())
            if c is not None:
                return c
        return None

    col_ts = _pick("Timestamp", "timestamp", "ts", "datetime", "date")
    col_o = _pick("Open", "open", "o")
    col_h = _pick("High", "high", "h")
    col_l = _pick("Low", "low", "l")
    col_c = _pick("Close", "close", "c")
    col_v = _pick("Volume", "volume", "v")

    required_missing = [
        n
        for n, c in [("Timestamp", col_ts), ("Open", col_o), ("High", col_h), ("Low", col_l), ("Close", col_c)]
        if c is None
    ]
    if required_missing:
        raise ValueError(f"{csv_path} missing required columns: {required_missing}. Found: {list(raw.columns)}")

    df = pd.DataFrame(
        {
            "Timestamp": pd.to_datetime(raw[col_ts], utc=True, errors="coerce"),
            "Open": pd.to_numeric(raw[col_o], errors="coerce"),
            "High": pd.to_numeric(raw[col_h], errors="coerce"),
            "Low": pd.to_numeric(raw[col_l], errors="coerce"),
            "Close": pd.to_numeric(raw[col_c], errors="coerce"),
            "Volume": pd.to_numeric(raw[col_v], errors="coerce") if col_v is not None else 0.0,
        }
    )
    df = df.dropna(subset=["Timestamp", "Open", "High", "Low", "Close"]).sort_values("Timestamp").reset_index(drop=True)
    return df


def _compute_profit_factor_from_returns(returns: np.ndarray) -> float:
    r = returns[~np.isnan(returns)]
    wins = r[r > 0]
    losses = r[r < 0]
    if wins.size == 0 and losses.size == 0:
        return 0.0
    if wins.size == 0:
        return 0.0
    if losses.size == 0:
        return float("inf")
    gp = float(wins.sum())
    gl = float(abs(losses.sum()))
    return gp / gl if gl > 0 else float("inf")


def _max_drawdown(equity: np.ndarray) -> float:
    peak = np.maximum.accumulate(equity)
    dd = (equity / np.maximum(peak, 1e-12)) - 1.0
    return float(np.nanmin(dd))


def _compute_metrics(ts: pd.Series, returns: np.ndarray, equity: np.ndarray, avg_exposure: float) -> Dict[str, float]:
    tsu = pd.to_datetime(ts, utc=True)
    seconds = (tsu.max() - tsu.min()).total_seconds()
    years = seconds / (365.25 * 24 * 3600.0) if seconds > 0 else float("nan")

    total_return = (equity[-1] / equity[0] - 1.0) if equity[0] > 0 else float("nan")
    if years and years > 0 and equity[0] > 0 and equity[-1] > 0:
        cagr = float((equity[-1] / equity[0]) ** (1.0 / years) - 1.0)
    else:
        cagr = float("nan")

    mdd_raw = _max_drawdown(equity)
    maxdd = abs(mdd_raw)
    calmar = cagr / maxdd if maxdd > 1e-12 and np.isfinite(cagr) else float("nan")
    pf = _compute_profit_factor_from_returns(returns)

    return {
        "CAGR": cagr,
        "MaxDD": maxdd,
        "Calmar": calmar,
        "PF": pf,
        "exposure": float(avg_exposure),
        "final_equity": float(equity[-1]),
        "total_return": float(total_return),
        "bars": int(len(equity)),
        "years": float(years) if years == years else float("nan"),
    }


def _run_sleeve(cfg: SleeveConfig, *, lookback: int, initial_equity: float, fee_bps: float, slippage_bps: float) -> pd.DataFrame:
    data_path = cfg.data
    if not data_path.is_absolute():
        data_path = (REPO_ROOT / data_path).resolve()
    if not data_path.exists():
        raise FileNotFoundError(f"{cfg.name}: data file not found: {data_path}")

    df = _load_ohlcv_flexible(data_path)
    fn = load_strategy_func(cfg.strategy_module, cfg.strategy_func)
    eq_df, _ = run_backtest(
        df,
        fn,
        lookback=lookback,
        initial_equity=initial_equity,
        fee_bps=fee_bps,
        slippage_bps=slippage_bps,
        closed_only=True,
        moonwire_feed=None,
    )

    out = eq_df[["Timestamp", "equity", "price", "desired_exposure_frac", "exposure"]].copy()
    out["Timestamp"] = pd.to_datetime(out["Timestamp"], utc=True, errors="coerce")
    out = out.dropna(subset=["Timestamp"]).sort_values("Timestamp").reset_index(drop=True)
    out["return"] = out["equity"].pct_change().fillna(0.0)
    out = out.rename(
        columns={
            "equity": f"{cfg.name}_equity",
            "price": f"{cfg.name}_price",
            "desired_exposure_frac": f"{cfg.name}_desired_exposure",
            "exposure": f"{cfg.name}_exposure",
            "return": f"{cfg.name}_return",
        }
    )
    return out


def _align_sleeves(btc_df: pd.DataFrame, eth_df: pd.DataFrame) -> pd.DataFrame:
    merged = pd.merge(btc_df, eth_df, on="Timestamp", how="inner").sort_values("Timestamp").reset_index(drop=True)
    if len(merged) < 2:
        raise ValueError("Aligned BTC/ETH timeline has fewer than 2 rows.")
    return merged


def _suffix_output_path(raw_path: str, suffix: str) -> Path:
    """
    Make output paths unique per run.
    - If caller provides "{suffix}" token, replace it.
    - Else append "_{suffix}" before file extension.
    """
    p = Path(raw_path)
    p_str = str(p)
    if "{suffix}" in p_str:
        return Path(p_str.replace("{suffix}", suffix))
    return p.with_name(f"{p.stem}_{suffix}{p.suffix}")


def _simulate_combined(
    merged: pd.DataFrame,
    initial_equity: float,
    *,
    fee_bps: float,
    slippage_bps: float,
    allocator_mode: str = "normalize",
    allow_gross_above_one: bool = False,
    enable_sanity_checks: bool = True,
) -> pd.DataFrame:
    df = merged.copy()
    e_btc = np.clip(df["btc_desired_exposure"].to_numpy(dtype=float), 0.0, 1.0)
    e_eth = np.clip(df["eth_desired_exposure"].to_numpy(dtype=float), 0.0, 1.0)
    total = e_btc + e_eth

    mode = (allocator_mode or "normalize").strip().lower()
    w_btc = np.zeros_like(e_btc, dtype=float)
    w_eth = np.zeros_like(e_eth, dtype=float)

    if mode == "normalize":
        # Safe normalization without divide-by-zero warnings.
        no_norm = total <= 1.0
        w_btc[no_norm] = e_btc[no_norm]
        w_eth[no_norm] = e_eth[no_norm]
        need_norm = total > 1.0
        np.divide(e_btc, total, out=w_btc, where=need_norm)
        np.divide(e_eth, total, out=w_eth, where=need_norm)
    elif mode in ("btc_priority", "eth_when_btc_flat"):
        # Priority allocator: BTC first; ETH only when BTC flat.
        btc_on = e_btc > 0.0
        w_btc[btc_on] = np.minimum(e_btc[btc_on], 1.0)
        w_eth[btc_on] = 0.0
        btc_off = ~btc_on
        w_btc[btc_off] = 0.0
        w_eth[btc_off] = np.minimum(e_eth[btc_off], 1.0)
    elif mode == "btc_capped_eth":
        # BTC primary, ETH opportunistic with hard cap (no normalization).
        w_btc = np.minimum(e_btc, 1.0)
        btc_on = w_btc > 0.0
        w_eth[btc_on] = np.minimum(e_eth[btc_on], 0.25)
        w_eth[~btc_on] = np.minimum(e_eth[~btc_on], 1.0)
        if not allow_gross_above_one:
            # Optional guard: clip ETH only, never scale BTC.
            remaining = np.maximum(1.0 - w_btc, 0.0)
            w_eth = np.minimum(w_eth, remaining)
    else:
        raise ValueError(f"Unknown allocator_mode: {allocator_mode}")

    # Use raw asset returns (not sleeve-equity returns) for portfolio blending.
    r_btc_asset = pd.Series(df["btc_price"].to_numpy(dtype=float)).pct_change().fillna(0.0).to_numpy()
    r_eth_asset = pd.Series(df["eth_price"].to_numpy(dtype=float)).pct_change().fillna(0.0).to_numpy()

    # No lookahead: apply lagged weights to current bar asset returns.
    w_btc_lag = np.roll(w_btc, 1)
    w_eth_lag = np.roll(w_eth, 1)
    if len(w_btc_lag) > 0:
        w_btc_lag[0] = 0.0
        w_eth_lag[0] = 0.0

    btc_contrib = w_btc_lag * r_btc_asset
    eth_contrib = w_eth_lag * r_eth_asset
    gross_ret = btc_contrib + eth_contrib

    # Portfolio-level turnover/cost from allocator weight changes.
    turnover = np.zeros_like(gross_ret, dtype=float)
    if len(turnover) > 1:
        turnover[1:] = np.abs(w_btc[1:] - w_btc_lag[1:]) + np.abs(w_eth[1:] - w_eth_lag[1:])
    cost_rate = (fee_bps + slippage_bps) / 10_000.0
    costs = turnover * cost_rate
    net_ret = gross_ret - costs

    eq = np.empty(len(net_ret), dtype=float)
    eq[0] = initial_equity
    for i in range(1, len(net_ret)):
        eq[i] = eq[i - 1] * (1.0 + net_ret[i])

    if enable_sanity_checks:
        gross_exposure = w_btc + w_eth
        # 1) Gross exposure cap when leverage is not explicitly allowed.
        if (not allow_gross_above_one) and np.nanmax(gross_exposure) > 1.0000001:
            raise AssertionError(f"Gross exposure exceeded 1 (mode={mode}): max={np.nanmax(gross_exposure):.6f}")
        # 2) Net return identity.
        if not np.allclose(net_ret, btc_contrib + eth_contrib - costs, atol=1e-12, rtol=1e-9, equal_nan=True):
            raise AssertionError("Net return identity failed: net != btc_contrib + eth_contrib - costs")
        # 3) Guard against blending sleeve-equity returns in combined path.
        if "btc_return" in df.columns and "eth_return" in df.columns:
            wrong_path = (w_btc_lag * df["btc_return"].to_numpy(dtype=float)) + (w_eth_lag * df["eth_return"].to_numpy(dtype=float))
            if np.allclose(net_ret, wrong_path, atol=1e-10, rtol=1e-7, equal_nan=True):
                raise AssertionError("Combined path appears to be using sleeve-equity returns (forbidden for audit run)")

    df["w_btc"] = w_btc
    df["w_eth"] = w_eth
    df["w_btc_lag"] = w_btc_lag
    df["w_eth_lag"] = w_eth_lag
    df["btc_asset_return"] = r_btc_asset
    df["eth_asset_return"] = r_eth_asset
    df["btc_contrib"] = btc_contrib
    df["eth_contrib"] = eth_contrib
    df["gross_return_before_costs"] = gross_ret
    df["turnover"] = turnover
    df["costs"] = costs
    df["combined_return"] = net_ret
    df["combined_equity"] = eq
    df["total_exposure_pre_norm"] = total
    df["allocator_mode"] = mode
    df["gross_exposure"] = w_btc + w_eth
    return df


def main() -> None:
    ap = argparse.ArgumentParser(description="Run deterministic BTC+ETH multi-sleeve VB portfolio simulation.")
    ap.add_argument("--btc_data", type=str, default=str(_infer_data_path("btc")))
    ap.add_argument("--eth_data", type=str, default=str(_infer_data_path("eth")))
    ap.add_argument("--btc_strategy_module", type=str, default="research.strategies.sg_volatility_breakout_v1")
    ap.add_argument("--btc_strategy_func", type=str, default="generate_intent")
    ap.add_argument("--eth_strategy_module", type=str, default="research.strategies.sg_volatility_breakout_v3_volfilter")
    ap.add_argument("--eth_strategy_func", type=str, default="generate_intent")
    ap.add_argument("--lookback", type=int, default=200)
    ap.add_argument("--initial_equity", type=float, default=10000.0)
    ap.add_argument("--fee_bps", type=float, default=10.0)
    ap.add_argument("--slippage_bps", type=float, default=5.0)
    ap.add_argument(
        "--allocator_mode",
        type=str,
        choices=["normalize", "btc_priority", "btc_capped_eth", "eth_when_btc_flat"],
        default="normalize",
        help="Deterministic allocator mode (default normalize for backward compatibility).",
    )
    ap.add_argument(
        "--allow_gross_above_one",
        action="store_true",
        help="Allow gross exposure > 1.0 (only relevant for btc_capped_eth mode).",
    )
    ap.add_argument("--audit_rows", type=int, default=100, help="Number of rows to export in audit table.")
    ap.add_argument(
        "--out_audit",
        type=str,
        default="",
        help="Output path for compact audit CSV (default includes cost suffix).",
    )
    ap.add_argument(
        "--out_metrics",
        type=str,
        default="",
        help="Output path for metrics CSV (default includes cost suffix).",
    )
    ap.add_argument(
        "--out_equity",
        type=str,
        default="",
        help="Output path for combined equity curve CSV (default includes cost suffix).",
    )
    args = ap.parse_args()

    mode_slug = args.allocator_mode
    suffix = f"{mode_slug}_{int(args.fee_bps)}_{int(args.slippage_bps)}"
    if args.out_metrics:
        out_metrics = _suffix_output_path(args.out_metrics, suffix)
    else:
        out_metrics = REPO_ROOT / "research" / "experiments" / "output" / f"vb_allocator_{suffix}.csv"
    if args.out_equity:
        out_equity = _suffix_output_path(args.out_equity, suffix)
    else:
        out_equity = REPO_ROOT / "research" / "experiments" / "output" / f"vb_allocator_equity_{suffix}.csv"
    if args.out_audit:
        out_audit = _suffix_output_path(args.out_audit, suffix)
    else:
        out_audit = REPO_ROOT / "research" / "experiments" / "output" / f"vb_allocator_audit_{suffix}.csv"
    if not out_metrics.is_absolute():
        out_metrics = (REPO_ROOT / out_metrics).resolve()
    if not out_equity.is_absolute():
        out_equity = (REPO_ROOT / out_equity).resolve()
    if not out_audit.is_absolute():
        out_audit = (REPO_ROOT / out_audit).resolve()

    btc_cfg = SleeveConfig(
        name="btc",
        asset="btc",
        data=Path(args.btc_data),
        strategy_module=args.btc_strategy_module,
        strategy_func=args.btc_strategy_func,
    )
    eth_cfg = SleeveConfig(
        name="eth",
        asset="eth",
        data=Path(args.eth_data),
        strategy_module=args.eth_strategy_module,
        strategy_func=args.eth_strategy_func,
    )

    print("=" * 60)
    print("MULTI-SLEEVE VB PORTFOLIO RUN")
    print("=" * 60)
    print(f"BTC sleeve: {btc_cfg.strategy_module} | {btc_cfg.data}")
    print(f"ETH sleeve: {eth_cfg.strategy_module} | {eth_cfg.data}")
    print(f"Lookback: {args.lookback} | Initial equity: {args.initial_equity:,.2f}")
    print(f"Costs: fee_bps={args.fee_bps}, slippage_bps={args.slippage_bps}")
    print(f"Allocator mode: {args.allocator_mode} | allow_gross_above_one={bool(args.allow_gross_above_one)}")

    btc_df = _run_sleeve(
        btc_cfg,
        lookback=args.lookback,
        initial_equity=args.initial_equity,
        fee_bps=args.fee_bps,
        slippage_bps=args.slippage_bps,
    )
    eth_df = _run_sleeve(
        eth_cfg,
        lookback=args.lookback,
        initial_equity=args.initial_equity,
        fee_bps=args.fee_bps,
        slippage_bps=args.slippage_bps,
    )

    merged = _align_sleeves(btc_df, eth_df)
    sim = _simulate_combined(
        merged,
        initial_equity=args.initial_equity,
        fee_bps=args.fee_bps,
        slippage_bps=args.slippage_bps,
        allocator_mode=args.allocator_mode,
        allow_gross_above_one=bool(args.allow_gross_above_one),
        enable_sanity_checks=True,
    )

    m_btc = _compute_metrics(
        sim["Timestamp"],
        sim["btc_return"].to_numpy(dtype=float),
        sim["btc_equity"].to_numpy(dtype=float),
        avg_exposure=float(sim["btc_exposure"].mean()),
    )
    m_eth = _compute_metrics(
        sim["Timestamp"],
        sim["eth_return"].to_numpy(dtype=float),
        sim["eth_equity"].to_numpy(dtype=float),
        avg_exposure=float(sim["eth_exposure"].mean()),
    )
    m_comb = _compute_metrics(
        sim["Timestamp"],
        sim["combined_return"].to_numpy(dtype=float),
        sim["combined_equity"].to_numpy(dtype=float),
        avg_exposure=float((sim["w_btc"] + sim["w_eth"]).mean()),
    )

    rows = []
    for sleeve_name, m in [("BTC_only", m_btc), ("ETH_only", m_eth), ("Combined", m_comb)]:
        rows.append(
            {
                "sleeve": sleeve_name,
                "CAGR": m["CAGR"],
                "MaxDD": m["MaxDD"],
                "Calmar": m["Calmar"],
                "PF": m["PF"],
                "exposure": m["exposure"],
                "final_equity": m["final_equity"],
                "total_return": m["total_return"],
                "bars": m["bars"],
                "years": m["years"],
                "fee_bps": args.fee_bps,
                "slippage_bps": args.slippage_bps,
                "allocator_mode": args.allocator_mode,
            }
        )
    metrics_df = pd.DataFrame(rows)

    out_metrics.parent.mkdir(parents=True, exist_ok=True)
    out_equity.parent.mkdir(parents=True, exist_ok=True)
    out_audit.parent.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(out_metrics, index=False)
    sim_out = sim[
        [
            "Timestamp",
            "btc_equity",
            "eth_equity",
            "combined_equity",
            "btc_return",
            "eth_return",
            "btc_asset_return",
            "eth_asset_return",
            "combined_return",
            "btc_desired_exposure",
            "eth_desired_exposure",
            "total_exposure_pre_norm",
            "w_btc",
            "w_eth",
            "w_btc_lag",
            "w_eth_lag",
            "btc_contrib",
            "eth_contrib",
            "gross_return_before_costs",
            "turnover",
            "costs",
        ]
    ].copy()
    sim_out.to_csv(out_equity, index=False)

    audit_cols = [
        "Timestamp",
        "btc_price",
        "eth_price",
        "btc_asset_return",
        "eth_asset_return",
        "btc_desired_exposure",
        "eth_desired_exposure",
        "total_exposure_pre_norm",
        "w_btc",
        "w_eth",
        "w_btc_lag",
        "w_eth_lag",
        "btc_contrib",
        "eth_contrib",
        "gross_return_before_costs",
        "turnover",
        "costs",
        "combined_return",
        "combined_equity",
    ]
    audit_n = max(1, int(args.audit_rows))
    sim[audit_cols].head(audit_n).to_csv(out_audit, index=False)

    print("\n" + "=" * 60)
    print("PORTFOLIO SUMMARY")
    print("=" * 60)
    for _, row in metrics_df.iterrows():
        pf = row["PF"]
        pf_str = f"{float(pf):.2f}" if np.isfinite(float(pf)) else "inf"
        print(f"[{row['sleeve']}]")
        print(f"  CAGR:   {float(row['CAGR']) * 100:8.2f}%")
        print(f"  MaxDD:  {float(row['MaxDD']) * 100:8.2f}%")
        print(f"  Calmar: {float(row['Calmar']):8.2f}")
        print(f"  PF:     {pf_str:>8}")
        print(f"  Exp:    {float(row['exposure']) * 100:8.2f}%")
        print(f"  Final:  ${float(row['final_equity']):10,.2f}")
    print("=" * 60)
    print(f"Wrote metrics: {out_metrics}")
    print(f"Wrote equity:  {out_equity}")
    print(f"Wrote audit:   {out_audit}")


if __name__ == "__main__":
    main()

