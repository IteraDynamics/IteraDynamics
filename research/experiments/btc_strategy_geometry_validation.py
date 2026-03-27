"""
BTC Strategy Geometry Validation (RESEARCH ONLY)

Objective:
  Isolate BTC-only geometry for:
    - Core strategy:   sg_core_exposure_v2
    - Sleeve 2A:       sg_volatility_breakout_v1
    - Static blends:   80/20, 70/30, 50/50 (Core/Sleeve2A)

Notes:
  - Reuses the same cost interface and metric definitions as portfolio_geometry_validation.py.
  - Does NOT touch ETH or MoonWire; BTC-only timeline and weights.
  - Closed-bar determinism: decision at bar close t applies to return t -> t+1.
  - Aligned with Argus harness: load BTC without drop-last (like load_flight_recorder),
    and use ARGUS_LOOKBACK=200 so first trading bar and history length match the harness.
"""

from __future__ import annotations

import argparse
import importlib
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
_RUNTIME_ARGUS = REPO_ROOT / "runtime" / "argus"
_EXPERIMENTS = REPO_ROOT / "research" / "experiments"
if str(_RUNTIME_ARGUS) not in sys.path:
    sys.path.insert(0, str(_RUNTIME_ARGUS))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(_EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(_EXPERIMENTS))

from cost_regimes import (  # type: ignore[import]
    COST_REGIME_CUSTOM,
    COST_REGIME_INSTITUTIONAL,
    COST_REGIME_PRO_TARGET,
    COST_REGIME_RETAIL_LAUNCH,
    resolve_cost_params,
)

# Canonical windows (match existing geometry)
FULL_START, FULL_END = "2019-01-01", "2025-12-30"
CRASH_START, CRASH_END = "2021-07-01", "2022-12-31"
POST_START, POST_END = "2023-01-01", "2025-12-30"

WINDOWS: List[Tuple[str, str, str]] = [
    ("full_cycle", FULL_START, FULL_END),
    ("crash_window", CRASH_START, CRASH_END),
    ("post_crash", POST_START, POST_END),
]

SCENARIOS = [
    "CORE_ONLY",
    "SLEEVE2A_ONLY",
    "BLEND_80_20",
    "BLEND_70_30",
    "BLEND_50_50",
]

DEFAULT_FEE_BPS = 10
DEFAULT_SLIPPAGE_BPS = 5

# Match Argus harness: first trading bar at index ARGUS_LOOKBACK (no trading before that).
# Harness default ARGUS_LOOKBACK=200; Core needs 100 bars min but harness uses 200 for warmup.
ARGUS_LOOKBACK = 200
PROGRESS_INTERVAL_PCT = 5  # print every 5% of bars (e.g. ~20 lines per strategy)

DEFAULT_RUNTIME_ARGUS_DIR = REPO_ROOT / "runtime" / "argus"

# Strategy modules
CORE_MODULE_CANDIDATES = [
    "research.strategies.sg_core_exposure_v2",
    "runtime.argus.research.strategies.sg_core_exposure_v2",
]
CORE_FUNC = "generate_intent"

SLEEVE2A_MODULE_CANDIDATES = [
    "research.strategies.sg_volatility_breakout_v1",
    "runtime.argus.research.strategies.sg_volatility_breakout_v1",
]
SLEEVE2A_FUNC = "generate_intent"


@dataclass(frozen=True)
class RunConfig:
    mode: str
    fee_bps: float
    slippage_bps: float
    cost_regime: str
    out_dir: Path
    out_csv: Path
    btc_data_file: Path
    initial_equity: float = 10000.0


# ---------------------------------------------------------------------
# Loading + preprocessing (BTC-only)
# ---------------------------------------------------------------------


def _read_price_csv_from_path(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise ValueError(f"Data file not found: {path}")
    return pd.read_csv(path)


def _load_btc_harness_style(path: Path) -> pd.DataFrame:
    """
    Load BTC like the Argus harness load_flight_recorder: no drop-last, same OHLC/Volume handling.
    Ensures Core and Sleeve2A see the same bar set and history length as the harness.
    """
    raw = _read_price_csv_from_path(path)
    if "Timestamp" not in raw.columns:
        raise ValueError("BTC CSV must contain Timestamp column")
    for col in ("Open", "High", "Low", "Close"):
        if col not in raw.columns:
            raise ValueError(f"BTC CSV missing required OHLC column: {col}")
    df = raw.copy()
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["Timestamp"]).sort_values("Timestamp").reset_index(drop=True)
    for col in ("Open", "High", "Low", "Close"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["Open", "High", "Low", "Close"]).reset_index(drop=True)
    if "Volume" in df.columns:
        df["Volume"] = pd.to_numeric(df["Volume"], errors="coerce")
    else:
        df["Volume"] = 0.0
    if len(df) < ARGUS_LOOKBACK + 2:
        raise ValueError(
            f"BTC has {len(df)} rows; need >= {ARGUS_LOOKBACK + 2} to match harness lookback {ARGUS_LOOKBACK}"
        )
    return df


# ---------------------------------------------------------------------
# Strategy adapters
# ---------------------------------------------------------------------


def _load_callable(candidates: List[str], func_name: str):
    last_err: Exception | None = None
    for mod_name in candidates:
        try:
            mod = importlib.import_module(mod_name)
            fn = getattr(mod, func_name, None)
            if callable(fn):
                return fn
            last_err = RuntimeError(f"Func not callable: {mod_name}.{func_name}")
        except Exception as e:  # pragma: no cover - defensive
            last_err = e
    raise RuntimeError(f"Failed to import callable from {candidates}: {last_err}")


def _compute_exposure_core(
    btc_closed: pd.DataFrame,
) -> pd.DataFrame:
    generate_intent = _load_callable(CORE_MODULE_CANDIDATES, CORE_FUNC)
    df = btc_closed.copy()
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["Timestamp"]).sort_values("Timestamp").reset_index(drop=True)
    df = df.drop_duplicates(subset=["Timestamp"], keep="last").reset_index(drop=True)

    if len(df) < ARGUS_LOOKBACK + 2:
        raise ValueError(f"insufficient_bars_for_core:{len(df)} (need >= {ARGUS_LOOKBACK + 2})")

    total = len(df) - ARGUS_LOOKBACK
    progress_step = max(1, total * PROGRESS_INTERVAL_PCT // 100)
    print(f"  Core: computing exposure ({total} bars, lookback={ARGUS_LOOKBACK}) ...", flush=True)

    records: List[Dict[str, Any]] = []
    for idx, i in enumerate(range(ARGUS_LOOKBACK, len(df))):
        if idx > 0 and idx % progress_step == 0:
            print(f"  Core: {idx}/{total} bars ({100 * idx // total}%)", flush=True)
        sub = df.iloc[: i + 1].copy()
        ctx: Dict[str, Any] = {"mode": "research", "product_id": "BTC-USD"}
        out = generate_intent(sub, ctx, closed_only=True)
        if not isinstance(out, dict):
            raise RuntimeError("generate_intent must return dict")
        ts = sub["Timestamp"].iloc[-1]
        x_core = float(out.get("desired_exposure_frac", 0.0) or 0.0)
        records.append({"Timestamp": ts, "x_core": x_core})

    res = pd.DataFrame(records)
    res["Timestamp"] = pd.to_datetime(res["Timestamp"], utc=True, errors="coerce")
    res = res.dropna(subset=["Timestamp"]).sort_values("Timestamp").reset_index(drop=True)
    res = res.drop_duplicates(subset=["Timestamp"], keep="last").reset_index(drop=True)
    res["x_core"] = pd.to_numeric(res["x_core"], errors="coerce").fillna(0.0)
    print(f"  Core: done ({len(res)} bars).", flush=True)
    return res


def _compute_exposure_sleeve2a(
    btc_closed: pd.DataFrame,
) -> pd.DataFrame:
    generate_intent = _load_callable(SLEEVE2A_MODULE_CANDIDATES, SLEEVE2A_FUNC)
    df = btc_closed.copy()
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["Timestamp"]).sort_values("Timestamp").reset_index(drop=True)
    df = df.drop_duplicates(subset=["Timestamp"], keep="last").reset_index(drop=True)

    # Sleeve2A starts at same bar as Core so timeline aligns with harness
    if len(df) < ARGUS_LOOKBACK + 2:
        raise ValueError(f"insufficient_bars_for_sleeve2a:{len(df)} (need >= {ARGUS_LOOKBACK + 2})")

    total = len(df) - ARGUS_LOOKBACK
    progress_step = max(1, total * PROGRESS_INTERVAL_PCT // 100)
    print(f"  Sleeve2A: computing exposure ({total} bars, lookback={ARGUS_LOOKBACK}) ...", flush=True)

    state: Dict[str, Any] = {}
    records: List[Dict[str, Any]] = []
    for idx, i in enumerate(range(ARGUS_LOOKBACK, len(df))):
        if idx > 0 and idx % progress_step == 0:
            print(f"  Sleeve2A: {idx}/{total} bars ({100 * idx // total}%)", flush=True)
        sub = df.iloc[: i + 1].copy()
        out = generate_intent(sub, state, closed_only=True)
        if not isinstance(out, dict):
            raise RuntimeError("generate_intent must return dict")
        ts = sub["Timestamp"].iloc[-1]
        x_sleeve = float(out.get("desired_exposure_frac", 0.0) or 0.0)
        records.append({"Timestamp": ts, "x_sleeve2a": x_sleeve})

    res = pd.DataFrame(records)
    res["Timestamp"] = pd.to_datetime(res["Timestamp"], utc=True, errors="coerce")
    res = res.dropna(subset=["Timestamp"]).sort_values("Timestamp").reset_index(drop=True)
    res = res.drop_duplicates(subset=["Timestamp"], keep="last").reset_index(drop=True)
    res["x_sleeve2a"] = pd.to_numeric(res["x_sleeve2a"], errors="coerce").fillna(0.0)
    print(f"  Sleeve2A: done ({len(res)} bars).", flush=True)
    return res


# ---------------------------------------------------------------------
# Turnover, costs, BTC-only simulator
# ---------------------------------------------------------------------


def _compute_turnover_btc(w_btc: np.ndarray, w_cash: np.ndarray) -> np.ndarray:
    dw_btc = np.abs(np.diff(w_btc, prepend=w_btc[0]))
    dw_cash = np.abs(np.diff(w_cash, prepend=w_cash[0]))
    turnover = 0.5 * (dw_btc + dw_cash)
    return turnover


def _compute_cost(turnover_t: np.ndarray, mode: str, fee_bps: float, slippage_bps: float) -> np.ndarray:
    if mode == "gross":
        return np.zeros_like(turnover_t, dtype=float)
    drag_bps = float(fee_bps) + float(slippage_bps)
    return turnover_t * (drag_bps / 10000.0)


def _simulate_btc_only(
    timeline_prices: pd.DataFrame,
    w_btc_t: np.ndarray,
    mode: str,
    fee_bps: float,
    slippage_bps: float,
    initial_equity: float = 10000.0,
) -> pd.DataFrame:
    df = timeline_prices.copy()
    btc = df["btc_close"].to_numpy(dtype=float)

    # Next-bar returns t -> t+1
    r_btc_next = np.full(len(df), np.nan, dtype=float)
    r_btc_next[:-1] = (btc[1:] / btc[:-1]) - 1.0

    w_cash_t = 1.0 - w_btc_t
    port_ret_gross_next = w_btc_t * r_btc_next

    turnover_t = _compute_turnover_btc(w_btc_t, w_cash_t)
    cost_next = _compute_cost(turnover_t, mode=mode, fee_bps=fee_bps, slippage_bps=slippage_bps)
    port_ret_net_next = port_ret_gross_next - cost_next

    equity_index = np.full(len(df), np.nan, dtype=float)
    equity_index[0] = 1.0
    for i in range(len(df) - 1):
        r = port_ret_net_next[i]
        if np.isnan(r):
            equity_index[i + 1] = equity_index[i]
        else:
            equity_index[i + 1] = equity_index[i] * (1.0 + r)
    equity_usd = equity_index * float(initial_equity)

    out = pd.DataFrame(
        {
            "Timestamp": pd.to_datetime(df["Timestamp"], utc=True),
            "btc_close": btc,
            "w_btc": w_btc_t,
            "w_cash": w_cash_t,
            "gross_exposure": w_btc_t,
            "turnover": turnover_t,
            "r_btc_next": r_btc_next,
            "port_ret_gross_next": port_ret_gross_next,
            "cost_next": cost_next,
            "port_ret_net_next": port_ret_net_next,
            "equity_index": equity_index,
            "equity_usd": equity_usd,
        }
    )
    return out


# ---------------------------------------------------------------------
# Metrics (copied from portfolio_geometry_validation, ETH dropped)
# ---------------------------------------------------------------------


def _max_drawdown(equity: np.ndarray) -> float:
    peak = np.maximum.accumulate(equity)
    dd = (equity / peak) - 1.0
    return float(np.nanmin(dd))


def _ulcer_index(equity: np.ndarray) -> float:
    peak = np.maximum.accumulate(equity)
    dd_pct = (equity / peak - 1.0) * 100.0
    dd_pct = np.minimum(dd_pct, 0.0)
    ui = np.sqrt(np.nanmean(dd_pct**2))
    return float(ui)


def _time_to_recovery_bars(equity: np.ndarray) -> int:
    peak = np.maximum.accumulate(equity)
    max_ttr = 0
    i = 0
    n = len(equity)
    while i < n:
        if equity[i] >= peak[i] - 1e-12:
            target = equity[i]
            j = i + 1
            while j < n and equity[j] < target - 1e-12:
                j += 1
            if j < n:
                max_ttr = max(max_ttr, j - i)
            else:
                max_ttr = max(max_ttr, (n - 1) - i)
        i += 1
    return int(max_ttr)


def _sortino(returns: np.ndarray, periods_per_year: float) -> float:
    r = returns.copy()
    r = r[~np.isnan(r)]
    if len(r) < 5:
        return float("nan")
    mean = np.mean(r)
    downside = r[r < 0]
    if len(downside) == 0:
        return float("inf")
    downside_dev = np.sqrt(np.mean(downside**2))
    if downside_dev <= 0:
        return float("nan")
    return float((mean / downside_dev) * np.sqrt(periods_per_year))


def _cagr(equity: np.ndarray, years: float) -> float:
    if years <= 0:
        return float("nan")
    start = equity[0]
    end = equity[-1]
    if start <= 0 or end <= 0:
        return float("nan")
    return float((end / start) ** (1.0 / years) - 1.0)


def _compute_metrics(sim: pd.DataFrame, start: str, end: str) -> Dict[str, float]:
    ts = pd.to_datetime(sim["Timestamp"], utc=True)
    start_utc = pd.Timestamp(start, tz="UTC")
    end_utc = pd.Timestamp(end, tz="UTC")
    w = sim[(ts >= start_utc) & (ts <= end_utc)].copy()
    if len(w) < 10:
        return {
            "CAGR": float("nan"),
            "MaxDD": float("nan"),
            "Calmar": float("nan"),
            "Sortino": float("nan"),
            "UlcerIndex": float("nan"),
            "TimeToRecoveryBars": float("nan"),
            "AvgGrossExposure": float("nan"),
            "Turnover": float("nan"),
            "period_days": float("nan"),
            "total_return_pct": float("nan"),
        }

    eq = w["equity_index"].to_numpy(dtype=float)
    rets = w["port_ret_net_next"].to_numpy(dtype=float)

    t_min = pd.to_datetime(w["Timestamp"].min(), utc=True)
    t_max = pd.to_datetime(w["Timestamp"].max(), utc=True)
    period_seconds = (t_max - t_min).total_seconds()
    period_days = period_seconds / 86400.0
    years = period_days / 365.25 if period_days > 0 else float("nan")
    n_bars = len(w) - 1
    periods_per_year = (n_bars / years) if years and years > 0 else (365.25 * 24.0)

    cagr_v = _cagr(eq, years=years)
    maxdd_v = _max_drawdown(eq)
    calmar_v = float("nan") if (maxdd_v >= 0 or np.isnan(maxdd_v) or abs(maxdd_v) < 1e-12) else (cagr_v / abs(maxdd_v))
    sortino_v = _sortino(rets, periods_per_year=periods_per_year)
    ulcer_v = _ulcer_index(eq)
    ttr_bars = _time_to_recovery_bars(eq)

    avg_gross = float(np.nanmean(w["gross_exposure"].to_numpy(dtype=float)))
    turnover_mean = float(np.nanmean(w["turnover"].to_numpy(dtype=float)))

    start_eq = eq[0]
    end_eq = eq[-1]
    total_return_pct = float((end_eq / start_eq - 1.0) * 100.0) if start_eq and start_eq > 0 else float("nan")

    return {
        "CAGR": cagr_v,
        "MaxDD": maxdd_v,
        "Calmar": calmar_v,
        "Sortino": sortino_v,
        "UlcerIndex": ulcer_v,
        "TimeToRecoveryBars": float(ttr_bars),
        "AvgGrossExposure": avg_gross,
        "Turnover": turnover_mean,
        "period_days": float(period_days),
        "total_return_pct": total_return_pct,
    }


# ---------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------


def run(cfg: RunConfig) -> pd.DataFrame:
    print("Loading BTC data (harness-style: no drop-last, lookback=%d) ..." % ARGUS_LOOKBACK, flush=True)
    btc = _load_btc_harness_style(cfg.btc_data_file)
    print(f"  BTC: {len(btc)} bars.", flush=True)

    # Strategy exposures on BTC-only series
    print("Computing Core exposure ...", flush=True)
    core = _compute_exposure_core(btc)
    print("Computing Sleeve2A exposure ...", flush=True)
    sleeve = _compute_exposure_sleeve2a(btc)

    # Align on Timestamp intersection (core and sleeve both have bars from ARGUS_LOOKBACK to end)
    merged = pd.merge(
        core[["Timestamp", "x_core"]],
        sleeve[["Timestamp", "x_sleeve2a"]],
        on="Timestamp",
        how="inner",
    ).sort_values("Timestamp").reset_index(drop=True)

    # Timeline prices: same bar range as harness (from lookback to end; no drop-last)
    btc_tl = btc.iloc[ARGUS_LOOKBACK:][["Timestamp", "Close"]].copy()
    btc_tl = btc_tl.rename(columns={"Close": "btc_close"}).reset_index(drop=True)
    if len(btc_tl) != len(merged):
        raise ValueError(
            f"Timeline length mismatch: btc_tl={len(btc_tl)} merged={len(merged)} (expected equal)"
        )
    if len(btc_tl) < 10:
        raise ValueError("Merged BTC timeline too small after alignment")

    rows: List[Dict[str, Any]] = []

    x_core = merged["x_core"].to_numpy(dtype=float)
    x_sleeve = merged["x_sleeve2a"].to_numpy(dtype=float)

    scenario_weights: Dict[str, np.ndarray] = {
        "CORE_ONLY": x_core,
        "SLEEVE2A_ONLY": x_sleeve,
        "BLEND_80_20": 0.8 * x_core + 0.2 * x_sleeve,
        "BLEND_70_30": 0.7 * x_core + 0.3 * x_sleeve,
        "BLEND_50_50": 0.5 * x_core + 0.5 * x_sleeve,
    }

    print(f"Simulating {len(SCENARIOS)} scenarios × {len(WINDOWS)} windows ...", flush=True)
    for scenario in SCENARIOS:
        print(f"  {scenario} ...", flush=True)
        w_btc = scenario_weights[scenario]
        sim = _simulate_btc_only(
            timeline_prices=btc_tl,
            w_btc_t=w_btc,
            mode=cfg.mode,
            fee_bps=cfg.fee_bps,
            slippage_bps=cfg.slippage_bps,
            initial_equity=cfg.initial_equity,
        )

        for window_name, start, end in WINDOWS:
            m = _compute_metrics(sim, start=start, end=end)
            ts = pd.to_datetime(sim["Timestamp"], utc=True)
            w = sim[(ts >= pd.Timestamp(start, tz="UTC")) & (ts <= pd.Timestamp(end, tz="UTC"))]
            final_equity_usd = float(w["equity_usd"].iloc[-1]) if len(w) > 0 else float("nan")

            crash_window_dd = m["MaxDD"] if window_name == "crash_window" else float("nan")
            post_crash_cagr = m["CAGR"] if window_name == "post_crash" else float("nan")

            rows.append(
                {
                    "scenario": scenario,
                    "window": window_name,
                    "cost_regime": cfg.cost_regime,
                    "mode": cfg.mode,
                    "fee_bps": cfg.fee_bps,
                    "slippage_bps": cfg.slippage_bps,
                    "CAGR": m["CAGR"],
                    "MaxDD": m["MaxDD"],
                    "Calmar": m["Calmar"],
                    "Sortino": m["Sortino"],
                    "UlcerIndex": m["UlcerIndex"],
                    "TimeToRecoveryBars": m["TimeToRecoveryBars"],
                    "AvgGrossExposure": m["AvgGrossExposure"],
                    "Turnover": m["Turnover"],
                    "period_days": m["period_days"],
                    "total_return_pct": m["total_return_pct"],
                    "CrashWindowDD": crash_window_dd,
                    "PostCrashCAGR": post_crash_cagr,
                    "final_equity_usd": final_equity_usd,
                }
            )

    out = pd.DataFrame(rows)
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    out.to_csv(cfg.out_csv, index=False)
    print(f"Wrote {cfg.out_csv}", flush=True)

    with pd.option_context("display.max_rows", 200, "display.max_columns", 40, "display.width", 200):
        print(out.to_string(index=False))

    return out


def main() -> None:
    ap = argparse.ArgumentParser(
        description="BTC-only strategy geometry: Core vs Sleeve2A and static blends.",
    )
    ap.add_argument(
        "--mode",
        choices=["net", "gross"],
        default="net",
        help="net includes fees+slippage; gross disables costs",
    )
    ap.add_argument(
        "--btc_data_file",
        type=str,
        required=True,
        help="Path to BTC price CSV (mandatory)",
    )
    ap.add_argument(
        "--cost_regime",
        choices=[COST_REGIME_RETAIL_LAUNCH, COST_REGIME_PRO_TARGET, COST_REGIME_INSTITUTIONAL, COST_REGIME_CUSTOM],
        default=COST_REGIME_CUSTOM,
        help=(
            "Cost regime: retail_launch (120/10), pro_target (10/5), institutional (2/2), "
            "or custom (use --fee_bps/--slippage_bps or defaults 10/5)."
        ),
    )
    ap.add_argument(
        "--fee_bps",
        type=float,
        default=None,
        help="Fee in bps (overrides cost_regime default when set). With cost_regime=custom, defaults to 10.",
    )
    ap.add_argument(
        "--slippage_bps",
        type=float,
        default=None,
        help="Slippage in bps (overrides cost_regime default when set). With cost_regime=custom, defaults to 5.",
    )
    ap.add_argument(
        "--out_dir",
        type=str,
        default=str(REPO_ROOT / "research" / "experiments" / "output"),
    )
    ap.add_argument(
        "--out_csv",
        type=str,
        default=str(
            REPO_ROOT / "research" / "experiments" / "output" / "btc_strategy_geometry_validation.csv"
        ),
        help="Output CSV path.",
    )
    ap.add_argument(
        "--initial_equity",
        type=float,
        default=10000.0,
        help="Initial equity in USD; equity_usd = equity_index * this (default 10000).",
    )
    args = ap.parse_args()

    btc_path = Path(args.btc_data_file)
    if not btc_path.is_absolute():
        btc_path = (REPO_ROOT / args.btc_data_file).resolve()
    if not btc_path.exists():
        raise ValueError(f"BTC data file not found: {btc_path}")

    fee_bps, slippage_bps = resolve_cost_params(
        cost_regime=args.cost_regime,
        fee_bps_cli=args.fee_bps,
        slippage_bps_cli=args.slippage_bps,
        custom_fee_bps=DEFAULT_FEE_BPS,
        custom_slippage_bps=DEFAULT_SLIPPAGE_BPS,
    )

    out_dir = Path(args.out_dir)
    out_csv = Path(args.out_csv)

    cfg = RunConfig(
        mode=args.mode,
        fee_bps=fee_bps,
        slippage_bps=slippage_bps,
        cost_regime=args.cost_regime,
        out_dir=out_dir,
        out_csv=out_csv,
        btc_data_file=btc_path,
        initial_equity=float(args.initial_equity),
    )
    run(cfg)


if __name__ == "__main__":
    main()

