# research/experiments/portfolio_geometry_validation.py
"""
Portfolio Geometry Validation Phase (RESEARCH ONLY)

Objective:
Determine whether cross-asset allocation (BTC + ETH Core) improves portfolio-level Calmar
versus BTC Core alone, without materially worsening crash drawdown.

Hard Locks:
- No architectural refactors
- No new sleeves / indicators / optimization / parameter tuning
- Use existing Layer 1 and Layer 2 Core implementations for BTC and ETH
- Closed-bar determinism: decision at bar close t applies to return t -> t+1
- Merge discipline: intersection join on timestamps, no forward-fill
- Fees/slippage switch: default net ON (fee_bps=10, slippage_bps=5)

Outputs:
- One consolidated CSV: Scenario × Window rows with standardized metrics
- Clean summary printout (numbers only)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None  # type: ignore[misc, assignment]

# ---------------------------------------------------------------------
# Repo path bootstrap (research-only): Core lives under runtime/argus/research
# ---------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
_RUNTIME_ARGUS = REPO_ROOT / "runtime" / "argus"
if str(_RUNTIME_ARGUS) not in sys.path:
    sys.path.insert(0, str(_RUNTIME_ARGUS))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# ---------------------------------------------------------------------
# Canonical windows (MUST match existing Itera reports)
# ---------------------------------------------------------------------
FULL_START, FULL_END = "2019-01-01", "2025-12-30"
CRASH_START, CRASH_END = "2021-07-01", "2022-12-31"
POST_START, POST_END = "2023-01-01", "2025-12-30"

WINDOWS = [
    ("full_cycle", FULL_START, FULL_END),
    ("crash_window", CRASH_START, CRASH_END),
    ("post_crash", POST_START, POST_END),
]

# ---------------------------------------------------------------------
# Scenarios (hard locked)
# ---------------------------------------------------------------------
SCENARIOS = [
    "BTC_CORE_ONLY",
    "ETH_CORE_ONLY",
    "STATIC_80_20",
    "STATIC_70_30",
    "STATIC_50_50",
    "BTC_MACRO_BEAR_CASH__BULL_70_30",
]

# ---------------------------------------------------------------------
# Costs (default net ON)
# ---------------------------------------------------------------------
DEFAULT_FEE_BPS = 10
DEFAULT_SLIPPAGE_BPS = 5

# ---------------------------------------------------------------------
# Data (explicit paths only; no discovery/fallback)
# ---------------------------------------------------------------------
DEFAULT_RUNTIME_ARGUS_DIR = REPO_ROOT / "runtime" / "argus"

# ---------------------------------------------------------------------
# Core implementation (RESEARCH ONLY) — hardcoded to match BTC Core reports
# ---------------------------------------------------------------------
# Same module + callable as crash_window_report.py / backtest_runner.py:
#   research.strategies.sg_core_exposure_v2.generate_intent
# generate_intent returns a dict; we build Timestamp/x_core/btc_macro_is_bear via bar-by-bar adapter.
CORE_MODULE = CORE_MODULE_CANDIDATES = [
    "research.strategies.sg_core_exposure_v2",                 # expected when runtime/argus is on sys.path
    "runtime.argus.research.strategies.sg_core_exposure_v2",   # fallback if repo root import style is needed
]
CORE_FUNC = "generate_intent"


@dataclass(frozen=True)
class RunConfig:
    mode: str  # "net" or "gross"
    fee_bps: float
    slippage_bps: float
    out_dir: Path
    out_csv: Path
    env_file: Optional[Path]  # optional .env path for deterministic Core params
    btc_data_file: Path
    eth_data_file: Path
    debug_trace_max_bars: Optional[int] = None  # limit timeline for fast trace (e.g. 2000)


# ---------------------------------------------------------------------
# Loading + preprocessing (closed bars only)
# ---------------------------------------------------------------------
def _read_price_csv_from_path(path: Path) -> pd.DataFrame:
    """Load price CSV from explicit path. Raises ValueError if file not found."""
    if not path.exists():
        raise ValueError(f"Data file not found: {path}")
    return pd.read_csv(path)


def _prep_closed_bars(df: pd.DataFrame) -> pd.DataFrame:
    """
    Closed-bar only:
    - parse Timestamp as UTC
    - sort
    - drop duplicate timestamps
    - drop the last row to avoid using a possibly still-forming bar
    """
    if "Timestamp" not in df.columns:
        raise ValueError("Price DF missing Timestamp column")

    df = df.copy()
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["Timestamp"])
    df = df.sort_values("Timestamp")
    df = df.drop_duplicates(subset=["Timestamp"], keep="last")

    if len(df) < 3:
        raise ValueError("Not enough bars after cleaning")

    # Drop last bar to enforce closed-bar only determinism
    df = df.iloc[:-1].reset_index(drop=True)
    return df


def _merge_prices(df_btc: pd.DataFrame, df_eth: pd.DataFrame) -> pd.DataFrame:
    """
    Merge discipline:
    - intersection join on Timestamp
    - no forward-fill
    """
    if "Close" not in df_btc.columns or "Close" not in df_eth.columns:
        raise ValueError("Price DF missing Close column")

    a = df_btc[["Timestamp", "Close"]].rename(columns={"Close": "btc_close"})
    b = df_eth[["Timestamp", "Close"]].rename(columns={"Close": "eth_close"})
    merged = pd.merge(a, b, on="Timestamp", how="inner")
    merged = merged.sort_values("Timestamp").reset_index(drop=True)

    if merged.empty or len(merged) < 10:
        raise ValueError("Merged timeline too small after intersection join")

    return merged


# ---------------------------------------------------------------------
# Existing Core implementation adapter (no logic changes)
# ---------------------------------------------------------------------
def _load_core_callable():
    import importlib

    last_err = None
    for mod_name in CORE_MODULE_CANDIDATES:
        try:
            mod = importlib.import_module(mod_name)
            fn = getattr(mod, CORE_FUNC, None)
            if callable(fn):
                return fn
            last_err = RuntimeError(f"Core func not callable: {mod_name}.{CORE_FUNC}")
        except Exception as e:
            last_err = e

    raise RuntimeError(f"Failed to import Core callable from candidates {CORE_MODULE_CANDIDATES}: {last_err}")


def _resolve_core_params_btc(btc_prepped: pd.DataFrame) -> Dict[str, Any]:
    """
    Call generate_intent once on a small BTC slice to resolve Core params from env.
    Returns meta["params"] for audit and manifest. Uses at least MIN_BARS for history.
    """
    generate_intent = _load_core_callable()
    MIN_BARS = 100
    if len(btc_prepped) < (MIN_BARS + 2):
        raise ValueError(f"insufficient_btc_bars_for_resolve:{len(btc_prepped)} (need >= {MIN_BARS + 2})")
    sub = btc_prepped.iloc[: MIN_BARS + 1].copy()
    ctx: Dict = {"mode": "research", "product_id": "BTC-USD"}
    out = generate_intent(sub, ctx, closed_only=True)
    if not isinstance(out, dict):
        raise RuntimeError("generate_intent must return dict")
    meta = out.get("meta", {}) if isinstance(out.get("meta"), dict) else {}
    params = meta.get("params")
    if params is None:
        return {}
    return dict(params)


def _extract_macro_bull(out: Dict) -> Optional[bool]:
    """From generate_intent output: meta.macro.macro_bull -> btc_macro_is_bear = not macro_bull."""
    meta = out.get("meta", {}) if isinstance(out, dict) else {}
    macro = meta.get("macro", {}) if isinstance(meta, dict) else {}
    mb = macro.get("macro_bull", None) if isinstance(macro, dict) else None
    if mb is None:
        return None
    return bool(mb)


def _compute_core_series(product_id: str, df_closed: pd.DataFrame) -> pd.DataFrame:
    """
    Uses existing Layer 1 + Layer 2 Core: research.strategies.sg_core_exposure_v2.generate_intent.
    Bar-by-bar with growing history through bar close t only (inclusive). Builds:
      - Timestamp (bar close time t)
      - x_core = desired_exposure_frac
      - for BTC only: btc_macro_is_bear = not meta["macro"]["macro_bull"] (required)
    """
    generate_intent = _load_core_callable()

    if "Timestamp" not in df_closed.columns:
        raise ValueError("df_closed must have Timestamp column")

    # Defensive cleaning (df_closed should already be prepped)
    df_closed = df_closed.copy()
    df_closed["Timestamp"] = pd.to_datetime(df_closed["Timestamp"], utc=True, errors="coerce")
    df_closed = (
        df_closed.dropna(subset=["Timestamp"])
        .sort_values("Timestamp")
        .drop_duplicates(subset=["Timestamp"], keep="last")
        .reset_index(drop=True)
    )

    MIN_BARS = 100
    if len(df_closed) < (MIN_BARS + 2):
        raise ValueError(f"insufficient_bars_for_core:{product_id}:{len(df_closed)}")

    total = len(df_closed) - MIN_BARS
    progress_interval = max(1, total // 20)  # ~20 progress lines per asset
    print(f"  {product_id}: computing core series ({total} bars) ...", flush=True)

    records: List[Dict] = []
    for idx, i in enumerate(range(MIN_BARS, len(df_closed))):
        if idx > 0 and idx % progress_interval == 0:
            print(f"  {product_id}: {idx}/{total} bars ...", flush=True)
        # history through bar close t (inclusive)
        sub = df_closed.iloc[: i + 1].copy()

        ctx: Dict = {
            "mode": "research",
            "product_id": product_id,
        }

        out = generate_intent(sub, ctx, closed_only=True)
        if not isinstance(out, dict):
            raise RuntimeError("generate_intent must return dict")

        ts = sub["Timestamp"].iloc[-1]
        x_core = float(out.get("desired_exposure_frac", 0.0) or 0.0)

        row: Dict = {"Timestamp": ts, "x_core": x_core}

        if product_id.upper() == "BTC-USD":
            macro_bull = _extract_macro_bull(out)
            # When strategy doesn't emit macro (e.g. macro filter off, or non-TREND_UP regime),
            # treat as bull so btc_macro_is_bear = False; macro-bear scenario only applies where macro was computed.
            row["btc_macro_is_bear"] = (not bool(macro_bull)) if macro_bull is not None else False

        records.append(row)

    df = pd.DataFrame(records)
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], utc=True, errors="coerce")
    df = (
        df.dropna(subset=["Timestamp"])
        .sort_values("Timestamp")
        .drop_duplicates(subset=["Timestamp"], keep="last")
    )
    df["x_core"] = pd.to_numeric(df["x_core"], errors="coerce").fillna(0.0)

    if "btc_macro_is_bear" in df.columns:
        if df["btc_macro_is_bear"].isna().any():
            raise RuntimeError("btc_macro_is_bear contains NaNs after Core extraction (should never happen).")

    print(f"  {product_id}: done.", flush=True)
    return df.reset_index(drop=True)


def _align_series_to_timeline(timeline: pd.DataFrame, series: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """
    Align to merged timeline using strict intersection (no forward fill).
    """
    keep = ["Timestamp"] + cols
    s = series[keep].copy()
    aligned = pd.merge(timeline[["Timestamp"]], s, on="Timestamp", how="inner")
    aligned = aligned.sort_values("Timestamp").reset_index(drop=True)
    if len(aligned) != len(timeline):
        # Enforce merge discipline: single unified timeline means we must drop to intersection everywhere.
        # So we re-intersect timeline to aligned timestamps.
        timeline2 = pd.merge(timeline, aligned[["Timestamp"]], on="Timestamp", how="inner")
        timeline2 = timeline2.sort_values("Timestamp").reset_index(drop=True)
        aligned = (
            pd.merge(timeline2[["Timestamp"]], s, on="Timestamp", how="inner")
            .sort_values("Timestamp")
            .reset_index(drop=True)
        )
        return timeline2, aligned
    return timeline, aligned


# ---------------------------------------------------------------------
# Scenario weights (locked interpretation)
# ---------------------------------------------------------------------
def _build_weights(
    scenario: str,
    x_btc: np.ndarray,
    x_eth: np.ndarray,
    btc_macro_is_bear: Optional[np.ndarray],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns (w_btc, w_eth, w_cash) arrays, where weights are computed at time t.
    Static splits apply on top of Core exposure (no forced exposure).
    """
    if scenario not in SCENARIOS:
        raise ValueError(f"Unknown scenario: {scenario}")

    if scenario == "BTC_CORE_ONLY":
        w_btc = 1.0 * x_btc
        w_eth = 0.0 * x_eth
    elif scenario == "ETH_CORE_ONLY":
        w_btc = 0.0 * x_btc
        w_eth = 1.0 * x_eth
    elif scenario == "STATIC_80_20":
        w_btc = 0.8 * x_btc
        w_eth = 0.2 * x_eth
    elif scenario == "STATIC_70_30":
        w_btc = 0.7 * x_btc
        w_eth = 0.3 * x_eth
    elif scenario == "STATIC_50_50":
        w_btc = 0.5 * x_btc
        w_eth = 0.5 * x_eth
    elif scenario == "BTC_MACRO_BEAR_CASH__BULL_70_30":
        if btc_macro_is_bear is None:
            raise RuntimeError("Scenario requires BTC macro bear flag, but it was not provided by Core output.")
        bear = btc_macro_is_bear.astype(bool)
        w_btc = np.where(bear, 0.0, 0.7 * x_btc)
        w_eth = np.where(bear, 0.0, 0.3 * x_eth)
    else:
        raise ValueError(f"Unhandled scenario: {scenario}")

    gross = w_btc + w_eth
    w_cash = 1.0 - gross
    return w_btc, w_eth, w_cash


# ---------------------------------------------------------------------
# Turnover + costs (locked formulas)
# ---------------------------------------------------------------------
def _compute_turnover(w_btc: np.ndarray, w_eth: np.ndarray, w_cash: np.ndarray) -> np.ndarray:
    """
    turnover(t) = 0.5 * (|Δw_btc| + |Δw_eth| + |Δw_cash|)
    """
    dw_btc = np.abs(np.diff(w_btc, prepend=w_btc[0]))
    dw_eth = np.abs(np.diff(w_eth, prepend=w_eth[0]))
    dw_cash = np.abs(np.diff(w_cash, prepend=w_cash[0]))
    turnover = 0.5 * (dw_btc + dw_eth + dw_cash)
    return turnover


def _compute_cost(turnover_t: np.ndarray, mode: str, fee_bps: float, slippage_bps: float) -> np.ndarray:
    """
    cost_{t→t+1} = turnover(t) * (fee_bps + slippage_bps) / 10000   (net mode)
    cost = 0 (gross mode)
    """
    if mode == "gross":
        return np.zeros_like(turnover_t, dtype=float)
    drag_bps = float(fee_bps) + float(slippage_bps)
    return turnover_t * (drag_bps / 10000.0)


# ---------------------------------------------------------------------
# Simulator (explicit t -> t+1 shift)
# ---------------------------------------------------------------------
def _simulate(
    timeline_prices: pd.DataFrame,
    w_btc_t: np.ndarray,
    w_eth_t: np.ndarray,
    w_cash_t: np.ndarray,
    mode: str,
    fee_bps: float,
    slippage_bps: float,
) -> pd.DataFrame:
    """
    Determinism:
    - weights computed at bar close t apply to return t -> t+1
    """
    df = timeline_prices.copy()
    btc = df["btc_close"].to_numpy(dtype=float)
    eth = df["eth_close"].to_numpy(dtype=float)

    # Next-bar returns (t -> t+1), aligned to index t (last bar will have NaN)
    r_btc_next = np.full(len(df), np.nan, dtype=float)
    r_eth_next = np.full(len(df), np.nan, dtype=float)
    r_btc_next[:-1] = (btc[1:] / btc[:-1]) - 1.0
    r_eth_next[:-1] = (eth[1:] / eth[:-1]) - 1.0

    # Gross return for next bar uses weights at t
    port_ret_gross_next = w_btc_t * r_btc_next + w_eth_t * r_eth_next

    # Turnover computed at t; cost applied to t->t+1
    turnover_t = _compute_turnover(w_btc_t, w_eth_t, w_cash_t)
    cost_next = _compute_cost(turnover_t, mode=mode, fee_bps=fee_bps, slippage_bps=slippage_bps)

    port_ret_net_next = port_ret_gross_next - cost_next

    # Equity curve: E(t+1)=E(t)*(1+R_net_{t→t+1})
    equity = np.full(len(df), np.nan, dtype=float)
    equity[0] = 1.0
    for i in range(len(df) - 1):
        r = port_ret_net_next[i]
        if np.isnan(r):
            equity[i + 1] = equity[i]
        else:
            equity[i + 1] = equity[i] * (1.0 + r)

    out = pd.DataFrame(
        {
            "Timestamp": pd.to_datetime(df["Timestamp"], utc=True),
            "btc_close": btc,
            "eth_close": eth,
            "w_btc": w_btc_t,
            "w_eth": w_eth_t,
            "w_cash": w_cash_t,
            "gross_exposure": (w_btc_t + w_eth_t),
            "turnover": turnover_t,
            "r_btc_next": r_btc_next,
            "r_eth_next": r_eth_next,
            "port_ret_gross_next": port_ret_gross_next,
            "cost_next": cost_next,
            "port_ret_net_next": port_ret_net_next,
            "equity": equity,
        }
    )
    return out


# ---------------------------------------------------------------------
# Metrics (standardized)
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
    # For each peak point, measure bars until equity >= that peak again; take max.
    max_ttr = 0
    i = 0
    n = len(equity)
    while i < n:
        # find a new peak at i
        if equity[i] >= peak[i] - 1e-12:
            target = equity[i]
            j = i + 1
            while j < n and equity[j] < target - 1e-12:
                j += 1
            if j < n:
                max_ttr = max(max_ttr, j - i)
            else:
                # never recovered by end
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
    # Normalize to UTC so we never compare tz-naive and tz-aware (sim["Timestamp"] can be naive after .values)
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
        }

    eq = w["equity"].to_numpy(dtype=float)
    rets = w["port_ret_net_next"].to_numpy(dtype=float)

    # Hourly bars: approximate periods/year = 365.25*24
    periods_per_year = 365.25 * 24.0
    years = (len(w) / periods_per_year)

    cagr_v = _cagr(eq, years=years)
    maxdd_v = _max_drawdown(eq)
    calmar_v = float("nan") if (maxdd_v >= 0 or np.isnan(maxdd_v) or abs(maxdd_v) < 1e-12) else (cagr_v / abs(maxdd_v))
    sortino_v = _sortino(rets, periods_per_year=periods_per_year)
    ulcer_v = _ulcer_index(eq)
    ttr_bars = _time_to_recovery_bars(eq)

    avg_gross = float(np.nanmean(w["gross_exposure"].to_numpy(dtype=float)))
    turnover_mean = float(np.nanmean(w["turnover"].to_numpy(dtype=float)))

    return {
        "CAGR": cagr_v,
        "MaxDD": maxdd_v,
        "Calmar": calmar_v,
        "Sortino": sortino_v,
        "UlcerIndex": ulcer_v,
        "TimeToRecoveryBars": float(ttr_bars),
        "AvgGrossExposure": avg_gross,
        "Turnover": turnover_mean,
    }


# ---------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------
def _ensure_out_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def run(cfg: RunConfig) -> pd.DataFrame:
    # 1) Load + prep OHLCV (closed bars) from explicit paths
    btc_raw = _read_price_csv_from_path(cfg.btc_data_file)
    eth_raw = _read_price_csv_from_path(cfg.eth_data_file)
    btc = _prep_closed_bars(btc_raw)
    eth = _prep_closed_bars(eth_raw)

    # 2) Resolve and print Core params (audit / reproducibility)
    resolved_params = _resolve_core_params_btc(btc)
    print("Resolved Core params:")
    print(json.dumps(resolved_params, indent=2))

    # 3) Merge timeline (intersection only)
    timeline = _merge_prices(btc, eth)

    # Optional: limit bars early so core + sim run on subset only (debug trace)
    if getattr(cfg, "debug_trace_max_bars", None) and cfg.debug_trace_max_bars > 0:
        n = min(len(timeline), cfg.debug_trace_max_bars)
        timeline = timeline.iloc[:n].reset_index(drop=True)
        last_ts = timeline["Timestamp"].iloc[-1]
        btc = btc[btc["Timestamp"] <= last_ts].reset_index(drop=True)
        eth = eth[eth["Timestamp"] <= last_ts].reset_index(drop=True)
        print(f"  Limited to first {n} bars (--debug_trace_max_bars); core/sim on subset only")

    # 4) Compute Core series via existing implementation (no changes)
    btc_core = _compute_core_series("BTC-USD", btc)
    eth_core = _compute_core_series("ETH-USD", eth)

    # 5) Align core series to unified timeline (intersection only, no fill)
    timeline, btc_aligned = _align_series_to_timeline(
        timeline,
        btc_core,
        cols=["x_core"] + (["btc_macro_is_bear"] if "btc_macro_is_bear" in btc_core.columns else []),
    )
    timeline, eth_aligned = _align_series_to_timeline(timeline, eth_core, cols=["x_core"])

    # BTC macro bear flag (required for regime-conditioned scenario)
    btc_macro = None
    if "btc_macro_is_bear" in btc_aligned.columns:
        if btc_aligned["btc_macro_is_bear"].isna().any():
            raise RuntimeError("btc_macro_is_bear has NaNs after alignment; cannot run macro-bear scenario.")
        btc_macro = btc_aligned["btc_macro_is_bear"].astype(bool).to_numpy()
    else:
        btc_macro = None

    x_btc = btc_aligned["x_core"].to_numpy(dtype=float)
    x_eth = eth_aligned["x_core"].to_numpy(dtype=float)

    rows = []

    # 6) Scenario loop: build weights -> simulate -> window metrics
    for scenario in SCENARIOS:
        w_btc, w_eth, w_cash = _build_weights(scenario, x_btc=x_btc, x_eth=x_eth, btc_macro_is_bear=btc_macro)

        sim = _simulate(
            timeline_prices=timeline,
            w_btc_t=w_btc,
            w_eth_t=w_eth,
            w_cash_t=w_cash,
            mode=cfg.mode,
            fee_bps=cfg.fee_bps,
            slippage_bps=cfg.slippage_bps,
        )

        # Per-bar trace export for BTC_CORE_ONLY (behavioral diff)
        if scenario == "BTC_CORE_ONLY":
            debug_dir = REPO_ROOT / "debug"
            debug_dir.mkdir(parents=True, exist_ok=True)
            trace_path = debug_dir / "geometry_btc_trace.csv"
            trace_df = pd.DataFrame({
                "timestamp": sim["Timestamp"],
                "close_price": sim["btc_close"],
                "exposure": sim["gross_exposure"],
                "next_bar_return": sim["r_btc_next"],
                "portfolio_return": sim["port_ret_net_next"],
                "equity": sim["equity"],
            })
            trace_df.to_csv(trace_path, index=False)
            print(f"Trace written: {trace_path}")

        for window_name, start, end in WINDOWS:
            m = _compute_metrics(sim, start=start, end=end)

            # Include crash-window DD and post-crash CAGR in appropriate window rows
            crash_window_dd = m["MaxDD"] if window_name == "crash_window" else float("nan")
            post_crash_cagr = m["CAGR"] if window_name == "post_crash" else float("nan")

            rows.append(
                {
                    "scenario": scenario,
                    "window": window_name,
                    "start": start,
                    "end": end,
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
                    "CrashWindowDD": crash_window_dd,
                    "PostCrashCAGR": post_crash_cagr,
                }
            )

    out = pd.DataFrame(rows)

    _ensure_out_dir(cfg.out_dir)
    out.to_csv(cfg.out_csv, index=False)

    # Run manifest for audit
    manifest_path = cfg.out_dir / "portfolio_geometry_run_manifest.json"
    manifest = {
        "env_file": str(cfg.env_file) if cfg.env_file is not None else None,
        "btc_data_file": str(cfg.btc_data_file),
        "eth_data_file": str(cfg.eth_data_file),
        "fee_bps": cfg.fee_bps,
        "slippage_bps": cfg.slippage_bps,
        "mode": cfg.mode,
        "resolved_core_params": resolved_params,
    }
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    # Clean summary printout (numbers only)
    # (Formatting only; no interpretation)
    with pd.option_context("display.max_rows", 500, "display.max_columns", 50, "display.width", 240):
        print(out.to_string(index=False))

    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["net", "gross"], default="net", help="net includes fees+slippage; gross disables costs")
    ap.add_argument("--env_file", type=str, default=None, help="Optional .env path for deterministic Core params (load_dotenv with override=True)")
    ap.add_argument("--btc_data_file", type=str, required=True, help="Path to BTC price CSV (mandatory)")
    ap.add_argument("--eth_data_file", type=str, required=True, help="Path to ETH price CSV (mandatory)")
    ap.add_argument("--fee_bps", type=float, default=DEFAULT_FEE_BPS)
    ap.add_argument("--slippage_bps", type=float, default=DEFAULT_SLIPPAGE_BPS)
    ap.add_argument("--out_dir", type=str, default=str(REPO_ROOT / "research" / "experiments" / "output"))
    ap.add_argument(
        "--out_csv",
        type=str,
        default=str(REPO_ROOT / "research" / "experiments" / "output" / "portfolio_geometry_validation.csv"),
    )
    ap.add_argument("--debug_trace_max_bars", type=int, default=None, help="Limit timeline to N bars for fast trace (e.g. 2000)")
    args = ap.parse_args()

    # Resolve paths (allow relative to cwd or repo root)
    btc_path = Path(args.btc_data_file)
    if not btc_path.is_absolute():
        btc_path = (REPO_ROOT / args.btc_data_file).resolve()
    eth_path = Path(args.eth_data_file)
    if not eth_path.is_absolute():
        eth_path = (REPO_ROOT / args.eth_data_file).resolve()
    if not btc_path.exists():
        raise ValueError(f"BTC data file not found: {btc_path}")
    if not eth_path.exists():
        raise ValueError(f"ETH data file not found: {eth_path}")

    env_file_path: Optional[Path] = None
    if args.env_file is not None:
        env_file_path = Path(args.env_file)
        if not env_file_path.is_absolute():
            env_file_path = (REPO_ROOT / args.env_file).resolve()
        if not env_file_path.exists():
            raise ValueError(f"Env file not found: {env_file_path}")
        if load_dotenv is None:
            raise ValueError("dotenv is not installed; pip install python-dotenv to use --env_file")
        load_dotenv(env_file_path, override=True)
        print(f"Loaded env file: {env_file_path}")
    else:
        print("No env file provided — using current process environment")

    if os.environ.get("ARGUS_FEE_BPS") is not None or os.environ.get("ARGUS_SLIPPAGE_BPS") is not None:
        print("NOTE: Ignoring ARGUS_FEE_BPS / ARGUS_SLIPPAGE_BPS env vars. Using CLI fee/slippage values.")

    cfg = RunConfig(
        mode=args.mode,
        fee_bps=float(args.fee_bps),
        slippage_bps=float(args.slippage_bps),
        out_dir=Path(args.out_dir),
        out_csv=Path(args.out_csv),
        env_file=env_file_path,
        btc_data_file=btc_path,
        eth_data_file=eth_path,
        debug_trace_max_bars=args.debug_trace_max_bars,
    )
    run(cfg)


if __name__ == "__main__":
    main()