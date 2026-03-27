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
- Cost regimes: retail_launch (120/10 bps), pro_target (10/5), institutional (2/2) for friction modeling.
  Use --cost_regime to pick one; CLI --fee_bps/--slippage_bps override regime defaults when provided.

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
_EXPERIMENTS = REPO_ROOT / "research" / "experiments"
if str(_RUNTIME_ARGUS) not in sys.path:
    sys.path.insert(0, str(_RUNTIME_ARGUS))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(_EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(_EXPERIMENTS))

from cost_regimes import (
    COST_REGIME_CUSTOM,
    COST_REGIME_INSTITUTIONAL,
    COST_REGIME_PRO_TARGET,
    COST_REGIME_RETAIL_LAUNCH,
    resolve_cost_params,
)
from moonwire_loader import (
    config_from_env as moonwire_config_from_env,
    get_series_for_timeline as moonwire_get_series_for_timeline,
    load_feed as moonwire_load_feed,
)

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
    "BTC_MOONWIRE_ONLY",
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

# Alignment with harness backtest_runner (closed-bar t→t+1 semantics)
# Harness first row = bar lookback (default 200); geometry core starts at CORE_MIN_BARS (100).
# For BTC_CORE_ONLY trace we use BTC-only timeline and slice from (HARNESS_LOOKBACK - CORE_MIN_BARS)
# so the first trace row matches the harness first row (same bar set, no merge/slicing mismatch).
HARNESS_LOOKBACK = 200
CORE_MIN_BARS = 100

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
    cost_regime: str  # retail_launch | pro_target | institutional | custom
    out_dir: Path
    out_csv: Path
    env_file: Optional[Path]  # optional .env path for deterministic Core params
    btc_data_file: Path
    eth_data_file: Path
    debug_trace_max_bars: Optional[int] = None  # limit timeline for fast trace (e.g. 2000)
    max_bars: Optional[int] = None  # fast run: truncate to first N closed bars BEFORE core computation
    start_date: Optional[str] = None  # optional YYYY-MM-DD: slice merged timeline to ts >= start (before core)
    end_date: Optional[str] = None  # optional YYYY-MM-DD: slice merged timeline to ts <= end (before core)
    initial_equity: float = 10000.0  # USD; geometry uses equity_index (1.0) internally, equity_usd = index * this
    output_suffix: Optional[str] = None  # e.g. "retail_launch" for batch; used in manifest + trace paths


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


def _load_btc_for_harness_trace(btc_data_file: Path, *, lookback: int, max_trading_bars: Optional[int]) -> pd.DataFrame:
    """
    Load BTC for trace so the bar set and OHLCV format match the harness exactly.
    Mirrors load_flight_recorder: no drop-last, require Open/High/Low/Close (numeric),
    no duplicate-timestamp drop, cap at lookback + max_trading_bars.
    Regime engine needs full OHLCV; otherwise classify_regime can fail or return CHOP.
    """
    raw = _read_price_csv_from_path(btc_data_file)
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
    if len(df) < lookback + 2:
        raise ValueError(f"BTC has {len(df)} rows; need >= {lookback + 2} for trace")
    if max_trading_bars is not None and max_trading_bars > 0:
        cap = lookback + max_trading_bars
        if len(df) > cap:
            df = df.iloc[:cap].reset_index(drop=True)
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


def _slice_df_to_timeline_window(
    asset_df: pd.DataFrame,
    timeline: pd.DataFrame,
    lookback: int,
) -> pd.DataFrame:
    """
    Slice asset_df to the minimal window needed for core/sim: lookback bars before
    the first timeline timestamp, then all bars through the last timeline timestamp.
    Ensures compute_core_series receives only the rows required for the chosen timeline.
    """
    if timeline.empty or len(asset_df) < lookback + 1:
        return asset_df
    first_ts = pd.to_datetime(timeline["Timestamp"].iloc[0], utc=True)
    last_ts = pd.to_datetime(timeline["Timestamp"].iloc[-1], utc=True)
    ts = pd.to_datetime(asset_df["Timestamp"], utc=True)
    # Integer positions: start = first row >= first_ts; end = one past last row <= last_ts
    pos0 = ts.searchsorted(first_ts, side="left")
    pos_end = ts.searchsorted(last_ts, side="right")
    start = max(0, pos0 - lookback)
    end = pos_end
    if end <= start:
        return asset_df
    return asset_df.iloc[start:end].reset_index(drop=True)


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


def _log_run_df_bars(
    product_id: str,
    run_df_bars: int,
    sliced: bool,
    merged_bars: int,
) -> None:
    """Guard log before core compute: RUN DF BARS and optional assert when slice is active."""
    print(f"  RUN DF BARS: {product_id} df={run_df_bars} (sliced={sliced}) merged_timeline={merged_bars}", flush=True)
    if sliced and merged_bars > 0:
        # Core output bars = run_df_bars - CORE_MIN_BARS; should align to merged_timeline
        expected_core_bars = run_df_bars - CORE_MIN_BARS
        assert expected_core_bars <= merged_bars + 2, (
            f"Run sliced: {product_id} core would produce {expected_core_bars} bars but merged has {merged_bars}; "
            "run df must match slice."
        )


def _compute_core_series(
    product_id: str,
    df_closed: pd.DataFrame,
    *,
    match_harness_bar_set: bool = False,
) -> pd.DataFrame:
    """
    Uses existing Layer 1 + Layer 2 Core: research.strategies.sg_core_exposure_v2.generate_intent.
    Bar-by-bar with growing history through bar close t only (inclusive). Builds:
      - Timestamp (bar close time t)
      - x_core = desired_exposure_frac
      - for BTC only: btc_macro_is_bear = not meta["macro"]["macro_bull"] (required)

    When match_harness_bar_set=True (trace only), skip drop_duplicates so row indices and
    bar set match the harness (which does not drop duplicate timestamps).
    """
    generate_intent = _load_core_callable()

    if "Timestamp" not in df_closed.columns:
        raise ValueError("df_closed must have Timestamp column")

    df_closed = df_closed.copy()
    df_closed["Timestamp"] = pd.to_datetime(df_closed["Timestamp"], utc=True, errors="coerce")
    df_closed = df_closed.dropna(subset=["Timestamp"]).sort_values("Timestamp").reset_index(drop=True)
    if not match_harness_bar_set:
        df_closed = df_closed.drop_duplicates(subset=["Timestamp"], keep="last").reset_index(drop=True)

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
    df = df.dropna(subset=["Timestamp"]).sort_values("Timestamp").reset_index(drop=True)
    if not match_harness_bar_set:
        df = df.drop_duplicates(subset=["Timestamp"], keep="last").reset_index(drop=True)
    df["x_core"] = pd.to_numeric(df["x_core"], errors="coerce").fillna(0.0)

    if "btc_macro_is_bear" in df.columns:
        if df["btc_macro_is_bear"].isna().any():
            raise RuntimeError("btc_macro_is_bear contains NaNs after Core extraction (should never happen).")

    print(f"  {product_id}: done.", flush=True)
    return df.reset_index(drop=True)


def _compute_moonwire_series_for_timeline(
    timeline: pd.DataFrame,
    signal_path: Path,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute MoonWire w_btc and prob_used for the merged timeline (Experiment 1).
    Uses moonwire_loader: reads JSONL feed, applies weight rule, STRICT_TS_MATCH for missing bars.
    Returns (w_btc, prob_used) arrays of length len(timeline). Weights at bar t apply to return t->t+1.
    """
    feed = moonwire_load_feed(signal_path)
    config = moonwire_config_from_env()
    ts_utc = pd.to_datetime(timeline["Timestamp"], utc=True)
    bar_timestamps_unix = np.array([int(t.timestamp()) for t in ts_utc], dtype=float)
    print("  BTC (MoonWire): computing series (%d bars) ..." % len(timeline), flush=True)
    w_btc, prob_used = moonwire_get_series_for_timeline(bar_timestamps_unix, feed, config)
    print("  BTC (MoonWire): done.", flush=True)
    return w_btc, prob_used


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
    x_btc_moonwire: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns (w_btc, w_eth, w_cash) arrays, where weights are computed at time t.
    Static splits apply on top of Core exposure (no forced exposure).
    For BTC_MOONWIRE_ONLY, x_btc_moonwire must be provided (MoonWire feed-driven exposure).
    """
    if scenario not in SCENARIOS:
        raise ValueError(f"Unknown scenario: {scenario}")

    if scenario == "BTC_CORE_ONLY":
        w_btc = 1.0 * x_btc
        w_eth = 0.0 * x_eth
    elif scenario == "BTC_MOONWIRE_ONLY":
        if x_btc_moonwire is None:
            raise RuntimeError("BTC_MOONWIRE_ONLY requires x_btc_moonwire; ensure MOONWIRE_SIGNAL_FILE is set.")
        w_btc = 1.0 * x_btc_moonwire
        w_eth = np.zeros_like(w_btc)
        gross = w_btc + w_eth
        w_cash = 1.0 - np.abs(w_btc)  # no leverage
        return w_btc, w_eth, w_cash
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
    initial_equity: float = 10000.0,
) -> pd.DataFrame:
    """
    Determinism:
    - weights computed at bar close t apply to return t -> t+1

    Equity: normalized curve starts at 1.0 (equity_index). equity_usd = equity_index * initial_equity.
    Metrics are computed on equity_index (scale-invariant); equity_usd is for display/cross-check with harness.
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

    # Equity curve (normalized index starting at 1.0): E(t+1)=E(t)*(1+R_net_{t→t+1})
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
            "equity_index": equity_index,
            "equity_usd": equity_usd,
        }
    )
    return out


# ---------------------------------------------------------------------
# Metrics (standardized)
# All metrics use equity_index (scale-invariant). equity_usd is for display/cross-check only.
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
            "period_days": float("nan"),
            "total_return_pct": float("nan"),
        }

    # Metrics use equity_index (scale-invariant); no absolute-dollar metrics here
    eq = w["equity_index"].to_numpy(dtype=float)
    rets = w["port_ret_net_next"].to_numpy(dtype=float)

    # Period from actual timestamps (so annualized metrics use slice/window duration, not full-dataset years)
    t_min = pd.to_datetime(w["Timestamp"].min(), utc=True)
    t_max = pd.to_datetime(w["Timestamp"].max(), utc=True)
    period_seconds = (t_max - t_min).total_seconds()
    period_days = period_seconds / 86400.0
    years = period_days / 365.25 if period_days > 0 else float("nan")
    # Periods per year for Sortino: from bar count and period length (hourly => 365.25*24)
    n_bars = len(w) - 1  # number of complete return periods
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
def _window_overlaps_run(
    window_start: str,
    window_end: str,
    run_start_utc: pd.Timestamp,
    run_end_utc: pd.Timestamp,
) -> bool:
    """True if the run's date range [run_start_utc, run_end_utc] overlaps the window [window_start, window_end]."""
    w_start = pd.Timestamp(window_start, tz="UTC")
    w_end = pd.Timestamp(window_end, tz="UTC")
    return run_start_utc <= w_end and run_end_utc >= w_start


def _ensure_out_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def run(cfg: RunConfig) -> pd.DataFrame:
    # 1) Load + prep OHLCV (closed bars) from explicit paths
    btc_raw = _read_price_csv_from_path(cfg.btc_data_file)
    eth_raw = _read_price_csv_from_path(cfg.eth_data_file)
    btc = _prep_closed_bars(btc_raw)
    eth = _prep_closed_bars(eth_raw)

    # 2) Merge timeline (intersection only)
    timeline = _merge_prices(btc, eth)

    # Time window: slice to [start, end] AFTER merge, BEFORE core (targeted date ranges e.g. Dec 2025)
    start_date = getattr(cfg, "start_date", None)
    end_date = getattr(cfg, "end_date", None)
    if start_date is not None or end_date is not None:
        ts = pd.to_datetime(timeline["Timestamp"], utc=True)
        if start_date is not None:
            start_ts = pd.Timestamp(start_date, tz="UTC")
            timeline = timeline.loc[ts >= start_ts].reset_index(drop=True)
            ts = pd.to_datetime(timeline["Timestamp"], utc=True)
        if end_date is not None:
            end_ts = pd.Timestamp(end_date, tz="UTC")
            timeline = timeline.loc[ts <= end_ts].reset_index(drop=True)
        min_bars = CORE_MIN_BARS + 2
        if len(timeline) < min_bars:
            raise ValueError(
                f"WINDOW SLICE: after start={start_date!r} end={end_date!r} the timeline has {len(timeline)} bars; "
                f"need at least {min_bars} (lookback {CORE_MIN_BARS} + 2). Reduce the window or use full data."
            )
        btc = _slice_df_to_timeline_window(btc, timeline, CORE_MIN_BARS)
        eth = _slice_df_to_timeline_window(eth, timeline, CORE_MIN_BARS)
        print(f"WINDOW SLICE: start={start_date!r}, end={end_date!r}, bars={len(timeline)}")

    # Fast run: truncate to first N closed bars BEFORE any core computation (saves 45–60 min)
    max_bars = getattr(cfg, "max_bars", None)
    if max_bars is not None and max_bars > 0:
        n = min(len(timeline), max_bars)
        timeline = timeline.iloc[:n].reset_index(drop=True)
        btc = _slice_df_to_timeline_window(btc, timeline, CORE_MIN_BARS)
        eth = _slice_df_to_timeline_window(eth, timeline, CORE_MIN_BARS)
        print(f"FAST MODE: max_bars={n} (processing first {n} closed bars after intersection)")

    # Optional: limit bars for debug trace only (when --max_bars not set); does not slice asset dfs by count
    elif getattr(cfg, "debug_trace_max_bars", None) and cfg.debug_trace_max_bars > 0:
        n = min(len(timeline), cfg.debug_trace_max_bars)
        timeline = timeline.iloc[:n].reset_index(drop=True)
        last_ts = timeline["Timestamp"].iloc[-1]
        btc = btc[btc["Timestamp"] <= last_ts].reset_index(drop=True)
        eth = eth[eth["Timestamp"] <= last_ts].reset_index(drop=True)
        print(f"  Limited to first {n} bars (--debug_trace_max_bars); core/sim on subset only")

    # Canonical run dataframes: ALL downstream uses this slice only (no raw unsliced df references).
    run_sliced = (
        start_date is not None
        or end_date is not None
        or (max_bars is not None and max_bars > 0)
    )
    merged_df_run = timeline.copy()
    btc_df_run = btc.copy()
    eth_df_run = eth.copy()
    if run_sliced:
        print(f"RUN DF: merged={len(merged_df_run)} bars, btc={len(btc_df_run)} bars, eth={len(eth_df_run)} bars (sliced=True)")

    # Run date boundaries (for reporting and window overlap); use actual timestamps from merged run
    run_start_ts = pd.to_datetime(merged_df_run["Timestamp"].iloc[0], utc=True)
    run_end_ts = pd.to_datetime(merged_df_run["Timestamp"].iloc[-1], utc=True)
    run_start_date = run_start_ts.strftime("%Y-%m-%d")
    run_end_date = run_end_ts.strftime("%Y-%m-%d")
    run_bars = len(merged_df_run)

    # Resolve and print Core params from run dataframe (audit / reproducibility)
    resolved_params = _resolve_core_params_btc(btc_df_run)
    print("Resolved Core params:")
    print(json.dumps(resolved_params, indent=2))

    # 4) Compute Core series via existing implementation (no changes) — always on run dataframes only.
    _log_run_df_bars("BTC-USD", len(btc_df_run), run_sliced, len(merged_df_run))
    if run_sliced:
        assert len(btc_df_run) < 20000, (
            f"Run is sliced but btc_df_run has {len(btc_df_run)} bars; expected < 20k. "
            "Core must run on sliced window only."
        )
    btc_core = _compute_core_series("BTC-USD", btc_df_run)
    _log_run_df_bars("ETH-USD", len(eth_df_run), run_sliced, len(merged_df_run))
    if run_sliced:
        assert len(eth_df_run) < 20000, (
            f"Run is sliced but eth_df_run has {len(eth_df_run)} bars; expected < 20k. "
            "Core must run on sliced window only."
        )
    eth_core = _compute_core_series("ETH-USD", eth_df_run)

    # 5) Align core series to unified timeline (intersection only, no fill)
    merged_df_run, btc_aligned = _align_series_to_timeline(
        merged_df_run,
        btc_core,
        cols=["x_core"] + (["btc_macro_is_bear"] if "btc_macro_is_bear" in btc_core.columns else []),
    )
    merged_df_run, eth_aligned = _align_series_to_timeline(merged_df_run, eth_core, cols=["x_core"])

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

    # Scenarios to run: skip BTC_MOONWIRE_ONLY unless MOONWIRE_SIGNAL_FILE is set and file exists
    scenarios_to_run = [
        s for s in SCENARIOS
        if s != "BTC_MOONWIRE_ONLY" or os.environ.get("MOONWIRE_SIGNAL_FILE")
    ]
    if "BTC_MOONWIRE_ONLY" in SCENARIOS and "BTC_MOONWIRE_ONLY" not in scenarios_to_run:
        print("Skipping BTC_MOONWIRE_ONLY (MOONWIRE_SIGNAL_FILE not set)")

    # If MoonWire scenario is requested, ensure the signal file exists; otherwise skip with warning
    if "BTC_MOONWIRE_ONLY" in scenarios_to_run:
        signal_path = Path(os.environ["MOONWIRE_SIGNAL_FILE"])
        if not signal_path.is_absolute():
            signal_path = (REPO_ROOT / signal_path).resolve()
        if not signal_path.exists():
            scenarios_to_run = [s for s in scenarios_to_run if s != "BTC_MOONWIRE_ONLY"]
            print(
                "Skipping BTC_MOONWIRE_ONLY: signal file not found: %s\n"
                "  Ensure feed: python scripts/ensure_moonwire_signal_feed.py (or set MOONWIRE_SIGNAL_FILE to an existing path)."
                % signal_path
            )

    # MoonWire-only series (Experiment 1): loader reads JSONL, aligns to timeline, applies weight rule
    x_btc_moonwire: Optional[np.ndarray] = None
    prob_used_moonwire: Optional[np.ndarray] = None
    if "BTC_MOONWIRE_ONLY" in scenarios_to_run:
        x_btc_moonwire, prob_used_moonwire = _compute_moonwire_series_for_timeline(
            merged_df_run, signal_path
        )

    rows = []

    # 6) Scenario loop: build weights -> simulate -> window metrics
    for scenario in scenarios_to_run:
        w_btc, w_eth, w_cash = _build_weights(
            scenario,
            x_btc=x_btc,
            x_eth=x_eth,
            btc_macro_is_bear=btc_macro,
            x_btc_moonwire=x_btc_moonwire,
        )

        sim = _simulate(
            timeline_prices=merged_df_run,
            w_btc_t=w_btc,
            w_eth_t=w_eth,
            w_cash_t=w_cash,
            mode=cfg.mode,
            fee_bps=cfg.fee_bps,
            slippage_bps=cfg.slippage_bps,
            initial_equity=cfg.initial_equity,
        )

        # Per-bar trace export for BTC_CORE_ONLY (behavioral diff)
        # When run_sliced (--start/--end or --max_bars): use same run data only — export from sim (no second load/core).
        # When not sliced: match harness bar set (load BTC, compute core, cap at lookback+N).
        if scenario == "BTC_CORE_ONLY":
            debug_dir = REPO_ROOT / "debug"
            debug_dir.mkdir(parents=True, exist_ok=True)
            _trace_suffix = f"__{cfg.output_suffix}" if cfg.output_suffix else ""
            trace_path = debug_dir / f"geometry_btc_trace{_trace_suffix}.csv"
            if run_sliced:
                # Sliced run: sim is already on merged_df_run; export only the run window (no full-dataset compute).
                trace_df = pd.DataFrame({
                    "timestamp": sim["Timestamp"],
                    "close_price": sim["btc_close"],
                    "exposure": sim["gross_exposure"],
                    "next_bar_return": sim["r_btc_next"],
                    "portfolio_return": sim["port_ret_net_next"],
                    "equity_index": sim["equity_index"],
                    "equity_usd": sim["equity_usd"],
                    "desired_exposure_frac": sim["w_btc"],
                    "applied_exposure": sim["gross_exposure"],
                    "bar_return_px": sim["r_btc_next"],
                    "bar_return_applied": sim["port_ret_net_next"],
                    "fee_slippage_this_bar": sim["cost_next"],
                    "rebalanced": (sim["turnover"].to_numpy(dtype=float) > 1e-9),
                })
                trace_df["cost_regime"] = cfg.cost_regime
                trace_df.to_csv(trace_path, index=False)
                print(f"Trace written: {trace_path} ({len(trace_df)} rows, sliced run)")
            else:
                _trace_max_bars = cfg.max_bars if (getattr(cfg, "max_bars", None) is not None) else cfg.debug_trace_max_bars
                btc_trace = _load_btc_for_harness_trace(
                    cfg.btc_data_file,
                    lookback=HARNESS_LOOKBACK,
                    max_trading_bars=_trace_max_bars,
                )
                btc_core_trace = _compute_core_series(
                    "BTC-USD", btc_trace, match_harness_bar_set=True
                )
                timeline_btc_only = btc_trace[["Timestamp", "Close"]].copy()
                timeline_btc_only["btc_close"] = timeline_btc_only["Close"].astype(float)
                timeline_btc_only["eth_close"] = timeline_btc_only["Close"].astype(float)
                btc_core_cols = ["x_core"] + (["btc_macro_is_bear"] if "btc_macro_is_bear" in btc_core_trace.columns else [])
                timeline_btc_aligned, btc_core_btc_aligned = _align_series_to_timeline(
                    timeline_btc_only, btc_core_trace, cols=btc_core_cols
                )
                x_btc_trace = btc_core_btc_aligned["x_core"].to_numpy(dtype=float)
                w_eth_trace = np.zeros(len(x_btc_trace), dtype=float)
                w_cash_trace = 1.0 - x_btc_trace
                sim_trace = _simulate(
                    timeline_prices=timeline_btc_aligned,
                    w_btc_t=x_btc_trace,
                    w_eth_t=w_eth_trace,
                    w_cash_t=w_cash_trace,
                    mode=cfg.mode,
                    fee_bps=cfg.fee_bps,
                    slippage_bps=cfg.slippage_bps,
                    initial_equity=cfg.initial_equity,
                )
                trace_skip = HARNESS_LOOKBACK - CORE_MIN_BARS
                sim_trace = sim_trace.iloc[trace_skip:].reset_index(drop=True)
                trace_df = pd.DataFrame({
                    "timestamp": sim_trace["Timestamp"],
                    "close_price": sim_trace["btc_close"],
                    "exposure": sim_trace["gross_exposure"],
                    "next_bar_return": sim_trace["r_btc_next"],
                    "portfolio_return": sim_trace["port_ret_net_next"],
                    "equity_index": sim_trace["equity_index"],
                    "equity_usd": sim_trace["equity_usd"],
                    "desired_exposure_frac": sim_trace["w_btc"],
                    "applied_exposure": sim_trace["gross_exposure"],
                    "bar_return_px": sim_trace["r_btc_next"],
                    "bar_return_applied": sim_trace["port_ret_net_next"],
                    "fee_slippage_this_bar": sim_trace["cost_next"],
                    "rebalanced": (sim_trace["turnover"].to_numpy(dtype=float) > 1e-9),
                })
                trace_df["cost_regime"] = cfg.cost_regime
                trace_df.to_csv(trace_path, index=False)
                print(f"Trace written: {trace_path} ({len(trace_df)} rows, BTC loaded like harness: no drop-last, cap lookback+N)")

        # Per-bar trace for BTC_MOONWIRE_ONLY (Experiment 1): w_btc, w_cash, prob_used (or NaN when HOLD)
        if scenario == "BTC_MOONWIRE_ONLY":
            debug_dir = REPO_ROOT / "debug"
            debug_dir.mkdir(parents=True, exist_ok=True)
            _trace_suffix = f"__{cfg.output_suffix}" if cfg.output_suffix else ""
            trace_path_mw = debug_dir / f"geometry_btc_moonwire_trace{_trace_suffix}.csv"
            trace_df_mw = pd.DataFrame({
                "timestamp": sim["Timestamp"],
                "close": sim["btc_close"],
                "w_btc": sim["w_btc"],
                "w_cash": sim["w_cash"],
                "prob_used": prob_used_moonwire if prob_used_moonwire is not None else np.nan,
                "desired_exposure_frac": sim["w_btc"],
                "applied_exposure": sim["gross_exposure"],
                "bar_return_px": sim["r_btc_next"],
                "bar_return_applied": sim["port_ret_net_next"],
                "fee_slippage_this_bar": sim["cost_next"],
                "equity_index": sim["equity_index"],
                "equity_usd": sim["equity_usd"],
                "rebalanced": (sim["turnover"].to_numpy(dtype=float) > 1e-9),
            })
            trace_df_mw["cost_regime"] = cfg.cost_regime
            trace_df_mw.to_csv(trace_path_mw, index=False)
            print(f"Trace written: {trace_path_mw} ({len(trace_df_mw)} rows, BTC_MOONWIRE_ONLY)")

        # When sliced, only emit windows that overlap the run's date range; otherwise emit all windows
        windows_to_emit = WINDOWS
        if run_sliced:
            windows_to_emit = [
                (wn, ws, we)
                for wn, ws, we in WINDOWS
                if _window_overlaps_run(ws, we, run_start_ts, run_end_ts)
            ]

        for window_name, start, end in windows_to_emit:
            m = _compute_metrics(sim, start=start, end=end)
            # Final equity (USD) at end of window for cross-check with harness
            ts = pd.to_datetime(sim["Timestamp"], utc=True)
            w = sim[(ts >= pd.Timestamp(start, tz="UTC")) & (ts <= pd.Timestamp(end, tz="UTC"))]
            final_equity_usd = float(w["equity_usd"].iloc[-1]) if len(w) > 0 else float("nan")

            # Output start/end: slice boundaries when run is sliced, else window boundaries
            out_start = run_start_date if run_sliced else start
            out_end = run_end_date if run_sliced else end

            # Include crash-window DD and post-crash CAGR in appropriate window rows
            crash_window_dd = m["MaxDD"] if window_name == "crash_window" else float("nan")
            post_crash_cagr = m["CAGR"] if window_name == "post_crash" else float("nan")

            rows.append(
                {
                    "scenario": scenario,
                    "window": window_name,
                    "cost_regime": cfg.cost_regime,
                    "start": out_start,
                    "end": out_end,
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

    _ensure_out_dir(cfg.out_dir)
    out.to_csv(cfg.out_csv, index=False)

    # Run manifest for audit (includes slice boundaries and bar count when sliced)
    _manifest_suffix = f"__{cfg.output_suffix}" if cfg.output_suffix else ""
    manifest_path = cfg.out_dir / f"portfolio_geometry_run_manifest{_manifest_suffix}.json"
    manifest = {
        "cost_regime": cfg.cost_regime,
        "env_file": str(cfg.env_file) if cfg.env_file is not None else None,
        "btc_data_file": str(cfg.btc_data_file),
        "eth_data_file": str(cfg.eth_data_file),
        "fee_bps": cfg.fee_bps,
        "slippage_bps": cfg.slippage_bps,
        "mode": cfg.mode,
        "initial_equity": cfg.initial_equity,
        "max_bars": getattr(cfg, "max_bars", None),
        "start_date": getattr(cfg, "start_date", None),
        "end_date": getattr(cfg, "end_date", None),
        "resolved_core_params": resolved_params,
    }
    manifest["run_bars"] = run_bars
    manifest["run_start_date"] = run_start_date
    manifest["run_end_date"] = run_end_date
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    # Clean summary printout (numbers only)
    # (Formatting only; no interpretation)
    with pd.option_context("display.max_rows", 500, "display.max_columns", 50, "display.width", 240):
        print(out.to_string(index=False))

    return out


def main():
    ap = argparse.ArgumentParser(
        description="Portfolio geometry validation. Use --cost_regime for retail/pro/institutional friction; --run_all_cost_regimes runs all three."
    )
    ap.add_argument("--mode", choices=["net", "gross"], default="net", help="net includes fees+slippage; gross disables costs")
    ap.add_argument("--env_file", type=str, default=None, help="Optional .env path for deterministic Core params (load_dotenv with override=True)")
    ap.add_argument("--btc_data_file", type=str, required=True, help="Path to BTC price CSV (mandatory)")
    ap.add_argument("--eth_data_file", type=str, required=True, help="Path to ETH price CSV (mandatory)")
    ap.add_argument(
        "--cost_regime",
        choices=[COST_REGIME_RETAIL_LAUNCH, COST_REGIME_PRO_TARGET, COST_REGIME_INSTITUTIONAL, COST_REGIME_CUSTOM],
        default=COST_REGIME_CUSTOM,
        help="Cost regime: retail_launch (120/10 bps), pro_target (10/5), institutional (2/2), or custom (use --fee_bps/--slippage_bps or defaults 10/5). CLI --fee_bps/--slippage_bps override regime defaults.",
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
    ap.add_argument("--out_dir", type=str, default=str(REPO_ROOT / "research" / "experiments" / "output"))
    ap.add_argument(
        "--out_csv",
        type=str,
        default=str(REPO_ROOT / "research" / "experiments" / "output" / "portfolio_geometry_validation.csv"),
        help="Output CSV path; ignored when --run_all_cost_regimes (paths get regime suffix).",
    )
    ap.add_argument("--debug_trace_max_bars", type=int, default=None, help="Limit timeline to N bars for fast trace (e.g. 2000)")
    ap.add_argument("--max_bars", type=int, default=None, help="Fast run: use only first N closed bars (after intersection); truncates before core computation.")
    ap.add_argument("--start", type=str, default=None, metavar="YYYY-MM-DD", help="Slice merged timeline to bars on or after this date (before core computation).")
    ap.add_argument("--end", type=str, default=None, metavar="YYYY-MM-DD", help="Slice merged timeline to bars on or before this date (before core computation).")
    ap.add_argument("--initial_equity", type=float, default=10000.0, help="Initial equity in USD; equity_usd = equity_index * this (default 10000)")
    ap.add_argument(
        "--run_all_cost_regimes",
        action="store_true",
        help="Run experiment 3 times (retail_launch, pro_target, institutional) with distinct output files; print BTC_CORE_ONLY full_cycle summary table.",
    )
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

    env_file_path = _resolve_env_file(args.env_file)
    if env_file_path is None:
        print("No env file provided — using current process environment")

    if os.environ.get("ARGUS_FEE_BPS") is not None or os.environ.get("ARGUS_SLIPPAGE_BPS") is not None:
        print("NOTE: Ignoring ARGUS_FEE_BPS / ARGUS_SLIPPAGE_BPS env vars. Using CLI/cost_regime fee/slippage values.")

    if args.run_all_cost_regimes:
        _run_all_cost_regimes(args, btc_path, eth_path, env_file_path)
        return

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
        env_file=env_file_path,
        btc_data_file=btc_path,
        eth_data_file=eth_path,
        debug_trace_max_bars=args.debug_trace_max_bars,
        max_bars=args.max_bars,
        start_date=args.start,
        end_date=args.end,
        initial_equity=float(args.initial_equity),
        output_suffix=None,
    )
    run(cfg)


def _resolve_env_file(env_file_arg: Optional[str]) -> Optional[Path]:
    if env_file_arg is None:
        return None
    env_file_path = Path(env_file_arg)
    if not env_file_path.is_absolute():
        env_file_path = (REPO_ROOT / env_file_arg).resolve()
    if not env_file_path.exists():
        raise ValueError(f"Env file not found: {env_file_path}")
    if load_dotenv is None:
        raise ValueError("dotenv is not installed; pip install python-dotenv to use --env_file")
    load_dotenv(env_file_path, override=True)
    print(f"Loaded env file: {env_file_path}")
    return env_file_path


def _run_all_cost_regimes(
    args: argparse.Namespace,
    btc_path: Path,
    eth_path: Path,
    env_file_path: Optional[Path],
) -> None:
    """Run experiment for retail_launch, pro_target, institutional; distinct outputs; print summary table."""
    regimes = [COST_REGIME_RETAIL_LAUNCH, COST_REGIME_PRO_TARGET, COST_REGIME_INSTITUTIONAL]
    out_dir = Path(args.out_dir)
    base_csv = Path(args.out_csv)
    # e.g. output/portfolio_geometry_validation__retail_launch.csv
    base_stem = base_csv.stem
    base_parent = base_csv.parent

    summary_rows = []
    for cost_regime in regimes:
        fee_bps, slippage_bps = resolve_cost_params(
            cost_regime=cost_regime,
            fee_bps_cli=None,
            slippage_bps_cli=None,
        )
        out_csv = base_parent / f"{base_stem}__{cost_regime}.csv"
        cfg = RunConfig(
            mode=args.mode,
            fee_bps=fee_bps,
            slippage_bps=slippage_bps,
            cost_regime=cost_regime,
            out_dir=out_dir,
            out_csv=out_csv,
            env_file=env_file_path,
            btc_data_file=btc_path,
            eth_data_file=eth_path,
            debug_trace_max_bars=args.debug_trace_max_bars,
            max_bars=args.max_bars,
            start_date=args.start,
            end_date=args.end,
            initial_equity=float(args.initial_equity),
            output_suffix=cost_regime,
        )
        print(f"\n{'='*60}\nCost regime: {cost_regime} (fee_bps={fee_bps}, slippage_bps={slippage_bps})\n{'='*60}")
        out = run(cfg)
        # BTC_CORE_ONLY full_cycle row for summary
        subset = out[(out["scenario"] == "BTC_CORE_ONLY") & (out["window"] == "full_cycle")]
        if len(subset) == 1:
            row = subset.iloc[0]
            summary_rows.append({
                "cost_regime": cost_regime,
                "CAGR": row["CAGR"],
                "MaxDD": row["MaxDD"],
                "Calmar": row["Calmar"],
                "final_equity_usd": row["final_equity_usd"],
            })

    if summary_rows:
        print("\n" + "=" * 60)
        print("BTC_CORE_ONLY full_cycle — cost regime summary")
        print("=" * 60)
        summary_df = pd.DataFrame(summary_rows)
        with pd.option_context("display.max_columns", None, "display.width", 120):
            print(summary_df.to_string(index=False))
        print("=" * 60)


if __name__ == "__main__":
    main()