"""
Backtest Runner for Layer 2 Strategies
=======================================

Env-based injection:
    ARGUS_STRATEGY_MODULE = "research.strategies.sg_core_exposure_v1"
    ARGUS_STRATEGY_FUNC   = "generate_intent"

Data source:
    ARGUS_DATA_FILE = "./flight_recorder.csv" (default)

Run from repo root:
    $env:ARGUS_STRATEGY_MODULE="research.strategies.sg_core_exposure_v1"
    $env:ARGUS_STRATEGY_FUNC="generate_intent"
    python -c "import sys; sys.path.insert(0, r'./runtime/argus'); from research.harness.backtest_runner import main; main()"

Contract:
    - Strategy module must implement generate_intent(df, ctx, *, closed_only=True) -> dict
    - Dict keys: action, confidence, desired_exposure_frac, horizon_hours, reason, meta
    - Valid actions: ENTER_LONG, EXIT_LONG, HOLD, FLAT
"""

from __future__ import annotations

import os
import sys
import importlib
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ---------------------------
# Path setup
# ---------------------------

def _setup_path() -> Path:
    """Ensure runtime/argus is on sys.path. Returns argus_dir."""
    this_file = Path(__file__).resolve()
    argus_dir = this_file.parent.parent.parent

    if argus_dir.name == "argus" and (argus_dir / "research").exists():
        if str(argus_dir) not in sys.path:
            sys.path.insert(0, str(argus_dir))
        return argus_dir

    for candidate in [Path.cwd() / "runtime" / "argus", Path.cwd()]:
        if (candidate / "research").exists():
            if str(candidate) not in sys.path:
                sys.path.insert(0, str(candidate))
            return candidate

    return Path.cwd()


# ---------------------------
# Configuration from env
# ---------------------------

def _env_str(name: str, default: str) -> str:
    v = os.environ.get(name, "").strip()
    return v if v else default


def _env_int(name: str, default: int) -> int:
    v = os.environ.get(name, "").strip()
    if not v:
        return default
    try:
        return int(float(v))
    except Exception:
        return default


def _env_float(name: str, default: float) -> float:
    v = os.environ.get(name, "").strip()
    if not v:
        return default
    try:
        return float(v)
    except Exception:
        return default


# ---------------------------
# Data loading
# ---------------------------

def load_flight_recorder(csv_path: str) -> pd.DataFrame:
    """Load and validate OHLCV data."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Data file not found: {csv_path}")

    df = pd.read_csv(csv_path)

    if "Timestamp" not in df.columns:
        raise ValueError("CSV must contain a Timestamp column")

    df["Timestamp"] = pd.to_datetime(df["Timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["Timestamp"]).sort_values("Timestamp").reset_index(drop=True)

    for col in ("Open", "High", "Low", "Close"):
        if col not in df.columns:
            raise ValueError(f"CSV missing required OHLC column: {col}")
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if "Volume" in df.columns:
        df["Volume"] = pd.to_numeric(df["Volume"], errors="coerce")
    else:
        df["Volume"] = 0.0

    df = df.dropna(subset=["Open", "High", "Low", "Close"]).reset_index(drop=True)

    return df


# ---------------------------
# Strategy loader
# ---------------------------

def load_strategy_func(module_path: str, func_name: str) -> Callable:
    """
    Dynamically import strategy module and return the specified function.

    module_path: e.g., "research.strategies.sg_core_exposure_v1"
    func_name: e.g., "generate_intent"
    """
    mod = importlib.import_module(module_path)
    fn = getattr(mod, func_name, None)
    if fn is None:
        raise AttributeError(f"Module '{module_path}' has no attribute '{func_name}'")
    if not callable(fn):
        raise TypeError(f"'{module_path}.{func_name}' is not callable")
    return fn


# ---------------------------
# Action normalization
# ---------------------------

def _normalize_action(action: Optional[str]) -> str:
    """Normalize action string to canonical form."""
    if not action:
        return "HOLD"
    a = str(action).strip().upper()

    if a in ("ENTER_LONG", "ENTER", "BUY", "LONG"):
        return "ENTER"
    if a in ("EXIT_LONG", "EXIT", "SELL", "CLOSE"):
        return "EXIT"
    if a in ("FLAT", "HOLD", "NONE", "NOOP"):
        return "HOLD"

    return "HOLD"


# ---------------------------
# Metrics computation
# ---------------------------

def compute_sortino(returns: np.ndarray, risk_free: float = 0.0, ann_factor: float = 8760) -> float:
    """
    Compute annualized Sortino ratio.
    ann_factor=8760 for hourly data (365 * 24).
    """
    if len(returns) < 2:
        return 0.0

    excess = returns - risk_free / ann_factor
    neg_returns = excess[excess < 0]

    if len(neg_returns) == 0:
        return float("inf") if excess.mean() > 0 else 0.0

    downside_std = np.sqrt(np.mean(neg_returns ** 2))
    if downside_std < 1e-12:
        return float("inf") if excess.mean() > 0 else 0.0

    sortino = (excess.mean() / downside_std) * np.sqrt(ann_factor)
    return float(sortino)


def compute_calmar(cagr: float, max_dd: float) -> float:
    """Calmar ratio: CAGR / abs(max_drawdown)."""
    if abs(max_dd) < 1e-12:
        return float("inf") if cagr > 0 else 0.0
    return cagr / abs(max_dd)


def compute_drawdown_series(equity: np.ndarray) -> np.ndarray:
    """Compute drawdown series from equity curve."""
    peak = np.maximum.accumulate(equity)
    dd = (equity / np.maximum(peak, 1e-12)) - 1.0
    return dd


# ---------------------------
# Backtest engine
# ---------------------------

def run_backtest(
    df: pd.DataFrame,
    strategy_func: Callable,
    *,
    lookback: int = 200,
    initial_equity: float = 10000.0,
    fee_bps: float = 10.0,
    slippage_bps: float = 5.0,
    closed_only: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Walk-forward backtest using Layer 2 generate_intent.

    At each bar i (for i >= lookback):
      1. Call strategy_func(df[:i+1], ctx, closed_only=closed_only)
      2. Process action & desired_exposure_frac
      3. Adjust position to match desired exposure

    Returns:
        (equity_df, metrics_dict)
    """
    n = len(df)
    if n <= lookback:
        raise ValueError(f"Insufficient data: {n} rows, need > {lookback}")

    fee_rate = fee_bps / 10_000.0
    slip_rate = slippage_bps / 10_000.0

    # State
    equity = initial_equity
    cash = initial_equity
    pos_qty = 0.0  # BTC held
    pos_value = 0.0  # Value of position

    equity_rows: List[Dict[str, Any]] = []
    exposure_frac_history: List[float] = []
    in_market_flags: List[bool] = []

    ctx = {
        "mode": "backtest",
        "dry_run": True,
        "now_utc": datetime.now(timezone.utc),
    }

    for i in range(lookback, n):
        ts = df.loc[i, "Timestamp"]
        price = float(df.loc[i, "Close"])

        # Mark-to-market before action
        pos_value = pos_qty * price
        equity = cash + pos_value

        # Current exposure
        curr_expo = pos_value / equity if equity > 0 else 0.0

        # Slice up to and including current bar
        df_slice = df.iloc[: i + 1].copy()

        # Call strategy
        try:
            intent = strategy_func(df_slice, ctx, closed_only=closed_only)
        except Exception as e:
            # Fail-safe: hold on error
            intent = {
                "action": "HOLD",
                "confidence": 0.0,
                "desired_exposure_frac": curr_expo,
                "horizon_hours": 0,
                "reason": f"strategy_error: {e}",
                "meta": {},
            }

        action = _normalize_action(intent.get("action"))
        desired_expo = float(intent.get("desired_exposure_frac") or 0.0)
        desired_expo = max(0.0, min(1.0, desired_expo))

        # Compute target exposure based on action
        if action == "EXIT":
            target_expo = 0.0
        elif action == "ENTER":
            target_expo = desired_expo
        else:  # HOLD - maintain current exposure or move to desired if specified
            # If strategy specifies 0 exposure on HOLD, respect it (conservative)
            target_expo = desired_expo if desired_expo > 0 else curr_expo

        target_expo = max(0.0, min(1.0, target_expo))

        # Rebalance position
        target_pos_value = equity * target_expo
        delta_value = target_pos_value - pos_value

        if abs(delta_value) > 1.0:  # Min trade threshold $1
            if delta_value > 0:
                # BUY
                fill_price = price * (1.0 + slip_rate)
                qty_to_buy = delta_value / fill_price
                cost = qty_to_buy * fill_price * (1.0 + fee_rate)

                if cost <= cash:
                    cash -= cost
                    pos_qty += qty_to_buy
            else:
                # SELL
                fill_price = price * (1.0 - slip_rate)
                qty_to_sell = abs(delta_value) / fill_price
                qty_to_sell = min(qty_to_sell, pos_qty)  # Can't sell more than we have

                if qty_to_sell > 0:
                    proceeds = qty_to_sell * fill_price * (1.0 - fee_rate)
                    cash += proceeds
                    pos_qty -= qty_to_sell

        # Update mark-to-market
        pos_value = pos_qty * price
        equity = cash + pos_value

        # Track exposure
        curr_expo_final = pos_value / equity if equity > 0 else 0.0
        exposure_frac_history.append(curr_expo_final)
        in_market_flags.append(curr_expo_final > 0.01)  # > 1% exposure = in market

        equity_rows.append({
            "Timestamp": ts,
            "equity": equity,
            "cash": cash,
            "pos_qty": pos_qty,
            "price": price,
            "exposure": curr_expo_final,
        })

    equity_df = pd.DataFrame(equity_rows)
    eq = equity_df["equity"].values

    # Compute returns
    if len(eq) > 1:
        returns = np.diff(eq) / np.maximum(eq[:-1], 1e-12)
    else:
        returns = np.array([0.0])

    # Total return
    total_return = (eq[-1] / initial_equity) - 1.0

    # CAGR
    first_ts = equity_df["Timestamp"].iloc[0]
    last_ts = equity_df["Timestamp"].iloc[-1]
    years = max((last_ts - first_ts).total_seconds() / (365.25 * 24 * 3600), 1e-9)
    cagr = (eq[-1] / initial_equity) ** (1.0 / years) - 1.0

    # Max drawdown
    dd_series = compute_drawdown_series(eq)
    max_dd = float(dd_series.min())

    # Calmar
    calmar = compute_calmar(cagr, max_dd)

    # Sortino (hourly data)
    sortino = compute_sortino(returns, risk_free=0.0, ann_factor=8760)

    # Average exposure
    avg_exposure = float(np.mean(exposure_frac_history)) if exposure_frac_history else 0.0

    # Time in market
    time_in_market = float(np.mean(in_market_flags)) if in_market_flags else 0.0

    metrics = {
        "total_return": float(total_return),
        "cagr": float(cagr),
        "max_drawdown": float(abs(max_dd)),  # Report as positive
        "calmar": float(calmar) if np.isfinite(calmar) else 0.0,
        "sortino": float(sortino) if np.isfinite(sortino) else 0.0,
        "avg_exposure": float(avg_exposure),
        "time_in_market": float(time_in_market),
        "final_equity": float(eq[-1]),
        "bars": int(len(equity_df)),
        "years": float(years),
    }

    return equity_df, metrics


# ---------------------------
# Main entrypoint
# ---------------------------

def main() -> Dict[str, Any]:
    """
    Run backtest from env config.

    Env vars:
        ARGUS_STRATEGY_MODULE: e.g. "research.strategies.sg_core_exposure_v1"
        ARGUS_STRATEGY_FUNC: e.g. "generate_intent" (default)
        ARGUS_DATA_FILE: path to OHLCV CSV (default: flight_recorder.csv)
        ARGUS_LOOKBACK: min bars before trading (default: 200)
        ARGUS_INITIAL_EQUITY: starting equity (default: 10000)
        ARGUS_FEE_BPS: fee in bps per side (default: 10)
        ARGUS_SLIPPAGE_BPS: slippage in bps (default: 5)
    """
    argus_dir = _setup_path()

    # Load config from env
    module_path = _env_str("ARGUS_STRATEGY_MODULE", "research.strategies.sg_core_exposure_v1")
    func_name = _env_str("ARGUS_STRATEGY_FUNC", "generate_intent")

    # Default data file: flight_recorder.csv in argus dir
    default_data = str(argus_dir / "flight_recorder.csv")
    data_file = _env_str("ARGUS_DATA_FILE", default_data)

    lookback = _env_int("ARGUS_LOOKBACK", 200)
    initial_equity = _env_float("ARGUS_INITIAL_EQUITY", 10000.0)
    fee_bps = _env_float("ARGUS_FEE_BPS", 10.0)
    slippage_bps = _env_float("ARGUS_SLIPPAGE_BPS", 5.0)

    print("=" * 60)
    print("BACKTEST RUNNER (Layer 2 Strategy)")
    print("=" * 60)
    print(f"Strategy module: {module_path}")
    print(f"Strategy func:   {func_name}")
    print(f"Data file:       {data_file}")
    print(f"Lookback:        {lookback}")
    print(f"Initial equity:  ${initial_equity:,.2f}")
    print(f"Fee (bps):       {fee_bps}")
    print(f"Slippage (bps):  {slippage_bps}")
    print("-" * 60)

    # Load strategy
    print(f"Loading strategy: {module_path}.{func_name}")
    strategy_func = load_strategy_func(module_path, func_name)
    print(f"  -> Loaded successfully")

    # Load data
    print(f"Loading data: {data_file}")
    df = load_flight_recorder(data_file)
    print(f"  -> {len(df):,} rows ({df['Timestamp'].iloc[0]} to {df['Timestamp'].iloc[-1]})")

    # Optional: limit bars for fast trace generation (debug only)
    debug_max_bars = _env_int("ARGUS_DEBUG_TRACE_MAX_BARS", 0)
    if debug_max_bars > 0:
        cap = lookback + debug_max_bars
        if len(df) > cap:
            df = df.iloc[:cap].copy()
            print(f"  -> Limited to first {debug_max_bars} trading bars for trace (ARGUS_DEBUG_TRACE_MAX_BARS)")

    # Run backtest
    print("Running backtest...")
    equity_df, metrics = run_backtest(
        df,
        strategy_func,
        lookback=lookback,
        initial_equity=initial_equity,
        fee_bps=fee_bps,
        slippage_bps=slippage_bps,
        closed_only=True,
    )

    # Per-bar trace export for behavioral diff (BTC-only; same dataset/env/date window)
    repo_root = Path(__file__).resolve().parents[4]
    debug_dir = repo_root / "debug"
    debug_dir.mkdir(parents=True, exist_ok=True)
    trace_path = debug_dir / "harness_btc_trace.csv"
    n_eq = len(equity_df)
    next_bar_return = np.full(n_eq, np.nan, dtype=float)
    for j in range(n_eq - 1):
        idx_cur = lookback + j
        idx_next = lookback + j + 1
        if idx_next < len(df):
            next_bar_return[j] = (float(df.loc[idx_next, "Close"]) / float(df.loc[idx_cur, "Close"])) - 1.0
    eq_arr = equity_df["equity"].to_numpy(dtype=float)
    portfolio_return = np.full(n_eq, np.nan, dtype=float)
    for j in range(n_eq - 1):
        portfolio_return[j] = (eq_arr[j + 1] / eq_arr[j]) - 1.0
    trace_df = pd.DataFrame({
        "timestamp": equity_df["Timestamp"],
        "close_price": equity_df["price"],
        "exposure": equity_df["exposure"],
        "next_bar_return": next_bar_return,
        "portfolio_return": portfolio_return,
        "equity": equity_df["equity"],
    })
    trace_df.to_csv(trace_path, index=False)
    print(f"Trace written: {trace_path}")

    # Print results
    print("\n" + "=" * 60)
    print("METRICS SUMMARY")
    print("=" * 60)
    print(f"Total Return:    {metrics['total_return'] * 100:>10.2f}%")
    print(f"CAGR:            {metrics['cagr'] * 100:>10.2f}%")
    print(f"Max Drawdown:    {metrics['max_drawdown'] * 100:>10.2f}%")
    print(f"Calmar:          {metrics['calmar']:>10.2f}")
    print(f"Sortino:         {metrics['sortino']:>10.2f}")
    print(f"Avg Exposure:    {metrics['avg_exposure'] * 100:>10.2f}%")
    print(f"Time in Market:  {metrics['time_in_market'] * 100:>10.2f}%")
    print(f"Final Equity:    ${metrics['final_equity']:>10,.2f}")
    print(f"Bars:            {metrics['bars']:>10,}")
    print(f"Years:           {metrics['years']:>10.2f}")
    print("=" * 60)

    return metrics


if __name__ == "__main__":
    main()

