from __future__ import annotations

import os
import sys
import math
import time
import inspect
import importlib.util
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


# -----------------------------
# Defaults / Paths
# -----------------------------
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

DEFAULT_DATASET_1 = os.path.join(REPO_ROOT, "data", "historical", "btc_usd_1h.csv")
DEFAULT_DATASET_2 = os.path.join(REPO_ROOT, "data", "btcusd_3600s_2019-01-01_to_2025-12-30.csv")

RUNTIME_STRAT_FILE = os.path.join(
    REPO_ROOT, "runtime", "argus", "research", "strategies", "sg_regime_trend_v1.py"
)

OUT_DIR = os.path.join(REPO_ROOT, "output")
os.makedirs(OUT_DIR, exist_ok=True)


# -----------------------------
# Params
# -----------------------------
@dataclass(frozen=True)
class BTParams:
    trend_min: float
    vol_max: float
    horizon_h: int
    vol_exit_max: float
    reentry_cooldown_h: float
    exit_min_hold_h: float
    adx_min: float
    vol_win: int
    trend_win: int
    spike_ratio_max: float

    # Added: vol_min for entry filter
    vol_min: float = 0.003

    # ADX gating (strategy-owned but harness must pass to cfg)
    adx_rel_win: int = 168
    adx_rel_min: float = 1.0
    adx_mode: str = "soft"  # "off", "soft", "hard"
    adx_soft_penalty: float = 0.10

    # Optional indicator lengths (used by this harness)
    ema_fast: int = 20
    ema_slow: int = 50
    atr_len: int = 14
    adx_len: int = 14

    # Backtest frictions
    fee_bps_side: float = 10.0
    slippage_bps_fill: float = 2.0


# -----------------------------
# Utilities: Indicators
# -----------------------------
def _ema(s: pd.Series, n: int) -> pd.Series:
    n = max(int(n), 1)
    return s.ewm(span=n, adjust=False).mean()


def _true_range(df: pd.DataFrame) -> pd.Series:
    high = df["High"]
    low = df["Low"]
    close = df["Close"]
    prev_close = close.shift(1)
    tr1 = (high - low).abs()
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    return pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)


def _atr(df: pd.DataFrame, n: int) -> pd.Series:
    tr = _true_range(df)
    n = max(int(n), 1)
    return tr.ewm(alpha=1.0 / n, adjust=False).mean()


def _adx(df: pd.DataFrame, n: int) -> pd.Series:
    n = max(int(n), 1)
    high = df["High"]
    low = df["Low"]

    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)

    tr = _true_range(df)
    atr = tr.ewm(alpha=1.0 / n, adjust=False).mean()

    plus_di = 100.0 * (plus_dm.ewm(alpha=1.0 / n, adjust=False).mean() / atr.replace(0, math.nan))
    minus_di = 100.0 * (minus_dm.ewm(alpha=1.0 / n, adjust=False).mean() / atr.replace(0, math.nan))

    dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, math.nan)
    adx = dx.ewm(alpha=1.0 / n, adjust=False).mean()
    return adx.fillna(0.0)


def _roll_mean(s: pd.Series, win: int) -> pd.Series:
    win = max(int(win), 1)
    return s.rolling(window=win, min_periods=1).mean()


# -----------------------------
# Strategy loading (robust)
# -----------------------------
def _load_strategy_module_from_file(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Strategy file not found: {path}")

    # Register module in sys.modules BEFORE exec to avoid dataclass issues on Windows/Py3.13+
    mod_name = f"_rt_strat_{int(time.time() * 1e9)}"
    spec = importlib.util.spec_from_file_location(mod_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load spec for strategy: {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    return mod


def _get_strategy_cfg(mod) -> Dict[str, Any]:
    """
    Permissive config discovery:
      - _get_env_cfg() (preferred if present)
      - get_env_cfg()
      - DEFAULT_CFG dict
      - fallback minimal defaults
    """
    if hasattr(mod, "_get_env_cfg") and callable(getattr(mod, "_get_env_cfg")):
        cfg = mod._get_env_cfg()
        if isinstance(cfg, dict):
            return cfg

    if hasattr(mod, "get_env_cfg") and callable(getattr(mod, "get_env_cfg")):
        cfg = mod.get_env_cfg()
        if isinstance(cfg, dict):
            return cfg

    if hasattr(mod, "DEFAULT_CFG") and isinstance(getattr(mod, "DEFAULT_CFG"), dict):
        return dict(getattr(mod, "DEFAULT_CFG"))

    return {
        "ema_fast": 20,
        "ema_slow": 50,
        "atr_len": 14,
        "adx_len": 14,
        "vol_min": 0.003,
        "expo_frac": 0.25,
        "adx_rel_win": 168,
        "adx_rel_min": 1.0,
        "adx_mode": "hard",
        "adx_soft_penalty": 0.10,
    }


def _call_generate_intent(mod, candidates: Dict[str, Any]) -> Any:
    """
    Calls either:
      - generate_intent_from_values(..., **maybe_candidates)
      - generate_intent(...)
    while being signature-flexible.
    """
    fn = None
    if hasattr(mod, "generate_intent_from_values") and callable(getattr(mod, "generate_intent_from_values")):
        fn = mod.generate_intent_from_values
    elif hasattr(mod, "generate_intent") and callable(getattr(mod, "generate_intent")):
        fn = mod.generate_intent

    if fn is None:
        raise RuntimeError("Strategy module missing generate_intent_from_values() and generate_intent()")

    sig = inspect.signature(fn)

    # If it accepts **kwargs, pass everything
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
        return fn(**candidates)

    # Otherwise only pass accepted params
    kwargs = {}
    for k in sig.parameters.keys():
        if k in candidates:
            kwargs[k] = candidates[k]
    return fn(**kwargs)


def _intent_get(intent: Any, key: str, default=None):
    if isinstance(intent, dict):
        return intent.get(key, default)
    return getattr(intent, key, default)


def _normalize_action(raw_action: Optional[str]) -> str:
    if not raw_action:
        return "HOLD"
    a = str(raw_action).strip().upper()

    if a in ("ENTER_LONG", "ENTER", "BUY", "LONG"):
        return "ENTER"
    if a in ("EXIT_LONG", "EXIT", "SELL", "CLOSE"):
        return "EXIT"
    if a in ("FLAT", "HOLD", "NONE", "NOOP"):
        return "HOLD"

    return "HOLD"


# -----------------------------
# Data load
# -----------------------------
def _load_data() -> pd.DataFrame:
    dataset = os.environ.get("BT_DATASET", "").strip()
    if dataset:
        path = dataset
    else:
        path = DEFAULT_DATASET_2 if os.path.exists(DEFAULT_DATASET_2) else DEFAULT_DATASET_1

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset not found. Tried:\n  {path}\n"
            f"Set BT_DATASET env var to your CSV path."
        )

    df = pd.read_csv(path)
    if "Timestamp" not in df.columns:
        raise ValueError("CSV must contain a Timestamp column")

    df["Timestamp"] = pd.to_datetime(df["Timestamp"], utc=True)
    df = df.sort_values("Timestamp").reset_index(drop=True)

    for col in ("Open", "High", "Low", "Close"):
        if col not in df.columns:
            raise ValueError(f"CSV missing required OHLC column: {col}")

    print(f"[BT] Data rows: {len(df):,} | Range: {df['Timestamp'].iloc[0]} -> {df['Timestamp'].iloc[-1]}")
    print(f"[BT] Dataset: {path}")
    return df


# -----------------------------
# Backtest engine
# -----------------------------
def run_backtest(df: pd.DataFrame, params: BTParams) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    mod = _load_strategy_module_from_file(RUNTIME_STRAT_FILE)

    # Strategy may provide lots of knobs, but EXECUTOR owns the gating knobs.
    cfg = dict(_get_strategy_cfg(mod))

    # Indicator lengths: allow strategy to override if it explicitly set them, otherwise use BTParams.
    cfg.setdefault("ema_fast", params.ema_fast)
    cfg.setdefault("ema_slow", params.ema_slow)
    cfg.setdefault("atr_len", params.atr_len)
    cfg.setdefault("adx_len", params.adx_len)

    # ---- CRITICAL: force overwrite executor-owned controls (do NOT use setdefault) ----
    cfg["trend_min"] = float(params.trend_min)
    cfg["vol_min"] = float(params.vol_min)
    cfg["vol_max"] = float(params.vol_max)
    cfg["horizon_h"] = int(params.horizon_h)
    cfg["vol_exit_max"] = float(params.vol_exit_max)
    cfg["reentry_cooldown_h"] = float(params.reentry_cooldown_h)
    cfg["exit_min_hold_h"] = float(params.exit_min_hold_h)
    cfg["vol_win"] = int(params.vol_win)
    cfg["trend_win"] = int(params.trend_win)
    cfg["spike_ratio_max"] = float(params.spike_ratio_max)
    cfg["adx_min"] = float(params.adx_min)
    cfg["adx_rel_win"] = int(params.adx_rel_win)
    cfg["adx_rel_min"] = float(params.adx_rel_min)
    cfg["adx_mode"] = str(params.adx_mode).lower()
    cfg["adx_soft_penalty"] = float(params.adx_soft_penalty)

    # Compute indicators + derived features that your debug harness shows
    ema_fast_s = _ema(df["Close"], int(cfg["ema_fast"]))
    ema_slow_s = _ema(df["Close"], int(cfg["ema_slow"]))
    atr_s = _atr(df, int(cfg["atr_len"]))
    adx_s = _adx(df, int(cfg["adx_len"]))

    close_s = df["Close"].astype(float)

    # atr_pct
    atr_pct_raw_s = (atr_s / close_s.replace(0, math.nan)).fillna(0.0)
    vol_win = int(cfg.get("vol_win") or 1)
    atr_pct_s_s = _roll_mean(atr_pct_raw_s, vol_win)

    # trend_strength = abs((ema_fast - ema_slow) / atr)
    ema_diff = (ema_fast_s - ema_slow_s)
    trend_strength_raw_s = (ema_diff.abs() / atr_s.replace(0, math.nan)).fillna(0.0)
    trend_win = int(cfg.get("trend_win") or 1)
    trend_strength_s_s = _roll_mean(trend_strength_raw_s, trend_win)

    # spike_ratio: current vol vs smoothed vol
    spike_ratio_s = (atr_pct_raw_s / atr_pct_s_s.replace(0, math.nan)).fillna(1.0)

    # adx_rel: ratio vs rolling mean
    adx_rel_win = int(cfg.get("adx_rel_win", 168) or 1)
    adx_rel_s = (adx_s / _roll_mean(adx_s, adx_rel_win).replace(0, math.nan)).fillna(1.0)

    # Backtest state
    start_equity = 10_000.0
    equity = start_equity
    cash = start_equity
    pos_qty = 0.0
    entry_ts: Optional[pd.Timestamp] = None
    last_exit_ts: Optional[pd.Timestamp] = None

    trades: List[Dict[str, Any]] = []
    curve_rows: List[Dict[str, Any]] = []

    fee_rate = params.fee_bps_side / 10_000.0
    slip_rate = params.slippage_bps_fill / 10_000.0

    def mark_to_market(price: float) -> float:
        return cash + pos_qty * price

    seen_enter = 0
    seen_exit = 0
    seen_hold = 0

    # Executor-owned controls (use params directly; cfg mirrors them for strategy visibility only)
    min_hold_h = float(params.exit_min_hold_h or 0.0)
    cooldown_h = float(params.reentry_cooldown_h or 0.0)

    for i in range(len(df)):
        ts = df["Timestamp"].iloc[i]
        price = float(close_s.iloc[i])

        candidates = {
            "ts": ts,
            "price": price,
            "close": price,
            "ema_fast": float(ema_fast_s.iloc[i]),
            "ema_slow": float(ema_slow_s.iloc[i]),
            "atr": float(atr_s.iloc[i]),
            "adx": float(adx_s.iloc[i]),
            "atr_pct_raw": float(atr_pct_raw_s.iloc[i]),
            "atr_pct_s": float(atr_pct_s_s.iloc[i]),
            "trend_strength_raw": float(trend_strength_raw_s.iloc[i]),
            "trend_strength_s": float(trend_strength_s_s.iloc[i]),
            "spike_ratio": float(spike_ratio_s.iloc[i]),
            "adx_rel": float(adx_rel_s.iloc[i]),
            "p_long": None,
            "cfg": cfg,
        }

        intent = _call_generate_intent(mod, candidates=candidates)

        raw_action = _intent_get(intent, "action", None)
        action = _normalize_action(raw_action)

        if action == "ENTER":
            seen_enter += 1
        elif action == "EXIT":
            seen_exit += 1
        else:
            seen_hold += 1

        # Horizon (executor-owned)
        horizon_h = int(_intent_get(intent, "horizon_hours", params.horizon_h) or 0)
        horizon_due = False
        if entry_ts is not None and horizon_h > 0:
            held_hours = (ts - entry_ts).total_seconds() / 3600.0
            if held_hours >= horizon_h:
                horizon_due = True

        # Min hold + cooldown (executor-owned)
        can_exit = True
        if entry_ts is not None and min_hold_h > 0:
            held_hours = (ts - entry_ts).total_seconds() / 3600.0
            can_exit = held_hours >= min_hold_h

        can_enter = True
        if last_exit_ts is not None and cooldown_h > 0:
            since_exit_h = (ts - last_exit_ts).total_seconds() / 3600.0
            can_enter = since_exit_h >= cooldown_h

        desired_frac = float(_intent_get(intent, "desired_exposure_frac", cfg.get("expo_frac", 0.25)) or 0.0)
        desired_frac = max(0.0, min(1.0, desired_frac))

        # Execute: ENTER
        if action == "ENTER" and pos_qty == 0.0 and can_enter and desired_frac > 0.0:
            fill_price = price * (1.0 + slip_rate)
            notional = equity * desired_frac
            qty = notional / fill_price if fill_price > 0 else 0.0
            fee = notional * fee_rate

            cash -= (notional + fee)
            pos_qty += qty
            entry_ts = ts

            trades.append({
                "timestamp": ts,
                "side": "BUY",
                "price": fill_price,
                "qty": qty,
                "notional": notional,
                "fee": fee,
                "reason": _intent_get(intent, "reason", "enter"),
            })

        # Execute: EXIT (either intent exit or horizon exit)
        do_exit = (action == "EXIT") or horizon_due
        if do_exit and pos_qty != 0.0 and can_exit:
            fill_price = price * (1.0 - slip_rate)
            notional = pos_qty * fill_price
            fee = notional * fee_rate

            cash += (notional - fee)
            pos_qty = 0.0
            last_exit_ts = ts
            entry_ts = None

            trades.append({
                "timestamp": ts,
                "side": "SELL",
                "price": fill_price,
                "qty": None,
                "notional": notional,
                "fee": fee,
                "reason": ("horizon_exit" if horizon_due else _intent_get(intent, "reason", "exit")),
            })

        equity = mark_to_market(price)
        curve_rows.append({"Timestamp": ts, "Equity": equity})

    curve = pd.DataFrame(curve_rows)
    trades_df = pd.DataFrame(trades)

    # Summary
    total_return = (equity / start_equity) - 1.0
    years = max((df["Timestamp"].iloc[-1] - df["Timestamp"].iloc[0]).total_seconds() / (365.25 * 24 * 3600), 1e-9)
    cagr = (equity / start_equity) ** (1.0 / years) - 1.0

    peak = -float("inf")
    mdd = 0.0
    for v in curve["Equity"].values:
        peak = max(peak, v)
        dd = (v / peak) - 1.0 if peak > 0 else 0.0
        mdd = min(mdd, dd)

    summary = {
        "final_equity": float(equity),
        "total_return": float(total_return),
        "cagr": float(cagr),
        "max_drawdown": float(mdd),
        "trades": int(len(trades_df)),
        "seen_enter_intents": int(seen_enter),
        "seen_exit_intents": int(seen_exit),
        "seen_hold_intents": int(seen_hold),
    }
    return curve, trades_df, summary


def _env_float(name: str, default: float) -> float:
    v = os.environ.get(name, "").strip()
    if not v:
        return default
    try:
        return float(v)
    except Exception:
        return default


def _env_int(name: str, default: int) -> int:
    v = os.environ.get(name, "").strip()
    if not v:
        return default
    try:
        return int(float(v))
    except Exception:
        return default


def _env_str(name: str, default: str) -> str:
    v = os.environ.get(name, "").strip()
    return v if v else default


def _build_params_from_env() -> BTParams:
    reentry_cd = _env_float("PRIME_REENTRY_COOLDOWN_H", 24.0)
    exit_min_hold = _env_float("PRIME_EXIT_MIN_HOLD_H", 12.0)

    trend_min = _env_float("SGRT_TREND_MIN", 0.30)
    vol_min = _env_float("SGRT_VOL_MIN", 0.003)
    vol_max = _env_float("SGRT_VOL_MAX", 0.020)
    horizon_h = _env_int("SGRT_HORIZON_H", 48)
    vol_exit_max = _env_float("SGRT_VOL_EXIT_MAX", 0.030)

    vol_win = _env_int("SGRT_VOL_WIN", 24)
    trend_win = _env_int("SGRT_TREND_WIN", 24)
    spike_ratio_max = _env_float("SGRT_SPIKE_RATIO_MAX", 1.8)

    adx_min = _env_float("SGRT_ADX_MIN", 18.0)
    adx_rel_win = _env_int("SGRT_ADX_REL_WIN", 168)
    adx_rel_min = _env_float("SGRT_ADX_REL_MIN", 1.0)
    adx_mode = _env_str("SGRT_ADX_MODE", "soft").lower()
    adx_soft_penalty = _env_float("SGRT_ADX_SOFT_PENALTY", 0.10)

    return BTParams(
        trend_min=trend_min,
        vol_min=vol_min,
        vol_max=vol_max,
        horizon_h=horizon_h,
        vol_exit_max=vol_exit_max,
        reentry_cooldown_h=reentry_cd,
        exit_min_hold_h=exit_min_hold,
        adx_min=adx_min,
        adx_rel_win=adx_rel_win,
        adx_rel_min=adx_rel_min,
        adx_mode=adx_mode,
        adx_soft_penalty=adx_soft_penalty,
        vol_win=vol_win,
        trend_win=trend_win,
        spike_ratio_max=spike_ratio_max,
    )


def _log_params(params: BTParams) -> None:
    """Log effective BTParams for reproducibility."""
    print("\n[BT] ========== EFFECTIVE CONFIG ==========")
    print(f"[BT] Strategy: {RUNTIME_STRAT_FILE}")
    print(f"[BT] Dataset:  {os.environ.get('BT_DATASET', '(default)')}")
    print("[BT] --- Thresholds ---")
    print(f"[BT]   trend_min={params.trend_min}")
    print(f"[BT]   vol_min={params.vol_min}, vol_max={params.vol_max}, vol_exit_max={params.vol_exit_max}")
    print(f"[BT]   spike_ratio_max={params.spike_ratio_max}")
    print("[BT] --- ADX Gating ---")
    print(f"[BT]   adx_mode={params.adx_mode}")
    print(f"[BT]   adx_min={params.adx_min}, adx_rel_min={params.adx_rel_min}, adx_rel_win={params.adx_rel_win}")
    print(f"[BT]   adx_soft_penalty={params.adx_soft_penalty}")
    print("[BT] --- Executor (churn killers) ---")
    print(f"[BT]   reentry_cooldown_h={params.reentry_cooldown_h}")
    print(f"[BT]   exit_min_hold_h={params.exit_min_hold_h}")
    print(f"[BT]   horizon_h={params.horizon_h}")
    print("[BT] --- Smoothing windows ---")
    print(f"[BT]   vol_win={params.vol_win}, trend_win={params.trend_win}")
    print("[BT] --- Costs ---")
    print(f"[BT]   fee_bps_side={params.fee_bps_side}, slippage_bps_fill={params.slippage_bps_fill}")
    print("[BT] ==========================================\n")


def main() -> None:
    df = _load_data()
    print(f"[BT] Loaded strategy from: {RUNTIME_STRAT_FILE}")

    params = _build_params_from_env()
    _log_params(params)
    curve, trades, summary = run_backtest(df, params)

    print("\n========== TEAR SHEET (SG Regime Trend v1) ==========")
    print(f"Final equity:  ${summary['final_equity']:,.2f}")
    print(f"Total return:  {summary['total_return']*100:,.2f}%")
    print(f"CAGR:          {summary['cagr']*100:,.2f}%")
    print(f"Max drawdown:  {summary['max_drawdown']*100:,.2f}%")
    print(f"Trades:        {summary['trades']}")
    print(
        f"Intents seen:  enter={summary['seen_enter_intents']:,} "
        f"exit={summary['seen_exit_intents']:,} "
        f"hold={summary['seen_hold_intents']:,}"
    )
    print(f"Costs:         fee={params.fee_bps_side:.1f}bps/side, slippage={params.slippage_bps_fill:.1f}bps/fill")
    print("====================================================\n")

    eq_path = os.path.join(OUT_DIR, "bt_sg_regime_trend_v1_equity.csv")
    tr_path = os.path.join(OUT_DIR, "bt_sg_regime_trend_v1_trades.csv")
    curve.to_csv(eq_path, index=False)
    trades.to_csv(tr_path, index=False)
    print(f"[BT] Wrote equity curve: {eq_path}")
    print(f"[BT] Wrote trades:       {tr_path}")


if __name__ == "__main__":
    main()
