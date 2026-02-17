# ======================================================================
# FILE: research/backtests/experiments/debug_sg_regime_trend_v1_intents.py
# FULL REPLACEMENT
# ======================================================================

from __future__ import annotations

import os
import sys
import math
import time
import inspect
import importlib.util
from typing import Any, Dict, List, Tuple, Optional

import pandas as pd

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

DEFAULT_DATASET_1 = os.path.join(REPO_ROOT, "data", "historical", "btc_usd_1h.csv")
DEFAULT_DATASET_2 = os.path.join(REPO_ROOT, "data", "btcusd_3600s_2019-01-01_to_2025-12-30.csv")

RUNTIME_STRAT_FILE = os.path.join(
    REPO_ROOT, "runtime", "argus", "research", "strategies", "sg_regime_trend_v1.py"
)


# -----------------------------
# Utilities: Indicators (mirror the backtest harness lens)
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
# IO / Strategy loading
# -----------------------------
def _load_data() -> Tuple[pd.DataFrame, str]:
    dataset = os.environ.get("BT_DATASET", "").strip()
    if dataset:
        path = dataset
    else:
        path = DEFAULT_DATASET_2 if os.path.exists(DEFAULT_DATASET_2) else DEFAULT_DATASET_1

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset not found. Tried:\n  {path}\nSet BT_DATASET env var to your CSV path."
        )

    df = pd.read_csv(path)
    if "Timestamp" not in df.columns:
        raise ValueError("CSV must contain a Timestamp column")

    df["Timestamp"] = pd.to_datetime(df["Timestamp"], utc=True)
    df = df.sort_values("Timestamp").reset_index(drop=True)

    for col in ("Open", "High", "Low", "Close"):
        if col not in df.columns:
            raise ValueError(f"CSV missing required OHLC column: {col}")

    return df, path


def _load_strategy_from_file(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Strategy file not found: {path}")

    mod_name = f"_rt_strat_dbg_{int(time.time() * 1e9)}"
    spec = importlib.util.spec_from_file_location(mod_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load spec for strategy: {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod  # important for dataclasses on Windows
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    return mod


def _get_strategy_cfg(mod) -> Dict[str, Any]:
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
        "expo_frac": 0.25,
        "vol_min": 0.003,
        "vol_win": 24,
        "trend_win": 24,
        "adx_rel_win": 168,
    }


def _intent_get(intent: Any, key: str, default=None):
    if isinstance(intent, dict):
        return intent.get(key, default)
    return getattr(intent, key, default)


def _call_generate_intent_from_values(mod, candidates: Dict[str, Any]) -> Any:
    """
    Robustly call generate_intent_from_values with multiple compatibility styles:
      1) fn(candidates=candidates)          (new preferred style)
      2) fn(**candidates)                  (kwargs style)
      3) fn(candidates)                    (single positional dict)
    """
    if not hasattr(mod, "generate_intent_from_values") or not callable(getattr(mod, "generate_intent_from_values")):
        raise RuntimeError("Strategy module missing generate_intent_from_values()")

    fn = mod.generate_intent_from_values
    sig = inspect.signature(fn)

    # Fast path: accepts **kwargs -> try candidates= first (if supported), else **candidates
    has_varkw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
    has_candidates_param = "candidates" in sig.parameters

    # 1) candidates=...
    if has_candidates_param or has_varkw:
        try:
            return fn(candidates=candidates)
        except TypeError:
            pass

    # 2) **kwargs
    if has_varkw or True:
        try:
            return fn(**candidates)
        except TypeError:
            pass

    # 3) positional dict
    try:
        return fn(dict(candidates))
    except TypeError as e:
        raise RuntimeError(f"Could not call generate_intent_from_values with supported styles: {e}") from e


def _call_generate_intent_df(mod, df_window: pd.DataFrame, ctx: Any) -> Any:
    if not hasattr(mod, "generate_intent") or not callable(getattr(mod, "generate_intent")):
        raise RuntimeError("Strategy module missing generate_intent()")
    return mod.generate_intent(df_window, ctx)


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    df, dataset_path = _load_data()
    print(f"[DBG] Repo root:      {REPO_ROOT}")
    print(f"[DBG] Dataset:        {dataset_path}")
    print(f"[DBG] Rows:           {len(df):,}")
    print(f"[DBG] Range:          {df['Timestamp'].iloc[0]} -> {df['Timestamp'].iloc[-1]}")
    print(f"[DBG] Strategy file:  {RUNTIME_STRAT_FILE}")

    mod = _load_strategy_from_file(RUNTIME_STRAT_FILE)
    print(f"[DBG] Loaded module:  {getattr(mod, '__file__', 'unknown')}")
    print(f"[DBG] Has generate_intent:             {hasattr(mod, 'generate_intent')}")
    print(f"[DBG] Has generate_intent_from_values: {hasattr(mod, 'generate_intent_from_values')}")

    cfg = dict(_get_strategy_cfg(mod))

    # Env snapshot
    print("[DBG] Env snapshot (SGRT_* and PRIME_*):")
    keys = sorted([k for k in os.environ.keys() if k.startswith("SGRT_") or k.startswith("PRIME_") or k.startswith("BT_")])
    for k in keys:
        print(f"      {k}={os.environ.get(k)}")
    print()

    # Effective config from strategy
    print("[DBG] Effective cfg from strategy module:")
    cfg_keys = [
        "ema_fast", "ema_slow", "atr_len", "adx_len",
        "trend_min", "vol_min", "vol_max", "vol_exit_max",
        "horizon_h", "expo_frac",
        "vol_win", "trend_win", "spike_ratio_max",
        "adx_min", "adx_rel_win", "adx_rel_min", "adx_mode", "adx_soft_penalty",
        "reentry_cooldown_h", "exit_min_hold_h",
    ]
    for k in cfg_keys:
        if k in cfg:
            print(f"      {k}={cfg[k]}")
    print()

    # Compute the SAME lens as the backtest harness, so debug reflects real behavior.
    close_s = df["Close"].astype(float)

    ema_fast_n = int(cfg.get("ema_fast", cfg.get("ema_fast_n", 20)) or 20)
    ema_slow_n = int(cfg.get("ema_slow", cfg.get("ema_slow_n", 50)) or 50)
    atr_len = int(cfg.get("atr_len", 14) or 14)
    adx_len = int(cfg.get("adx_len", 14) or 14)

    vol_win = int(cfg.get("vol_win", 24) or 1)
    trend_win = int(cfg.get("trend_win", 24) or 1)
    adx_rel_win = int(cfg.get("adx_rel_win", 168) or 1)

    ema_fast_s = _ema(close_s, ema_fast_n)
    ema_slow_s = _ema(close_s, ema_slow_n)
    atr_s = _atr(df, atr_len)
    adx_s = _adx(df, adx_len)

    atr_pct_raw_s = (atr_s / close_s.replace(0, math.nan)).fillna(0.0)
    atr_pct_s_s = _roll_mean(atr_pct_raw_s, vol_win)

    ema_diff = (ema_fast_s - ema_slow_s)
    trend_strength_raw_s = (ema_diff.abs() / atr_s.replace(0, math.nan)).fillna(0.0)
    trend_strength_s_s = _roll_mean(trend_strength_raw_s, trend_win)

    spike_ratio_s = (atr_pct_raw_s / atr_pct_s_s.replace(0, math.nan)).fillna(1.0)
    adx_rel_s = (adx_s / _roll_mean(adx_s, adx_rel_win).replace(0, math.nan)).fillna(1.0)

    action_counts: Dict[str, int] = {}
    first_enters: List[Tuple[pd.Timestamp, Any]] = []
    first_exits: List[Tuple[pd.Timestamp, Any]] = []

    use_from_values = hasattr(mod, "generate_intent_from_values") and callable(getattr(mod, "generate_intent_from_values"))
    use_df_intent = hasattr(mod, "generate_intent") and callable(getattr(mod, "generate_intent"))

    if not use_from_values and not use_df_intent:
        raise RuntimeError("Strategy module has neither generate_intent_from_values nor generate_intent")

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

        if use_from_values:
            intent = _call_generate_intent_from_values(mod, candidates)
        else:
            # Fallback: call generate_intent on a growing window
            df_window = df.iloc[: i + 1].copy()
            intent = _call_generate_intent_df(mod, df_window, ctx={"p_long": None})

        act = _intent_get(intent, "action", "NONE")
        act_str = str(act)
        action_counts[act_str] = action_counts.get(act_str, 0) + 1

        if act_str == "ENTER_LONG" and len(first_enters) < 10:
            first_enters.append((ts, intent))
        if act_str == "EXIT_LONG" and len(first_exits) < 10:
            first_exits.append((ts, intent))

    print("[DBG] Action counts:")
    for k, v in sorted(action_counts.items(), key=lambda x: -x[1]):
        print(f"    {k:>20}: {v:,}")
    print()

    if first_enters:
        print("[DBG] First ENTER_LONG hits (up to 10):")
        for ts, it in first_enters:
            print(f"      {ts} -> {it}")
        print()

    if first_exits:
        print("[DBG] First EXIT_LONG hits (up to 10):")
        for ts, it in first_exits:
            print(f"      {ts} -> {it}")
        print()

    print("[DBG] Done.")


if __name__ == "__main__":
    main()
