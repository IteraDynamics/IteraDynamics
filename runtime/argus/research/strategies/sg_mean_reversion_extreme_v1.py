"""
Mean Reversion Extreme Strategy (Layer 2)

Long-only mean reversion sleeve candidate: enter after extreme downside (RSI oversold),
exit on RSI recovery or max hold. Intended as defensive / crash-alpha, not BTC replacement.

Research baseline:
- RSI length = 21, oversold <= 25, exit RSI > 50, max hold = 48 hours.

Configuration via environment (defaults = research baseline):
- SG_MR_RSI_LEN (default 21)
- SG_MR_OVERSOLD (default 25)
- SG_MR_EXIT_RSI (default 50)
- SG_MR_MAX_HOLD_BARS (default 48)
- SG_MR_MAX_EXPOSURE (default 0.985)

Contract: generate_intent(df, ctx, *, closed_only=True) -> dict
- action: ENTER_LONG | EXIT | HOLD
- desired_exposure_frac: 0 or max_exposure
- Closed-bar deterministic, no lookahead.

Author: Itera Dynamics
"""

from __future__ import annotations

import os
from typing import Any, Dict

import numpy as np
import pandas as pd


# ---------------------------
# Env config
# ---------------------------

def _get_config() -> Dict[str, Any]:
    """Read config from env with research defaults."""
    def _int(name: str, default: int) -> int:
        v = os.getenv(name, "").strip()
        if not v:
            return default
        try:
            return int(float(v))
        except Exception:
            return default

    def _float(name: str, default: float) -> float:
        v = os.getenv(name, "").strip()
        if not v:
            return default
        try:
            return float(v)
        except Exception:
            return default

    return {
        "rsi_len": _int("SG_MR_RSI_LEN", 21),
        "oversold": _float("SG_MR_OVERSOLD", 25.0),
        "exit_rsi": _float("SG_MR_EXIT_RSI", 50.0),
        "max_hold_bars": _int("SG_MR_MAX_HOLD_BARS", 48),
        "max_exposure": min(1.0, max(0.0, _float("SG_MR_MAX_EXPOSURE", 0.985))),
    }


def _normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure we have lowercase 'close' (and optional others). Leaves original cols."""
    if df is None or df.empty:
        return df
    out = df.copy(deep=False)
    cols = {str(c).strip().lower(): c for c in df.columns}
    if "close" not in out.columns and ("close" in cols or "c" in cols):
        src = cols.get("close") or cols.get("c")
        if src is not None:
            out["close"] = out[src].astype(float)
    if "Close" in df.columns and "close" not in out.columns:
        out["close"] = df["Close"].astype(float)
    return out


def _rsi(close: pd.Series, period: int) -> pd.Series:
    """
    RSI (Wilder-style smoothing). Uses only past and current bar; no lookahead.
    """
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100.0 - (100.0 / (1.0 + rs))


def _simulate_state(
    close: pd.Series,
    rsi_len: int,
    oversold: float,
    exit_rsi: float,
    max_hold_bars: int,
) -> tuple[bool, str, float, int | None]:
    """
    Walk bars from first valid RSI and determine state at the last bar.
    Returns: (in_long, action, rsi_current, entry_bar_idx or None).
    All logic uses only data through current bar (closed-bar, no lookahead).
    """
    if len(close) < rsi_len:
        return False, "HOLD", float("nan"), None

    rsi_series = _rsi(close, rsi_len)
    in_long = False
    entry_bar_idx: int | None = None
    exited_this_bar = False  # True if we were long and exited on the final bar

    for i in range(rsi_len - 1, len(close)):
        r = rsi_series.iloc[i]
        if not np.isfinite(r):
            continue

        if not in_long:
            if r <= oversold:
                in_long = True
                entry_bar_idx = i
        else:
            assert entry_bar_idx is not None
            bars_held = i - entry_bar_idx
            if r > exit_rsi or bars_held >= max_hold_bars:
                if i == len(close) - 1:
                    exited_this_bar = True
                in_long = False
                entry_bar_idx = None

    last_rsi = float(rsi_series.iloc[-1]) if np.isfinite(rsi_series.iloc[-1]) else float("nan")
    if in_long:
        action = "ENTER_LONG"  # maintain long; harness treats as target exposure
        return True, action, last_rsi, entry_bar_idx
    if exited_this_bar:
        return False, "EXIT", last_rsi, None
    return False, "HOLD", last_rsi, None


def _action_dict(
    action: str,
    confidence: float,
    desired_exposure_frac: float,
    horizon_hours: int | None,
    reason: str,
    meta: Dict[str, Any],
) -> Dict[str, Any]:
    """Build intent dict per harness contract."""
    return {
        "action": action,
        "confidence": float(confidence),
        "desired_exposure_frac": float(desired_exposure_frac),
        "horizon_hours": horizon_hours if horizon_hours is not None else 0,
        "reason": reason,
        "meta": meta,
    }


# ---------------------------
# Public API
# ---------------------------

def generate_intent(
    df: pd.DataFrame,
    ctx: Any,
    *,
    closed_only: bool = True,
    **kwargs: Any,
) -> Dict[str, Any]:
    """
    Mean Reversion Extreme intent from OHLCV history.

    - Long-only. Entry when RSI(21) <= 25; exit when RSI > 50 or hold >= 48 bars.
    - Uses only closed bars (last row = current bar close); no lookahead.
    - Same df + params => same intent (deterministic).

    Args:
        df: OHLCV DataFrame (tail = most recent bar). Must have Close (or close).
        ctx: StrategyContext (optional, unused).
        closed_only: Ignored; behavior is always closed-bar.
        **kwargs: Ignored for compatibility.

    Returns:
        Dict: action, confidence, desired_exposure_frac, horizon_hours, reason, meta.
    """
    cfg = _get_config()
    df0 = _normalize_ohlcv(df)
    if df0.empty or "close" not in df0.columns:
        return _action_dict(
            action="HOLD",
            confidence=0.0,
            desired_exposure_frac=0.0,
            horizon_hours=0,
            reason="MeanReversion: insufficient or missing close series",
            meta={"strategy": "sg_mean_reversion_extreme_v1"},
        )

    close = df0["close"].astype(float)
    in_long, action, rsi_val, entry_bar_idx = _simulate_state(
        close,
        rsi_len=cfg["rsi_len"],
        oversold=cfg["oversold"],
        exit_rsi=cfg["exit_rsi"],
        max_hold_bars=cfg["max_hold_bars"],
    )

    # Map to exposure and reason
    if action == "ENTER_LONG":
        desired = cfg["max_exposure"]
        reason = f"MeanReversion: long (RSI={rsi_val:.1f} oversold<=25, hold until RSI>{cfg['exit_rsi']} or {cfg['max_hold_bars']} bars)"
        confidence = 0.7
        horizon_hours = cfg["max_hold_bars"]  # 48h default
    elif action == "EXIT":
        desired = 0.0
        reason = f"MeanReversion: exit (RSI recovery >{cfg['exit_rsi']} or max hold {cfg['max_hold_bars']} bars)"
        confidence = 0.7
        horizon_hours = 0
    else:
        desired = 0.0
        reason = f"MeanReversion: flat (RSI={rsi_val:.1f}, no oversold entry)"
        confidence = 0.5
        horizon_hours = 0

    return _action_dict(
        action=action,
        confidence=confidence,
        desired_exposure_frac=desired,
        horizon_hours=horizon_hours,
        reason=reason,
        meta={
            "strategy": "sg_mean_reversion_extreme_v1",
            "rsi": rsi_val,
            "in_long": in_long,
            "entry_bar_idx": entry_bar_idx,
        },
    )
