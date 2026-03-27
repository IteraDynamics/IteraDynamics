"""
ETH Regime-Gated Mean Reversion v1 (Layer 2) — research sleeve candidate.

Screener-aligned fixed config:
  regime_ema_len=200
  rsi_period=14
  oversold=28
  exit_rsi=52
  z_window=72
  z_entry=-1.8
  z_exit=-0.3

Long-only intent:
  - Entry only in up-regime (close >= EMA(200)).
  - Require both RSI oversold and z-score stretch.
  - Exit on RSI recovery, z-score normalization, or regime break.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd

REGIME_EMA_LEN = 200
RSI_PERIOD = 14
OVERSOLD = 28.0
EXIT_RSI = 52.0
Z_WINDOW = 72
Z_ENTRY = -1.8
Z_EXIT = -0.3

ENTRY_CONFIDENCE = 0.62
HOLD_CONFIDENCE = 0.42
MAX_EXPOSURE = 0.985
STATE_KEY = "_sg_eth_regime_gated_mean_reversion_v1"
MIN_BARS = max(REGIME_EMA_LEN + 2, Z_WINDOW + 2, 256)
TRAIL_WINDOW = 900


def _normalize_ohlc(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy(deep=False)
    cols = {str(c).strip().lower(): c for c in out.columns}
    if "close" not in out.columns:
        src = cols.get("close")
        if src is not None:
            out["close"] = pd.to_numeric(out[src], errors="coerce")
    return out


def _intent_dict(
    action: str,
    confidence: float,
    desired_exposure_frac: float,
    reason: str,
    meta: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "action": action,
        "confidence": float(confidence),
        "desired_exposure_frac": float(desired_exposure_frac),
        "horizon_hours": int(Z_WINDOW),
        "reason": reason,
        "meta": meta,
    }


def generate_intent(
    df: pd.DataFrame,
    state: Any = None,
    *,
    closed_only: bool = True,
    **kwargs: Any,
) -> Dict[str, Any]:
    if df is None or df.empty:
        return _intent_dict("HOLD", 0.0, 0.0, "RegimeGatedMR: no data", {"strategy": STATE_KEY})

    if len(df) < MIN_BARS:
        return _intent_dict("HOLD", 0.0, 0.0, "RegimeGatedMR: insufficient bars", {"strategy": STATE_KEY})

    w = _normalize_ohlc(df.tail(TRAIL_WINDOW))
    if "close" not in w.columns:
        return _intent_dict("HOLD", 0.0, 0.0, "RegimeGatedMR: missing close", {"strategy": STATE_KEY})

    close = w["close"].astype(float)
    ema_regime = close.ewm(span=REGIME_EMA_LEN, adjust=False).mean()
    regime_ok = bool(close.iloc[-1] >= ema_regime.iloc[-2])

    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.rolling(RSI_PERIOD, min_periods=1).mean()
    avg_loss = loss.rolling(RSI_PERIOD, min_periods=1).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = (100.0 - (100.0 / (1.0 + rs))).fillna(50.0)

    mu = close.rolling(Z_WINDOW, min_periods=1).mean()
    sd = close.rolling(Z_WINDOW, min_periods=1).std(ddof=0).replace(0.0, np.nan)
    z = ((close - mu) / sd).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    entry_triggered = bool(regime_ok and (rsi.iloc[-2] <= OVERSOLD) and (z.iloc[-2] <= Z_ENTRY))
    exit_triggered = bool((rsi.iloc[-2] >= EXIT_RSI) or (z.iloc[-2] >= Z_EXIT) or (close.iloc[-1] < ema_regime.iloc[-2]))

    ctx = state if isinstance(state, dict) else None
    if ctx is not None:
        st = ctx.setdefault(STATE_KEY, {"in_trade": False})
    else:
        st = {"in_trade": False}
    in_trade = bool(st.get("in_trade", False))

    if in_trade and exit_triggered:
        if ctx is not None:
            ctx[STATE_KEY] = {"in_trade": False}
        return _intent_dict(
            "EXIT",
            HOLD_CONFIDENCE,
            0.0,
            "RegimeGatedMR: exit (recovery or regime break)",
            {"strategy": STATE_KEY, "in_trade": False},
        )

    if (not in_trade) and entry_triggered:
        if ctx is not None:
            ctx[STATE_KEY] = {"in_trade": True}
        return _intent_dict(
            "BUY",
            ENTRY_CONFIDENCE,
            MAX_EXPOSURE,
            "RegimeGatedMR: long (up-regime + RSI/z stretch)",
            {"strategy": STATE_KEY, "in_trade": True},
        )

    if in_trade:
        return _intent_dict(
            "BUY",
            HOLD_CONFIDENCE,
            MAX_EXPOSURE,
            "RegimeGatedMR: hold long",
            {"strategy": STATE_KEY, "in_trade": True},
        )

    return _intent_dict(
        "HOLD",
        HOLD_CONFIDENCE,
        0.0,
        "RegimeGatedMR: flat (no qualified setup)",
        {"strategy": STATE_KEY, "in_trade": False},
    )

