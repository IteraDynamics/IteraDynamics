"""
ETH State Machine v1 (Layer 2) — low-turnover research sleeve.

Design goals:
  - Long-only ETH sleeve candidate with explicit low-turnover constraints.
  - Regime-aware behavior via EMA trend state.
  - Deterministic closed-bar logic, no lookahead.

State machine:
  - RISK_ON_TREND: close >= EMA(200) and EMA(50) > EMA(200)
  - NEUTRAL: otherwise
  - RISK_OFF: deep break below EMA(200)

Entry (only in RISK_ON_TREND):
  - pullback-confirmation style: close >= EMA(50) and close breaks prior 24-bar high
  - blocked by cooldown bars after an exit

Exit (while in trade):
  - risk-off break OR close below EMA(200)
  - optional time stop (max hold bars)
  - min-hold lock blocks early exits except risk-off hard break
"""

from __future__ import annotations

from typing import Any, Dict

import pandas as pd

FAST_EMA = 50
SLOW_EMA = 200
ENTRY_BREAKOUT_BARS = 24

MIN_HOLD_BARS = 72       # ~3 days on 1h bars
REENTRY_COOLDOWN_BARS = 96  # ~4 days cooldown after any exit
MAX_HOLD_BARS = 720      # ~30 days hard cap

RISK_OFF_BUFFER = 0.02   # 2% below slow EMA => hard risk-off

ENTRY_CONFIDENCE = 0.62
HOLD_CONFIDENCE = 0.45
MAX_EXPOSURE = 0.80

STATE_KEY = "_sg_eth_state_machine_v1"
MIN_BARS = max(SLOW_EMA + 5, ENTRY_BREAKOUT_BARS + 5, 260)
TRAIL_WINDOW = 1200


def _normalize_close(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy(deep=False)
    cols = {str(c).strip().lower(): c for c in out.columns}
    if "close" not in out.columns:
        src = cols.get("close")
        if src is not None:
            out["close"] = pd.to_numeric(out[src], errors="coerce")
    return out


def _intent(action: str, conf: float, expo: float, reason: str, meta: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "action": action,
        "confidence": float(conf),
        "desired_exposure_frac": float(expo),
        "horizon_hours": int(ENTRY_BREAKOUT_BARS),
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
        return _intent("HOLD", 0.0, 0.0, "StateMachine: no data", {"strategy": STATE_KEY})
    if len(df) < MIN_BARS:
        return _intent("HOLD", 0.0, 0.0, "StateMachine: insufficient bars", {"strategy": STATE_KEY})

    w = _normalize_close(df.tail(TRAIL_WINDOW))
    if "close" not in w.columns:
        return _intent("HOLD", 0.0, 0.0, "StateMachine: missing close", {"strategy": STATE_KEY})

    close = w["close"].astype(float)
    ema_fast = close.ewm(span=FAST_EMA, adjust=False).mean()
    ema_slow = close.ewm(span=SLOW_EMA, adjust=False).mean()
    breakout_high = close.rolling(ENTRY_BREAKOUT_BARS, min_periods=1).max().shift(1)

    close_now = float(close.iloc[-1])
    fast_prev = float(ema_fast.iloc[-2])
    slow_prev = float(ema_slow.iloc[-2])
    bo_prev = float(breakout_high.iloc[-1])

    trend_up = (close_now >= slow_prev) and (fast_prev > slow_prev)
    risk_off_hard = close_now <= (slow_prev * (1.0 - RISK_OFF_BUFFER))

    if risk_off_hard:
        regime = "RISK_OFF"
    elif trend_up:
        regime = "RISK_ON_TREND"
    else:
        regime = "NEUTRAL"

    entry_trigger = bool((regime == "RISK_ON_TREND") and (close_now >= fast_prev) and (close_now >= bo_prev))
    exit_trigger_soft = bool(close_now < slow_prev)
    exit_trigger = bool(risk_off_hard or exit_trigger_soft)

    ctx = state if isinstance(state, dict) else None
    if ctx is not None:
        st = ctx.setdefault(
            STATE_KEY,
            {
                "in_trade": False,
                "bars_in_trade": 0,
                "cooldown_remaining": 0,
            },
        )
    else:
        st = {"in_trade": False, "bars_in_trade": 0, "cooldown_remaining": 0}

    in_trade = bool(st.get("in_trade", False))
    bars_in_trade = int(st.get("bars_in_trade", 0))
    cooldown = max(0, int(st.get("cooldown_remaining", 0)))

    # Progress state counters each bar.
    if in_trade:
        bars_in_trade += 1
    if cooldown > 0:
        cooldown -= 1

    can_exit_by_hold = bars_in_trade >= MIN_HOLD_BARS
    hard_exit = risk_off_hard
    time_exit = bars_in_trade >= MAX_HOLD_BARS

    if in_trade and (hard_exit or time_exit or (exit_trigger and can_exit_by_hold)):
        if ctx is not None:
            ctx[STATE_KEY] = {
                "in_trade": False,
                "bars_in_trade": 0,
                "cooldown_remaining": REENTRY_COOLDOWN_BARS,
            }
        why = "hard_risk_off_exit" if hard_exit else ("max_hold_exit" if time_exit else "trend_exit_after_min_hold")
        return _intent(
            "EXIT",
            HOLD_CONFIDENCE,
            0.0,
            f"StateMachine: {why}",
            {"strategy": STATE_KEY, "regime": regime, "cooldown_remaining": REENTRY_COOLDOWN_BARS},
        )

    if (not in_trade) and entry_trigger and cooldown == 0:
        if ctx is not None:
            ctx[STATE_KEY] = {
                "in_trade": True,
                "bars_in_trade": 0,
                "cooldown_remaining": 0,
            }
        return _intent(
            "BUY",
            ENTRY_CONFIDENCE,
            MAX_EXPOSURE,
            "StateMachine: enter trend continuation",
            {"strategy": STATE_KEY, "regime": regime, "cooldown_remaining": 0},
        )

    if in_trade:
        if ctx is not None:
            ctx[STATE_KEY] = {
                "in_trade": True,
                "bars_in_trade": bars_in_trade,
                "cooldown_remaining": cooldown,
            }
        return _intent(
            "BUY",
            HOLD_CONFIDENCE,
            MAX_EXPOSURE,
            "StateMachine: hold long",
            {"strategy": STATE_KEY, "regime": regime, "bars_in_trade": bars_in_trade, "cooldown_remaining": cooldown},
        )

    if ctx is not None:
        ctx[STATE_KEY] = {
            "in_trade": False,
            "bars_in_trade": 0,
            "cooldown_remaining": cooldown,
        }
    return _intent(
        "HOLD",
        HOLD_CONFIDENCE,
        0.0,
        "StateMachine: flat",
        {"strategy": STATE_KEY, "regime": regime, "cooldown_remaining": cooldown},
    )

