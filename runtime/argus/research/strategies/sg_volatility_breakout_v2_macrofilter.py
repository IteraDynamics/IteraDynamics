"""
Volatility Breakout Strategy v2 (Macro Filter Research Variant)

ETH-focused additive variant of sg_volatility_breakout_v1:
  - keeps v1 breakout/exit logic intact
  - adds a macro regime gate for new LONG entries only:
      allow entry only when Close > EMA(2000)

Rationale:
  Reduce churn / false breakout participation in broad bear or weak-trend regimes.

Important:
  - Long-only (no shorts)
  - Exits remain exactly as v1 behavior
  - No Layer 3 / harness changes
  - No lookahead (EMA uses only bars available in current df slice)
"""

from __future__ import annotations

import copy
from typing import Any, Dict

import pandas as pd

from research.strategies import sg_volatility_breakout_v1 as v1

# Macro filter constant for this research variant.
MACRO_EMA_LEN = 2000


def _resolve_close_series(df: pd.DataFrame) -> pd.Series:
    cols = {str(c).strip().lower(): c for c in df.columns}
    if "close" in df.columns:
        return pd.to_numeric(df["close"], errors="coerce")
    if "Close" in df.columns:
        return pd.to_numeric(df["Close"], errors="coerce")
    src = cols.get("close") or cols.get("c")
    if src is None:
        return pd.Series(dtype=float)
    return pd.to_numeric(df[src], errors="coerce")


def _macro_regime_ok(df: pd.DataFrame) -> tuple[bool, float | None, float | None]:
    close = _resolve_close_series(df)
    close = close.dropna()
    if close.empty or len(close) < MACRO_EMA_LEN:
        last_close = float(close.iloc[-1]) if not close.empty else None
        return False, last_close, None
    ema = close.ewm(span=MACRO_EMA_LEN, adjust=False).mean()
    close_last = float(close.iloc[-1])
    ema_last = float(ema.iloc[-1])
    return close_last > ema_last, close_last, ema_last


def _is_entry_action(action: str) -> bool:
    a = str(action or "").strip().upper()
    return a in {"BUY", "ENTER", "ENTER_LONG", "LONG"}


def generate_intent(
    df: pd.DataFrame,
    state: Any = None,
    *,
    closed_only: bool = True,
    **kwargs: Any,
) -> Dict[str, Any]:
    """
    Wrapper around v1:
      1) evaluate v1 on a shadow state
      2) if action is entry and macro filter fails -> force HOLD/flat
      3) otherwise run v1 on real state and return unchanged behavior

    Using a shadow state prevents mutating real state when an entry is blocked.
    """
    shadow_state = copy.deepcopy(state) if isinstance(state, dict) else state
    probe_intent = v1.generate_intent(df, shadow_state, closed_only=closed_only, **kwargs)

    regime_ok, close_last, ema_last = _macro_regime_ok(df)
    if _is_entry_action(probe_intent.get("action")) and not regime_ok:
        base_meta = probe_intent.get("meta") if isinstance(probe_intent.get("meta"), dict) else {}
        meta = dict(base_meta)
        meta.update(
            {
                "strategy": "sg_volatility_breakout_v2_macrofilter",
                "base_strategy": "sg_volatility_breakout_v1",
                "macro_filter": f"close > ema{MACRO_EMA_LEN}",
                "macro_filter_pass": False,
                "macro_close": close_last,
                "macro_ema": ema_last,
            }
        )
        return {
            "action": "HOLD",
            "confidence": float(probe_intent.get("confidence", 0.0)),
            "desired_exposure_frac": 0.0,
            "horizon_hours": int(probe_intent.get("horizon_hours", v1.BREAKOUT_LOOKBACK)),
            "reason": (
                f"VolBreakout v2: macro filter blocked entry "
                f"(close <= EMA{MACRO_EMA_LEN} or insufficient history)"
            ),
            "meta": meta,
        }

    # If filter passes (or action is not entry), preserve v1 behavior exactly on real state.
    real_intent = v1.generate_intent(df, state, closed_only=closed_only, **kwargs)
    if isinstance(real_intent.get("meta"), dict):
        real_intent["meta"] = dict(real_intent["meta"])
        real_intent["meta"]["strategy"] = "sg_volatility_breakout_v2_macrofilter"
        real_intent["meta"]["base_strategy"] = "sg_volatility_breakout_v1"
        real_intent["meta"]["macro_filter"] = f"close > ema{MACRO_EMA_LEN}"
        real_intent["meta"]["macro_filter_pass"] = bool(regime_ok)
        real_intent["meta"]["macro_close"] = close_last
        real_intent["meta"]["macro_ema"] = ema_last
    return real_intent

