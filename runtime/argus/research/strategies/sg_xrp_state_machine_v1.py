"""
XRP State Machine v1 (Layer 2) — low-turnover research sleeve.

Mirror of ETH state machine with identical logic/constraints.
"""

from runtime.argus.research.strategies.sg_eth_state_machine_v1 import (  # noqa: F401
    ENTRY_BREAKOUT_BARS,
    ENTRY_CONFIDENCE,
    FAST_EMA,
    HOLD_CONFIDENCE,
    MAX_EXPOSURE,
    MAX_HOLD_BARS,
    MIN_BARS,
    MIN_HOLD_BARS,
    REENTRY_COOLDOWN_BARS,
    RISK_OFF_BUFFER,
    SLOW_EMA,
    TRAIL_WINDOW,
    _intent,
    _normalize_close,
)

from typing import Any, Dict

import pandas as pd

STATE_KEY = "_sg_xrp_state_machine_v1"


def generate_intent(
    df: pd.DataFrame,
    state: Any = None,
    *,
    closed_only: bool = True,
    **kwargs: Any,
) -> Dict[str, Any]:
    from runtime.argus.research.strategies.sg_eth_state_machine_v1 import generate_intent as _eth_generate_intent

    out = _eth_generate_intent(df, state=state, closed_only=closed_only, **kwargs)
    meta = out.get("meta")
    if isinstance(meta, dict):
        meta = dict(meta)
        meta["strategy"] = STATE_KEY
        out["meta"] = meta
    return out

