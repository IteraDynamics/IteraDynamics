from __future__ import annotations
import pandas as pd
from runtime.argus.apex_core.strategy_intent import Intent, Action

def generate_intent(df, ctx):
    """
    External Strategy API for SG.

    Args:
        df: pandas DataFrame of flight_recorder candles
        ctx: StrategyContext (from SG)

    Returns:
        StrategyIntent OR dict with at least {"action": "..."}
    """
    # simplest stub: always HOLD
    return {
        "action": "HOLD",
        "reason": "stub strategy loaded (no-op)",
        "meta": {"source": "sg_stub_strategy"},
    }
