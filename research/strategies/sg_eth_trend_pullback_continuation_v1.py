"""
Compatibility wrapper for the Argus runtime strategy module.

The runnable harness strategy implementation lives in:
  runtime/argus/research/strategies/sg_eth_trend_pullback_continuation_v1.py
"""

from runtime.argus.research.strategies.sg_eth_trend_pullback_continuation_v1 import (  # noqa: F401
    FAST_EMA_BARS,
    HOLD_CONFIDENCE,
    MAX_EXPOSURE,
    PULLBACK_BARS,
    REENTRY_BREAKOUT_BARS,
    SLOW_EMA_BARS,
    generate_intent,
)

