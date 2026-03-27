"""
Compatibility wrapper for the Argus runtime strategy module.

The runnable harness strategy implementation lives in:
  runtime/argus/research/strategies/sg_eth_state_machine_v1.py
"""

from runtime.argus.research.strategies.sg_eth_state_machine_v1 import (  # noqa: F401
    ENTRY_BREAKOUT_BARS,
    FAST_EMA,
    HOLD_CONFIDENCE,
    MAX_EXPOSURE,
    MAX_HOLD_BARS,
    MIN_HOLD_BARS,
    REENTRY_COOLDOWN_BARS,
    RISK_OFF_BUFFER,
    SLOW_EMA,
    generate_intent,
)

