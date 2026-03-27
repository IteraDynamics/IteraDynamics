"""
Compatibility wrapper for the Argus runtime strategy module.

The runnable harness strategy implementation lives in:
  runtime/argus/research/strategies/sg_eth_volume_confirmed_breakout_v1.py
"""

from runtime.argus.research.strategies.sg_eth_volume_confirmed_breakout_v1 import (  # noqa: F401
    BREAKOUT_LOOKBACK,
    ENTRY_CONFIDENCE,
    HOLD_CONFIDENCE,
    MAX_EXPOSURE,
    VOL_MULT,
    VOLUME_MA_BARS,
    generate_intent,
)

