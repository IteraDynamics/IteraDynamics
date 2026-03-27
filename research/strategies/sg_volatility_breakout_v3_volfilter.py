"""
Compatibility wrapper for the Argus runtime strategy module.

The runnable harness strategy implementation lives in:
  runtime/argus/research/strategies/sg_volatility_breakout_v3_volfilter.py
"""

from runtime.argus.research.strategies.sg_volatility_breakout_v3_volfilter import (  # noqa: F401
    ATR_LEN,
    ATR_MA_LEN,
    VOL_EXPANSION_MIN,
    generate_intent,
)

