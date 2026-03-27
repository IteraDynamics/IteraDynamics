"""
Compatibility wrapper for the Argus runtime strategy module.

The runnable harness strategy implementation lives in:
  runtime/argus/research/strategies/sg_eth_regime_gated_mean_reversion_v1.py
"""

from runtime.argus.research.strategies.sg_eth_regime_gated_mean_reversion_v1 import (  # noqa: F401
    EXIT_RSI,
    HOLD_CONFIDENCE,
    MAX_EXPOSURE,
    OVERSOLD,
    REGIME_EMA_LEN,
    RSI_PERIOD,
    Z_ENTRY,
    Z_EXIT,
    Z_WINDOW,
    generate_intent,
)

