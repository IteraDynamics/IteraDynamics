"""
research.harness
================

Smoke tests and validation harnesses for the 3-layer architecture.

- regime_smoke: Validates Layer 1 (Regime Engine)
- strategy_smoke: Validates Layer 2 (Strategy)
"""

from . import regime_smoke
from . import strategy_smoke

__all__ = ["regime_smoke", "strategy_smoke"]

