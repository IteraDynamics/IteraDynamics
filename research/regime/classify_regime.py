"""
Canonical Layer 1: classify_regime.
Re-exports from shared implementation (runtime.argus.research.regime.regime_engine)
so repo-root imports resolve when sys.path includes repo root.
"""
from runtime.argus.research.regime.regime_engine import classify_regime

__all__ = ["classify_regime"]
