"""
research.regime
===============

Layer 1: Regime Engine - authoritative, stable, shared regime classification.

Usage (from runtime/argus context):
    from research.regime.regime_engine import classify_regime, RegimeState
"""

from .regime_engine import classify_regime, RegimeState, RegimeLabel

__all__ = ["classify_regime", "RegimeState", "RegimeLabel"]

