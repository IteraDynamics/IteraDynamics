"""
research.strategies
===================

Layer 2: Alpha Modules / Strategies (pluggable, deterministic, per-regime)

Available strategies:
    - sg_regime_trend_v1: Main trend-following strategy with regime integration
    - sg_stub_strategy: Minimal stub for testing SG external strategy plumbing

Usage (from runtime/argus context):
    from research.strategies.sg_regime_trend_v1 import generate_intent
    from research.strategies.sg_stub_strategy import generate_intent as stub_intent
"""

__all__ = ["sg_regime_trend_v1", "sg_stub_strategy"]

