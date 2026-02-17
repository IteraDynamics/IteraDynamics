"""
research.strategies
===================

Layer 2: Alpha Modules / Strategies (pluggable, deterministic, per-regime)

All strategies implement the same interface:
    def generate_intent(df, ctx, *, closed_only=True) -> dict

Available strategies:
    - sg_regime_trend_v1: Original trend-following with ADX gate
    - sg_core_exposure_v1: Volatility-scaled exposure by regime
    - sg_trend_probe_v1: TREND_UP only strict trend following
    - sg_vol_probe_v1: VOL_COMPRESSION breakout hunting
    - sg_stub_strategy: Minimal stub for testing

Usage (from runtime/argus context):
    from research.strategies.sg_core_exposure_v1 import generate_intent
    
    # Or dynamically:
    import importlib
    mod = importlib.import_module("research.strategies.sg_core_exposure_v1")
    fn = getattr(mod, "generate_intent")
"""

__all__ = [
    "sg_regime_trend_v1",
    "sg_core_exposure_v1",
    "sg_trend_probe_v1",
    "sg_vol_probe_v1",
    "sg_stub_strategy",
]
