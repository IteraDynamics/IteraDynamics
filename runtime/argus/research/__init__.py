"""
research
========

Three-layer architecture for Argus trading system.

Layer 1: Regime Engine (authoritative, stable, shared)
    from research.regime import classify_regime, RegimeState

Layer 2: Alpha Modules / Strategies (pluggable, deterministic, per-regime)
    from research.strategies import sg_regime_trend_v1, sg_stub_strategy

Harness: Smoke tests and validation
    from research.harness import regime_smoke, strategy_smoke

IMPORTANT:
    This package must be imported when sys.path includes runtime/argus.
    All imports resolve relative to that path.
"""

# Lazy imports to avoid circular dependencies
__all__ = [
    "regime",
    "strategies",
    "harness",
]


def __getattr__(name: str):
    """Lazy import submodules."""
    if name == "regime":
        from . import regime
        return regime
    if name == "strategies":
        from . import strategies
        return strategies
    if name == "harness":
        from . import harness
        return harness
    raise AttributeError(f"module 'research' has no attribute '{name}'")

