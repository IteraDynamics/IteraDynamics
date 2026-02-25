"""
research.portfolio
==================

Portfolio-level governance and cross-asset allocation.

Layer 2.5 / 3 boundary: maps per-asset (regime + intent) into deterministic
target portfolio weights. Closed-bar, stateless, no lookahead.

Usage (from repo root with research on path):
    from research.portfolio.cross_asset_allocator import (
        allocate_portfolio,
        AssetSignalBundle,
        PortfolioPolicy,
        AllocatorContext,
        PortfolioAllocationDecision,
        build_asset_bundle_from_layer_outputs,
    )
"""

from research.portfolio.cross_asset_allocator import (
    allocate_portfolio,
    validate_allocation,
    RegimeState as AllocatorRegimeLiteral,
    RegimeSnapshot,
    IntentAction,
    StrategyIntent,
    AssetSignalBundle,
    PortfolioPolicy,
    AllocatorContext,
    PortfolioAllocationDecision,
    build_asset_bundle_from_layer_outputs,
)

__all__ = [
    "allocate_portfolio",
    "validate_allocation",
    "AllocatorRegimeLiteral",
    "RegimeSnapshot",
    "IntentAction",
    "StrategyIntent",
    "AssetSignalBundle",
    "PortfolioPolicy",
    "AllocatorContext",
    "PortfolioAllocationDecision",
    "build_asset_bundle_from_layer_outputs",
]
