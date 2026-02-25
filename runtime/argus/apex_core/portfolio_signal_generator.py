"""
Portfolio-level signal generation + allocation (multi-asset).

Deterministic, closed-bar-only. Builds AssetSignalBundle per product from
Layer 1 (regime) and Layer 2 (intent) outputs, then calls the shared allocator.
No lookahead; no modification of Layer 1/Layer 2 or single-asset signal_generator.

Uses: research.portfolio.cross_asset_allocator
  (caller must have repo root on path so research.portfolio resolves.)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Mapping, Optional

from research.portfolio.cross_asset_allocator import (
    allocate_portfolio,
    build_asset_bundle_from_layer_outputs,
    AllocatorContext,
    PortfolioAllocationDecision,
    PortfolioPolicy,
)

if TYPE_CHECKING:
    import pandas as pd


def generate_portfolio_decision(
    dfs_by_product: Mapping[str, "pd.DataFrame"],
    layer1_outputs_by_product: Mapping[str, Any],
    layer2_intents_by_product: Mapping[str, Any],
    policy: PortfolioPolicy,
    bar_ts_utc: str,
    prev_target_weights: Optional[Dict[str, float]],
) -> PortfolioAllocationDecision:
    """
    Build per-product signal bundles from Layer 1 + Layer 2 outputs, run allocator, return decision.

    Deterministic: product order is sorted. DataFrames are not passed into the allocator
    (kept for context only; may be unused). No side effects; returns allocation decision only.
    """
    # Deterministic iteration over products that have both Layer 1 and Layer 2 data
    product_ids = sorted(
        set(layer1_outputs_by_product.keys()) & set(layer2_intents_by_product.keys())
    )
    asset_bundles: Dict[str, Any] = {}
    for product_id in product_ids:
        layer1_out = layer1_outputs_by_product[product_id]
        layer2_out = layer2_intents_by_product[product_id]
        bundle = build_asset_bundle_from_layer_outputs(
            product_id,
            bar_ts_utc,
            layer1_out,
            layer2_out,
            features=None,
        )
        asset_bundles[product_id] = bundle

    ctx = AllocatorContext(
        bar_ts_utc=bar_ts_utc,
        prev_target_weights=prev_target_weights,
    )
    decision = allocate_portfolio(asset_bundles, policy, ctx)

    # Add minimal deterministic meta for audit (products list)
    meta = dict(decision.meta)
    meta["products"] = product_ids
    return PortfolioAllocationDecision(
        bar_ts_utc=decision.bar_ts_utc,
        target_weights=decision.target_weights,
        cash_weight=decision.cash_weight,
        reason=decision.reason,
        meta=meta,
    )
