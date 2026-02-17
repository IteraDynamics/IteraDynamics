"""
SG Stub Strategy
================

Minimal external strategy implementation for testing SG plumbing.

This module demonstrates the correct API expected by Signal Generator:
    - generate_intent(df, ctx) -> dict compatible with StrategyIntent(**dict)

Usage (from runtime/argus context):
    from research.strategies.sg_stub_strategy import generate_intent
"""

from __future__ import annotations

from typing import Any, Dict

import pandas as pd


def generate_intent(df: pd.DataFrame, ctx: Any, *, closed_only: bool = True) -> Dict[str, Any]:
    """
    External Strategy API for SG.

    Args:
        df: pandas DataFrame of OHLCV candles (from flight_recorder)
        ctx: StrategyContext (from SG) - dict-like object with mode, dry_run, etc.
        closed_only: If True, decisions should be based on closed candles only.
                     (stub ignores this but accepts for API compatibility)

    Returns:
        dict with keys: action, confidence, desired_exposure_frac, horizon_hours, reason, meta

    Valid actions: "ENTER_LONG", "EXIT_LONG", "HOLD", "FLAT"
    """
    # Extract context info for meta
    mode = None
    dry_run = None
    if isinstance(ctx, dict):
        mode = ctx.get("mode")
        dry_run = ctx.get("dry_run")
    elif ctx is not None:
        mode = getattr(ctx, "mode", None)
        dry_run = getattr(ctx, "dry_run", None)

    return {
        "action": "HOLD",
        "confidence": 0.0,
        "desired_exposure_frac": 0.0,
        "horizon_hours": 48,
        "reason": "stub_strategy_loaded",
        "meta": {
            "source": "sg_stub_strategy",
            "module_file": __file__,
            "mode": mode,
            "dry_run": dry_run,
            "closed_only": closed_only,
            "df_rows": len(df) if df is not None else 0,
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# Module info for verification
# ─────────────────────────────────────────────────────────────────────────────

__file_info__ = {
    "module": "research.strategies.sg_stub_strategy",
    "layer": 2,
    "description": "Stub strategy for testing SG external strategy plumbing",
}
