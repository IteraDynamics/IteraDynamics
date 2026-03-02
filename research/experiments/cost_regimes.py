# research/experiments/cost_regimes.py
"""
Cost regimes for portfolio geometry: retail vs pro vs institutional friction.

Used by portfolio_geometry_validation.py to model launch friction without changing strategy logic.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

# Regime names (must match CLI choices)
COST_REGIME_RETAIL_LAUNCH = "retail_launch"
COST_REGIME_PRO_TARGET = "pro_target"
COST_REGIME_INSTITUTIONAL = "institutional"
COST_REGIME_CUSTOM = "custom"

# Default fee_bps, slippage_bps per regime
COST_REGIME_DEFAULTS: Dict[str, Tuple[float, float]] = {
    COST_REGIME_RETAIL_LAUNCH: (120.0, 10.0),
    COST_REGIME_PRO_TARGET: (10.0, 5.0),
    COST_REGIME_INSTITUTIONAL: (2.0, 2.0),
}

# For custom we use caller's defaults (e.g. pro_target values)
CUSTOM_DEFAULT_FEE_BPS = 10.0
CUSTOM_DEFAULT_SLIPPAGE_BPS = 5.0


def resolve_cost_params(
    cost_regime: str,
    fee_bps_cli: Any = None,
    slippage_bps_cli: Any = None,
    custom_fee_bps: float = CUSTOM_DEFAULT_FEE_BPS,
    custom_slippage_bps: float = CUSTOM_DEFAULT_SLIPPAGE_BPS,
) -> Tuple[float, float]:
    """
    Resolve fee_bps and slippage_bps from cost regime and optional CLI overrides.

    - If cost_regime != custom: use regime defaults unless fee_bps_cli / slippage_bps_cli
      are explicitly provided (not None). CLI overrides win.
    - If cost_regime == custom: use custom_fee_bps and custom_slippage_bps (typically
      from script defaults or explicit --fee_bps/--slippage_bps).

    Returns (fee_bps, slippage_bps).
    """
    if cost_regime == COST_REGIME_CUSTOM:
        fee = float(custom_fee_bps) if custom_fee_bps is not None else CUSTOM_DEFAULT_FEE_BPS
        slip = float(custom_slippage_bps) if custom_slippage_bps is not None else CUSTOM_DEFAULT_SLIPPAGE_BPS
        if fee_bps_cli is not None:
            fee = float(fee_bps_cli)
        if slippage_bps_cli is not None:
            slip = float(slippage_bps_cli)
        return fee, slip

    regime_defaults = COST_REGIME_DEFAULTS.get(
        cost_regime, (CUSTOM_DEFAULT_FEE_BPS, CUSTOM_DEFAULT_SLIPPAGE_BPS)
    )
    fee = float(fee_bps_cli) if fee_bps_cli is not None else regime_defaults[0]
    slip = float(slippage_bps_cli) if slippage_bps_cli is not None else regime_defaults[1]
    return fee, slip
