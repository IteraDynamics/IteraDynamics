# =====================================================================
# Itera Dynamics — Cross-Asset Allocator Contract (LOCKED)
#
# Purpose:
#   Portfolio-level governance that maps per-asset (regime + intent)
#   into deterministic target portfolio weights.
#
# Guarantees:
#   - Deterministic
#   - Closed-bar only (caller supplies only closed-bar inputs)
#   - Stateless (no persistence)
#   - No side effects
#   - No lookahead (allocator cannot access future bars)
# =====================================================================

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Optional, Mapping, Literal

__all__ = [
    "RegimeState",
    "RegimeSnapshot",
    "IntentAction",
    "StrategyIntent",
    "AssetSignalBundle",
    "PortfolioPolicy",
    "AllocatorContext",
    "PortfolioAllocationDecision",
    "allocate_portfolio",
    "validate_allocation",
    "build_asset_bundle_from_layer_outputs",
]


# -----------------------------
# Shared / canonical enums
# -----------------------------

RegimeState = Literal[
    "TREND_UP",
    "TREND_DOWN",
    "VOL_COMPRESSION",
    "VOL_EXPANSION",
    "RANGE",
    "UNKNOWN",
]

IntentAction = Literal[
    "HOLD",   # Maintain current exposure (governors still apply)
    "ENTER",  # Move toward desired exposure
    "EXIT",   # Move toward 0 exposure
]


# -----------------------------
# Layer 2 output (already defined in your docs)
# -----------------------------

@dataclass(frozen=True)
class StrategyIntent:
    action: IntentAction
    confidence: float                  # [0, 1]
    desired_exposure_frac: float       # typically [0, 1], but not enforced here
    horizon_hours: int
    reason: str
    meta: Dict[str, Any]


# -----------------------------
# Layer 1 output wrapper (allocator only needs regime label + confidence)
# -----------------------------

@dataclass(frozen=True)
class RegimeSnapshot:
    regime: RegimeState
    confidence: float                  # [0, 1]
    meta: Dict[str, Any]               # must be deterministic + closed-bar derived


# -----------------------------
# Per-asset bundle: "what the world looks like" for that asset at the bar close
# This is the ONLY per-asset input the allocator is allowed to consume.
# -----------------------------

@dataclass(frozen=True)
class AssetSignalBundle:
    product_id: str                    # e.g., "BTC-USD", "ETH-USD"
    bar_ts_utc: str                   # ISO8601 of the CLOSED bar end timestamp
    regime: RegimeSnapshot
    intent: StrategyIntent             # typically Core intent for that product
    # Optional, but allowed: purely informational, closed-bar computed stats
    # DO NOT include raw df, rolling series, or any stateful objects here.
    features: Optional[Dict[str, float]] = None


# -----------------------------
# Portfolio constraints (governance inputs to allocator)
# These are policy knobs, not alpha.
# -----------------------------

@dataclass(frozen=True)
class PortfolioPolicy:
    # Hard limits
    # LOCKED: max_gross_exposure is NOT applied inside allocate_portfolio. This allocator
    # outputs target_weights that sum to 1.0 (or sum + cash_weight == 1.0). Layer 3 applies
    # max_gross_exposure when converting weights to per-asset target exposure:
    #   target_exposure_i = weight_i * max_gross_exposure  (e.g. 0.25 weight * 1.0 = 25% of portfolio).
    max_gross_exposure: float          # e.g., 1.0 for fully invested long-only
    max_weight_per_asset: float        # e.g., 0.85 to preserve BTC dominance
    min_weight_per_asset: float        # e.g., 0.0

    # Optional preference: allow explicit cash weight. When True, cash is only created when
    # total desired exposure is zero (all EXIT). Caps do not create cash (Option A: caps redistribute).
    allow_cash: bool = True

    # Turnover governance (allocator emits targets; Layer 3 may enforce execution pacing)
    max_weight_delta_per_rebalance: Optional[float] = None  # e.g., 0.10


# -----------------------------
# Allocator context (portfolio-level, closed-bar only)
# This is NOT market data; it's run configuration + optional previous targets.
# -----------------------------

@dataclass(frozen=True)
class AllocatorContext:
    bar_ts_utc: str
    # If provided, allocator can be turnover-aware without persisting state.
    # Caller supplies previous target weights from Layer 3 state (deterministically stored).
    prev_target_weights: Optional[Dict[str, float]] = None

    # Deterministic fee/slippage knobs for research evaluation (allocator may ignore).
    fee_bps: Optional[float] = None
    slippage_bps: Optional[float] = None


# -----------------------------
# Allocator output (portfolio decision)
# This is what Layer 3 consumes to set per-asset exposure targets.
# -----------------------------

@dataclass(frozen=True)
class PortfolioAllocationDecision:
    bar_ts_utc: str

    # Target weights by product_id. Must be deterministic. Invariant: sum(target_weights) + cash_weight == 1.0.
    # LOCKED: This allocator does NOT apply max_gross_exposure. Layer 3 multiplies each weight by
    # policy.max_gross_exposure to obtain per-asset target exposure (e.g. weight 0.25 * 1.0 = 25% gross).
    target_weights: Dict[str, float]

    # Cash target weight. Only non-zero when policy.allow_cash=True and total desired exposure is zero (all EXIT).
    # Caps never create cash; they redistribute (Option A).
    cash_weight: float

    # Diagnostics for auditability / narrative; must be deterministic.
    reason: str
    meta: Dict[str, Any]


# -------------------------------------------------------------------------
# Compatibility: build allocator inputs from existing Layer 1 / Layer 2 outputs
# -------------------------------------------------------------------------

# Map regime_engine.RegimeLabel / label strings to allocator RegimeState
_ENGINE_LABEL_TO_REGIME: Dict[str, RegimeState] = {
    "TREND_UP": "TREND_UP",
    "TREND_DOWN": "TREND_DOWN",
    "VOL_COMPRESSION": "VOL_COMPRESSION",
    "VOL_EXPANSION": "VOL_EXPANSION",
    "CHOP": "RANGE",           # engine uses CHOP; allocator contract uses RANGE
    "PANIC": "VOL_EXPANSION",  # treat panic as vol expansion for allocation
}
# Any other label (e.g. unknown) -> UNKNOWN
_DEFAULT_REGIME: RegimeState = "UNKNOWN"

# Map runtime StrategyIntent.action (e.g. ENTER_LONG, EXIT_LONG, FLAT, HOLD) to IntentAction
_ACTION_TO_INTENT: Dict[str, IntentAction] = {
    "ENTER_LONG": "ENTER",
    "ENTER": "ENTER",
    "EXIT_LONG": "EXIT",
    "EXIT": "EXIT",
    "FLAT": "EXIT",
    "HOLD": "HOLD",
}


def _normalize_regime(label: str) -> RegimeState:
    """Map Layer 1 regime label to allocator RegimeState literal."""
    return _ENGINE_LABEL_TO_REGIME.get(str(label).strip().upper(), _DEFAULT_REGIME)


def _normalize_intent_action(action: str) -> IntentAction:
    """Map runtime strategy action to allocator IntentAction."""
    return _ACTION_TO_INTENT.get(str(action).strip().upper(), "HOLD")


def build_asset_bundle_from_layer_outputs(
    product_id: str,
    bar_ts_utc: str,
    regime_output: Any,
    intent_output: Any,
    features: Optional[Dict[str, float]] = None,
) -> AssetSignalBundle:
    """
    Build an AssetSignalBundle from existing Layer 1 (regime) and Layer 2 (intent) outputs.

    Use this when wiring the allocator to research.regime and runtime strategy intent.
    No imports from runtime/argus or regime_engine required: regime_output and
    intent_output can be dict-like or objects with the usual attributes.

    regime_output: must have .label (or ["label"]), .confidence (or 1.0), .meta (or {}).
    intent_output: must have .action (or ["action"]), .confidence, .desired_exposure_frac,
                   .horizon_hours, .reason, .meta (optional; default {}).

    Action mapping: ENTER_LONG/ENTER -> ENTER; EXIT_LONG/EXIT/FLAT -> EXIT; else HOLD.
    Regime mapping: CHOP -> RANGE; PANIC -> VOL_EXPANSION; others by name; unknown -> UNKNOWN.
    """
    def _get(obj: Any, key: str, default: Any = None) -> Any:
        if hasattr(obj, key):
            return getattr(obj, key)
        if isinstance(obj, (dict, Mapping)) and key in obj:
            return obj[key]
        return default

    label = _get(regime_output, "label", "UNKNOWN")
    regime_conf = _get(regime_output, "confidence", 1.0)
    regime_meta = _get(regime_output, "meta") or {}
    if not isinstance(regime_meta, dict):
        regime_meta = {}

    action_raw = _get(intent_output, "action", "HOLD")
    intent_conf = _get(intent_output, "confidence", 0.0)
    if intent_conf is None:
        intent_conf = 0.0
    desired = _get(intent_output, "desired_exposure_frac", 0.0)
    if desired is None:
        desired = 0.0
    horizon = _get(intent_output, "horizon_hours", 0)
    if horizon is None:
        horizon = 0
    reason = _get(intent_output, "reason", "n/a") or "n/a"
    intent_meta = _get(intent_output, "meta") or {}
    if not isinstance(intent_meta, dict):
        intent_meta = {}

    regime_snapshot = RegimeSnapshot(
        regime=_normalize_regime(str(label)),
        confidence=float(regime_conf),
        meta=dict(regime_meta),
    )
    intent = StrategyIntent(
        action=_normalize_intent_action(str(action_raw)),
        confidence=float(intent_conf),
        desired_exposure_frac=float(desired),
        horizon_hours=int(horizon),
        reason=str(reason),
        meta=dict(intent_meta),
    )
    return AssetSignalBundle(
        product_id=product_id,
        bar_ts_utc=bar_ts_utc,
        regime=regime_snapshot,
        intent=intent,
        features=dict(features) if features else None,
    )


# -------------------------------------------------------------------------
# Validation (deterministic; raises on contract violation)
# -------------------------------------------------------------------------

def validate_allocation(
    decision: PortfolioAllocationDecision,
    policy: PortfolioPolicy,
    tol: float = 1e-6,
) -> None:
    """
    Validate allocator output against policy and contract invariants.
    Raises ValueError with a clear message on violation. Use in research; runtime callers may catch.

    Checks:
      - All weights finite and >= 0.
      - If policy.allow_cash: abs(sum(target_weights) + cash_weight - 1.0) <= tol.
      - If not policy.allow_cash: abs(sum(target_weights) - 1.0) <= tol and cash_weight == 0.0.
      - Per-asset min/max caps respected unless decision.meta has cap_override=True.
    """
    tw = decision.target_weights
    cash = decision.cash_weight
    for pid, w in tw.items():
        if not math.isfinite(w):
            raise ValueError(f"validate_allocation: non-finite weight for {pid!r}: {w}")
        if w < -tol:
            raise ValueError(f"validate_allocation: negative weight for {pid!r}: {w}")
    if not math.isfinite(cash):
        raise ValueError(f"validate_allocation: non-finite cash_weight: {cash}")
    if cash < -tol:
        raise ValueError(f"validate_allocation: negative cash_weight: {cash}")

    total_w = sum(tw.values())
    if policy.allow_cash:
        if abs(total_w + cash - 1.0) > tol:
            raise ValueError(
                f"validate_allocation: allow_cash=True but sum(weights)+cash={total_w + cash:.10f} != 1.0 (tol={tol})"
            )
    else:
        if abs(total_w - 1.0) > tol:
            raise ValueError(
                f"validate_allocation: allow_cash=False but sum(weights)={total_w:.10f} != 1.0 (tol={tol})"
            )
        if abs(cash) > tol:
            raise ValueError(
                f"validate_allocation: allow_cash=False but cash_weight={cash} != 0"
            )

    cap_override = decision.meta.get("cap_override") is True
    if not cap_override:
        for pid, w in tw.items():
            if w < policy.min_weight_per_asset - tol:
                raise ValueError(
                    f"validate_allocation: weight for {pid!r}={w} below min_weight_per_asset={policy.min_weight_per_asset}"
                )
            if w > policy.max_weight_per_asset + tol:
                raise ValueError(
                    f"validate_allocation: weight for {pid!r}={w} above max_weight_per_asset={policy.max_weight_per_asset}"
                )


# =====================================================================
# LOCKED FUNCTION SIGNATURE
# =====================================================================

def allocate_portfolio(
    asset_bundles: Mapping[str, AssetSignalBundle],
    policy: PortfolioPolicy,
    ctx: AllocatorContext,
) -> PortfolioAllocationDecision:
    """
    Cross-asset portfolio allocator.

    WEIGHT SEMANTICS (LOCKED):
      - This allocator outputs target_weights such that sum(target_weights) + cash_weight == 1.0.
      - max_gross_exposure is NOT applied here. Layer 3 applies it when converting weights to
        per-asset target exposure: target_exposure_i = weight_i * policy.max_gross_exposure.

    POLICY (Option A — caps redistribute, not cash):
      - Apply per-asset min/max caps to raw desired weights; then renormalize across non-zero
        assets so that sum(target_weights) == 1.0. Caps do not create cash.
      - Cash is created only when policy.allow_cash=True AND total desired exposure is zero (all EXIT).
      - If allow_cash=False and constraints are infeasible (e.g. cannot sum to 1.0 within caps),
        caps are overridden minimally to satisfy sum==1.0 and decision.meta records cap_override.

    Inputs:
      asset_bundles: Dict keyed by product_id; each value has closed-bar timestamp, regime snapshot,
        strategy intent, optional scalar features only (no df/series/stateful).
      policy: Portfolio governance (max_gross_exposure for Layer 3; caps applied here).
      ctx: Closed-bar context (timestamp, optional prev targets for turnover-awareness).

    Contract invariants (MUST hold):
      1) Deterministic: same inputs -> same output
      2) Closed-bar, stateless, no side effects, no lookahead
      3) sum(target_weights) + cash_weight == 1.0
      4) When allow_cash=False: cash_weight == 0; never create cash. If infeasible, set cap_override in meta.
      5) Weights finite, >= 0; caps respected unless meta["cap_override"] is True
    """
    bar_ts_utc = ctx.bar_ts_utc
    prev = ctx.prev_target_weights or {}
    n = len(asset_bundles)
    if n == 0:
        if policy.allow_cash:
            decision = PortfolioAllocationDecision(
                bar_ts_utc=bar_ts_utc,
                target_weights={},
                cash_weight=1.0,
                reason="no_assets",
                meta={"allow_cash": True},
            )
            validate_allocation(decision, policy)
            return decision
        raise ValueError(
            "allocate_portfolio: infeasible policy: no_assets with allow_cash=False"
        )

    # Raw desired weights: intent-driven, regime-aware
    raw_weights: Dict[str, float] = {}
    for pid, bundle in asset_bundles.items():
        intent = bundle.intent
        regime = bundle.regime
        if intent.action == "EXIT":
            w = 0.0
        elif intent.action == "ENTER":
            w = intent.desired_exposure_frac * intent.confidence * regime.confidence
        else:  # HOLD
            prev_w = prev.get(pid, 0.0)
            w = prev_w if prev_w > 0 else intent.desired_exposure_frac * intent.confidence * regime.confidence
        w = max(0.0, min(1.0, float(w)))
        raw_weights[pid] = w

    # Option A: Apply caps first (caps redistribute; we will renormalize, not create cash from caps)
    capped: Dict[str, float] = {}
    for pid, w in raw_weights.items():
        w = max(policy.min_weight_per_asset, min(policy.max_weight_per_asset, w))
        capped[pid] = w

    total_capped = sum(capped.values())

    # Only create cash when allow_cash=True AND total desired exposure is zero (all EXIT)
    if total_capped <= 0:
        if policy.allow_cash:
            target_weights = {pid: 0.0 for pid in capped}
            decision = PortfolioAllocationDecision(
                bar_ts_utc=bar_ts_utc,
                target_weights=target_weights,
                cash_weight=1.0,
                reason="all_exit_or_zero",
                meta={
                    "allow_cash": True,
                    "max_gross_exposure": policy.max_gross_exposure,
                    "n_assets": n,
                },
            )
            validate_allocation(decision, policy)
            return decision
        # allow_cash=False and all EXIT: must assign weights that sum to 1 (infeasible without override)
        equal = 1.0 / n
        target_weights = {pid: equal for pid in capped}
        decision = PortfolioAllocationDecision(
            bar_ts_utc=bar_ts_utc,
            target_weights=target_weights,
            cash_weight=0.0,
            reason="all_exit_no_cash_override",
            meta={
                "allow_cash": False,
                "max_gross_exposure": policy.max_gross_exposure,
                "n_assets": n,
                "cap_override": True,
                "cap_override_reason": "infeasible_constraints",
            },
        )
        validate_allocation(decision, policy)
        return decision

    # Renormalize so sum(target_weights) == 1.0 (caps redistribute; no cash from caps)
    scale = 1.0 / total_capped
    target_weights = {pid: w * scale for pid, w in capped.items()}
    # Check if renormalization caused any weight to violate min or max cap (infeasible under strict caps)
    _tol = 1e-6  # same as validate_allocation default
    cap_override = False
    for w in target_weights.values():
        if w > policy.max_weight_per_asset + _tol or w < policy.min_weight_per_asset - _tol:
            cap_override = True
            break
    cash_weight = 0.0  # Option A: cash only when total_capped==0

    decision = PortfolioAllocationDecision(
        bar_ts_utc=bar_ts_utc,
        target_weights=target_weights,
        cash_weight=cash_weight,
        reason="intent_regime_weighted",
        meta={
            "allow_cash": policy.allow_cash,
            "max_gross_exposure": policy.max_gross_exposure,
            "n_assets": n,
            **({"cap_override": True, "cap_override_reason": "infeasible_constraints"} if cap_override else {}),
        },
    )
    validate_allocation(decision, policy)
    return decision
