"""
Automated Governance Parameter Adjustment

Automatically adjusts model governance parameters (confidence thresholds, debounce times)
based on observed performance metrics from backtesting or paper trading.

Key concepts:
- Closed-loop feedback: Performance → Adjustment → Tuning → Performance
- Conservative adjustment: Small incremental changes with safety bounds
- Multi-metric evaluation: Win rate, Sharpe ratio, drawdown
- Reason-based decisions: Transparent adjustment logic

Example:
    ```python
    from . import (
        GovernanceConfig,
        AdjustmentRules,
        PerformanceMetrics,
        adjust_governance_params
    )

    # Configure adjustment rules
    rules = AdjustmentRules(
        min_win_rate=0.55,
        target_win_rate=0.65,
        min_sharpe=1.0,
        target_sharpe=1.5,
        max_drawdown=0.15,
        delta_increase_large=0.05,
        delta_increase_medium=0.03,
        delta_decrease=-0.02
    )

    # Current governance parameters
    current_params = {
        "SPY": {"conf_min": 0.60, "debounce_min": 15},
        "QQQ": {"conf_min": 0.65, "debounce_min": 20}
    }

    # Observed performance
    performance = {
        "SPY": PerformanceMetrics(win_rate=0.48, sharpe=0.8, max_drawdown=0.18),
        "QQQ": PerformanceMetrics(win_rate=0.70, sharpe=1.8, max_drawdown=0.08)
    }

    # Adjust parameters
    result = adjust_governance_params(
        current_params=current_params,
        performance_metrics=performance,
        rules=rules
    )

    # Check adjustments
    for adjustment in result.adjustments:
        print(f"{adjustment.symbol}: {adjustment.old_conf_min} → {adjustment.new_conf_min}")
        print(f"  Reasons: {', '.join(adjustment.reasons)}")
    ```
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


# -----------------------
# Configuration
# -----------------------

@dataclass
class AdjustmentRules:
    """
    Rules for automated governance parameter adjustment.

    These thresholds determine when and how to adjust confidence thresholds
    based on performance metrics.

    Attributes:
        min_win_rate: Minimum acceptable win rate (triggers tightening)
        target_win_rate: Excellent win rate (enables relaxing)
        min_sharpe: Minimum acceptable Sharpe ratio (triggers tightening)
        target_sharpe: Excellent Sharpe ratio (enables relaxing)
        max_drawdown: Maximum acceptable drawdown (triggers tightening)
        target_drawdown: Excellent drawdown control (enables relaxing)
        delta_increase_large: Large increase for poor performance
        delta_increase_medium: Medium increase for mediocre performance
        delta_decrease: Decrease for excellent performance
        min_conf_threshold: Lower bound for confidence threshold
        max_conf_threshold: Upper bound for confidence threshold
    """
    min_win_rate: float = 0.55
    target_win_rate: float = 0.65
    min_sharpe: float = 1.0
    target_sharpe: float = 1.5
    max_drawdown: float = 0.15
    target_drawdown: float = 0.10
    delta_increase_large: float = 0.05
    delta_increase_medium: float = 0.03
    delta_decrease: float = -0.02
    min_conf_threshold: float = 0.50
    max_conf_threshold: float = 0.90


# -----------------------
# Data Models
# -----------------------

@dataclass
class PerformanceMetrics:
    """
    Performance metrics for a trading strategy or model.

    Attributes:
        win_rate: Fraction of profitable trades (0.0-1.0)
        sharpe: Sharpe ratio (risk-adjusted return)
        max_drawdown: Maximum drawdown from peak (0.0-1.0)
        total_return: Total return (optional)
        num_trades: Number of trades (optional)
        profit_factor: Profit factor (optional)
    """
    win_rate: float
    sharpe: float
    max_drawdown: float
    total_return: Optional[float] = None
    num_trades: Optional[int] = None
    profit_factor: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        result = {
            "win_rate": self.win_rate,
            "sharpe": self.sharpe,
            "max_drawdown": self.max_drawdown,
        }
        if self.total_return is not None:
            result["total_return"] = self.total_return
        if self.num_trades is not None:
            result["num_trades"] = self.num_trades
        if self.profit_factor is not None:
            result["profit_factor"] = self.profit_factor
        return result


@dataclass
class GovernanceParams:
    """
    Governance parameters for a symbol or strategy.

    Attributes:
        conf_min: Minimum confidence threshold for signals
        debounce_min: Minimum time between signals (minutes)
    """
    conf_min: float
    debounce_min: int = 15

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "conf_min": self.conf_min,
            "debounce_min": self.debounce_min,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> GovernanceParams:
        """Create from dictionary representation."""
        return cls(
            conf_min=float(data["conf_min"]),
            debounce_min=int(data.get("debounce_min", 15)),
        )


@dataclass
class ParameterAdjustment:
    """
    Record of a parameter adjustment for a symbol.

    Attributes:
        symbol: Asset symbol or identifier
        old_conf_min: Previous confidence threshold
        new_conf_min: New confidence threshold
        delta: Change applied (new - old)
        reasons: List of reasons for adjustment
        performance: Performance metrics that triggered adjustment
    """
    symbol: str
    old_conf_min: float
    new_conf_min: float
    delta: float
    reasons: List[str] = field(default_factory=list)
    performance: Optional[PerformanceMetrics] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        result = {
            "symbol": self.symbol,
            "old_conf_min": self.old_conf_min,
            "new_conf_min": self.new_conf_min,
            "delta": self.delta,
            "reasons": self.reasons,
        }
        if self.performance:
            result["performance"] = self.performance.to_dict()
        return result


@dataclass
class AdjustmentResult:
    """
    Complete result of governance parameter adjustment.

    Attributes:
        timestamp: When adjustment was performed
        adjustments: List of parameter adjustments made
        updated_params: Complete updated parameter set
        dry_run: Whether this was a dry run (no changes applied)
    """
    timestamp: datetime
    adjustments: List[ParameterAdjustment] = field(default_factory=list)
    updated_params: Dict[str, GovernanceParams] = field(default_factory=dict)
    dry_run: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "adjustments": [a.to_dict() for a in self.adjustments],
            "updated_params": {
                symbol: params.to_dict()
                for symbol, params in self.updated_params.items()
            },
            "dry_run": self.dry_run,
        }


# -----------------------
# Core Logic
# -----------------------

def adjust_threshold(
    current: float,
    delta: float,
    min_threshold: float = 0.50,
    max_threshold: float = 0.90
) -> float:
    """
    Adjust confidence threshold with safety bounds.

    Args:
        current: Current threshold value
        delta: Amount to adjust (positive = more selective, negative = less selective)
        min_threshold: Lower bound
        max_threshold: Upper bound

    Returns:
        New threshold value, bounded and rounded to 2 decimal places

    Example:
        >>> adjust_threshold(0.60, 0.05)
        0.65
        >>> adjust_threshold(0.85, 0.10)  # Capped at max
        0.90
        >>> adjust_threshold(0.55, -0.10)  # Capped at min
        0.50
    """
    new = current + delta
    bounded = max(min_threshold, min(max_threshold, new))
    return round(bounded, 2)


def evaluate_performance(
    symbol: str,
    metrics: PerformanceMetrics,
    rules: AdjustmentRules
) -> List[str]:
    """
    Evaluate performance metrics and return reasons for adjustment.

    Args:
        symbol: Asset symbol or identifier
        metrics: Performance metrics to evaluate
        rules: Adjustment rules with thresholds

    Returns:
        List of adjustment reasons (empty if no adjustment needed)

    Example:
        >>> rules = AdjustmentRules()
        >>> metrics = PerformanceMetrics(win_rate=0.45, sharpe=0.8, max_drawdown=0.20)
        >>> reasons = evaluate_performance("SPY", metrics, rules)
        >>> len(reasons)
        3
    """
    reasons = []

    # Check for poor performance (increase selectivity)
    if metrics.win_rate < rules.min_win_rate:
        reasons.append(
            f"Low win rate ({metrics.win_rate:.2f} < {rules.min_win_rate})"
        )

    if metrics.sharpe < rules.min_sharpe:
        reasons.append(
            f"Low Sharpe ratio ({metrics.sharpe:.2f} < {rules.min_sharpe})"
        )

    if metrics.max_drawdown > rules.max_drawdown:
        reasons.append(
            f"High drawdown ({metrics.max_drawdown:.2f} > {rules.max_drawdown})"
        )

    # Check for excellent performance (relax selectivity slightly)
    if (metrics.win_rate > rules.target_win_rate and
        metrics.sharpe > rules.target_sharpe and
        metrics.max_drawdown < rules.target_drawdown):
        reasons.append(
            f"Excellent performance (win={metrics.win_rate:.2f}, "
            f"sharpe={metrics.sharpe:.2f}, dd={metrics.max_drawdown:.2f})"
        )

    return reasons


def compute_adjustment(
    current_conf: float,
    reasons: List[str],
    rules: AdjustmentRules
) -> float:
    """
    Compute new confidence threshold based on reasons.

    Args:
        current_conf: Current confidence threshold
        reasons: List of reasons from evaluate_performance
        rules: Adjustment rules

    Returns:
        New confidence threshold

    Example:
        >>> rules = AdjustmentRules()
        >>> reasons = ["Low win rate (0.45 < 0.55)", "High drawdown (0.20 > 0.15)"]
        >>> new_conf = compute_adjustment(0.60, reasons, rules)
        >>> new_conf > 0.60
        True
    """
    new_conf = current_conf

    for reason in reasons:
        if "Low win rate" in reason or "Low Sharpe" in reason:
            new_conf = adjust_threshold(
                new_conf,
                rules.delta_increase_medium,
                rules.min_conf_threshold,
                rules.max_conf_threshold
            )

        if "High drawdown" in reason:
            new_conf = adjust_threshold(
                new_conf,
                rules.delta_increase_large,
                rules.min_conf_threshold,
                rules.max_conf_threshold
            )

        if "Excellent performance" in reason:
            new_conf = adjust_threshold(
                new_conf,
                rules.delta_decrease,
                rules.min_conf_threshold,
                rules.max_conf_threshold
            )

    return new_conf


def adjust_governance_params(
    current_params: Dict[str, Dict[str, Any]],
    performance_metrics: Dict[str, PerformanceMetrics],
    rules: Optional[AdjustmentRules] = None,
    default_conf_min: float = 0.60,
    default_debounce_min: int = 15,
    dry_run: bool = False
) -> AdjustmentResult:
    """
    Adjust governance parameters based on performance metrics.

    This is the main entry point for automated governance adjustment.

    Args:
        current_params: Current governance parameters by symbol
        performance_metrics: Observed performance by symbol
        rules: Adjustment rules (uses defaults if None)
        default_conf_min: Default confidence threshold for new symbols
        default_debounce_min: Default debounce time for new symbols
        dry_run: If True, compute but don't apply changes

    Returns:
        AdjustmentResult with all adjustments and updated parameters

    Example:
        >>> current = {"SPY": {"conf_min": 0.60, "debounce_min": 15}}
        >>> perf = {"SPY": PerformanceMetrics(win_rate=0.45, sharpe=0.8, max_drawdown=0.20)}
        >>> result = adjust_governance_params(current, perf)
        >>> result.adjustments[0].new_conf_min > 0.60
        True
    """
    if rules is None:
        rules = AdjustmentRules()

    # Convert current params to GovernanceParams objects
    params = {}
    for symbol, param_dict in current_params.items():
        if isinstance(param_dict, GovernanceParams):
            params[symbol] = param_dict
        else:
            params[symbol] = GovernanceParams.from_dict(param_dict)

    adjustments = []

    for symbol, metrics in performance_metrics.items():
        # Initialize default params if symbol not present
        if symbol not in params:
            params[symbol] = GovernanceParams(
                conf_min=default_conf_min,
                debounce_min=default_debounce_min
            )

        current_conf = params[symbol].conf_min

        # Evaluate performance and get reasons
        reasons = evaluate_performance(symbol, metrics, rules)

        if not reasons:
            # No adjustment needed
            continue

        # Compute new threshold
        new_conf = compute_adjustment(current_conf, reasons, rules)

        # Apply adjustment if changed
        if new_conf != current_conf:
            adjustment = ParameterAdjustment(
                symbol=symbol,
                old_conf_min=current_conf,
                new_conf_min=new_conf,
                delta=round(new_conf - current_conf, 2),
                reasons=reasons,
                performance=metrics
            )
            adjustments.append(adjustment)

            if not dry_run:
                params[symbol].conf_min = new_conf

    return AdjustmentResult(
        timestamp=datetime.now(timezone.utc),
        adjustments=adjustments,
        updated_params=params,
        dry_run=dry_run
    )


__all__ = [
    "AdjustmentRules",
    "PerformanceMetrics",
    "GovernanceParams",
    "ParameterAdjustment",
    "AdjustmentResult",
    "adjust_threshold",
    "evaluate_performance",
    "compute_adjustment",
    "adjust_governance_params",
]
