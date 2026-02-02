"""
Signal Engine Validation Module

Provides comprehensive validation tools for model performance including:
- Calibration analysis (ECE, Brier score, reliability diagrams)
- Performance validation (statistical tests, confidence intervals)
- Reliability assessment (feedback scoring and filtering)

Example:
    ```python
    # Calibration analysis
    from . import compute_calibration_metrics

    y_true = [1, 0, 1, 1, 0]
    y_prob = [0.9, 0.2, 0.8, 0.7, 0.3]
    metrics = compute_calibration_metrics(y_true, y_prob)
    print(f"ECE: {metrics.ece:.4f}, Brier: {metrics.brier:.4f}")

    # Performance validation
    from . import (
        compute_sharpe_ratio,
        compare_models_performance
    )

    returns_a = [0.02, -0.01, 0.03]
    sharpe = compute_sharpe_ratio(returns_a)

    returns_b = [0.03, 0.01, 0.04]
    comparison = compare_models_performance(returns_a, returns_b)
    print(f"Significant improvement: {comparison.is_significant}")

    # Reliability scoring
    from . import compute_temporal_reliability

    feedback = [...]  # List of feedback entries
    scored = compute_temporal_reliability(feedback)
    ```
"""

# Reliability assessment
from .reliability import (
    compute_basic_reliability,
    compute_temporal_reliability,
    aggregate_feedback_reliability,
    filter_reliable_feedback,
)

# Calibration analysis
from .calibration import (
    CalibrationBin,
    CalibrationMetrics,
    compute_brier_score,
    create_bin_edges,
    assign_to_bin,
    compute_ece_and_bins,
    compute_calibration_metrics,
    is_well_calibrated,
    compute_calibration_by_group,
)

# Performance validation
from .performance import (
    ConfidenceInterval,
    StatisticalTest,
    PerformanceComparison,
    compute_confidence_interval,
    is_statistically_significant,
    compute_sharpe_ratio,
    compute_sortino_ratio,
    compare_models_performance,
)

__all__ = [
    # Reliability
    "compute_basic_reliability",
    "compute_temporal_reliability",
    "aggregate_feedback_reliability",
    "filter_reliable_feedback",
    # Calibration
    "CalibrationBin",
    "CalibrationMetrics",
    "compute_brier_score",
    "create_bin_edges",
    "assign_to_bin",
    "compute_ece_and_bins",
    "compute_calibration_metrics",
    "is_well_calibrated",
    "compute_calibration_by_group",
    # Performance
    "ConfidenceInterval",
    "StatisticalTest",
    "PerformanceComparison",
    "compute_confidence_interval",
    "is_statistically_significant",
    "compute_sharpe_ratio",
    "compute_sortino_ratio",
    "compare_models_performance",
]
