"""
Governance Module - Model Lifecycle Management

This module provides tools for automated model governance, including:
- Blue-green deployment with automated promotion decisions
- Drift detection and response
- Automated parameter adjustment based on performance
- Retrain automation and triggering
- Model lineage and provenance tracking

Example:
    ```python
    # Blue-green deployment
    from . import (
        BlueGreenConfig,
        ModelMetrics,
        compare_models
    )

    result = compare_models(
        current_version="v0.7.7",
        candidate_version="v0.7.9",
        current_metrics=ModelMetrics(precision=0.76, recall=0.71, f1=0.74, ece=0.06),
        candidate_metrics=ModelMetrics(precision=0.78, recall=0.73, f1=0.76, ece=0.05),
        confidence=0.85
    )
    print(f"Decision: {result.classification}")

    # Drift detection
    from . import (
        DriftConfig,
        CalibrationPoint,
        detect_drift_candidates
    )

    points = [
        CalibrationPoint("2025-10-30T10:00:00Z", ece=0.08, n=50),
        CalibrationPoint("2025-10-30T11:00:00Z", ece=0.07, n=45),
        CalibrationPoint("2025-10-30T12:00:00Z", ece=0.09, n=60),
    ]
    candidates = detect_drift_candidates(
        calibration_points=points,
        origin="news_sentiment",
        model_version="v0.7.7"
    )

    # Auto-adjustment
    from . import (
        AdjustmentRules,
        PerformanceMetrics,
        adjust_governance_params
    )

    result = adjust_governance_params(
        current_params={"SPY": {"conf_min": 0.60, "debounce_min": 15}},
        performance_metrics={"SPY": PerformanceMetrics(win_rate=0.48, sharpe=0.8, max_drawdown=0.18)}
    )

    # Model lineage
    from . import (
        VersionNode,
        build_lineage_graph
    )

    versions = {
        "v0.7.0": VersionNode("v0.7.0", parent=None, precision=0.75),
        "v0.7.1": VersionNode("v0.7.1", parent="v0.7.0", precision=0.78)
    }
    lineage = build_lineage_graph(versions)
    ```
"""

# Blue-green deployment
from .bluegreen_promotion import (
    BlueGreenConfig,
    ModelMetrics,
    ComparisonResult,
    compute_delta,
    classify_promotion,
    compare_models,
    find_promotion_candidate,
    format_delta,
)

# Drift detection and response
from .drift_response import (
    DriftConfig,
    CalibrationPoint,
    DriftCandidate,
    DriftReport,
    parse_timestamp,
    filter_recent_points,
    detect_drift_candidates,
    analyze_drift_from_series,
)

# Automated parameter adjustment
from .auto_adjust import (
    AdjustmentRules,
    PerformanceMetrics,
    GovernanceParams,
    ParameterAdjustment,
    AdjustmentResult,
    adjust_threshold,
    evaluate_performance,
    compute_adjustment,
    adjust_governance_params,
)

# Retrain automation
from .retrain_automation import (
    RetrainConfig,
    RetrainCandidate,
    RetrainPlan,
    check_calibration_still_high,
    should_retrain_candidate,
    plan_retraining,
    plan_retraining_from_json,
)

# Model lineage tracking
from .model_lineage import (
    VersionNode,
    LineageEdge,
    ModelLineage,
    compute_metric_delta,
    build_lineage_edges,
    build_lineage_graph,
    discover_versions_from_directories,
    format_lineage_text,
    VERSION_DIR_RE,
)

__all__ = [
    # Blue-green deployment
    "BlueGreenConfig",
    "ModelMetrics",
    "ComparisonResult",
    "compute_delta",
    "classify_promotion",
    "compare_models",
    "find_promotion_candidate",
    "format_delta",
    # Drift detection
    "DriftConfig",
    "CalibrationPoint",
    "DriftCandidate",
    "DriftReport",
    "parse_timestamp",
    "filter_recent_points",
    "detect_drift_candidates",
    "analyze_drift_from_series",
    # Auto-adjustment
    "AdjustmentRules",
    "PerformanceMetrics",
    "GovernanceParams",
    "ParameterAdjustment",
    "AdjustmentResult",
    "adjust_threshold",
    "evaluate_performance",
    "compute_adjustment",
    "adjust_governance_params",
    # Retrain automation
    "RetrainConfig",
    "RetrainCandidate",
    "RetrainPlan",
    "check_calibration_still_high",
    "should_retrain_candidate",
    "plan_retraining",
    "plan_retraining_from_json",
    # Model lineage
    "VersionNode",
    "LineageEdge",
    "ModelLineage",
    "compute_metric_delta",
    "build_lineage_edges",
    "build_lineage_graph",
    "discover_versions_from_directories",
    "format_lineage_text",
    "VERSION_DIR_RE",
]