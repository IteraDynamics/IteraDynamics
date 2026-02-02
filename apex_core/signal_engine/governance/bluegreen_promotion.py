"""
Blue-Green Model Deployment and Promotion

Implements blue-green deployment pattern for ML models with gradual rollout
and automated promotion/rollback decisions based on performance metrics.

Key concepts:
- Blue: Current production model
- Green: Candidate model for promotion
- Classification: promote_ready, rollback_risk, or observe
- Metrics comparison: F1 score and Expected Calibration Error (ECE)

Example:
    ```python
    from . import (
        BlueGreenConfig,
        ModelMetrics,
        compare_models,
        classify_promotion
    )

    # Configure promotion criteria
    config = BlueGreenConfig(
        lookback_hours=72,
        delta_threshold=0.02,  # Minimum improvement
        confidence_threshold=0.8  # Minimum confidence for promotion
    )

    # Define metrics for current and candidate models
    current_metrics = ModelMetrics(
        precision=0.76,
        recall=0.71,
        f1=0.74,
        ece=0.06
    )

    candidate_metrics = ModelMetrics(
        precision=0.78,
        recall=0.73,
        f1=0.76,
        ece=0.05
    )

    # Compare models and get recommendation
    result = compare_models(
        current_version="v0.7.7",
        candidate_version="v0.7.9",
        current_metrics=current_metrics,
        candidate_metrics=candidate_metrics,
        confidence=0.85,
        config=config,
        reasons=["calibration_improved", "precision_increase"]
    )

    print(f"Classification: {result.classification}")
    print(f"F1 Delta: {result.delta_f1:+.2f}")
    print(f"ECE Delta: {result.delta_ece:+.2f}")
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
class BlueGreenConfig:
    """
    Configuration for blue-green deployment promotion logic.

    Attributes:
        lookback_hours: Time window for metric comparison (hours)
        delta_threshold: Minimum improvement required for promotion
        confidence_threshold: Minimum confidence level for promotion
    """
    lookback_hours: int = 72
    delta_threshold: float = 0.02
    confidence_threshold: float = 0.8


# -----------------------
# Data Models
# -----------------------

@dataclass
class ModelMetrics:
    """
    Performance metrics for a model version.

    Attributes:
        precision: Precision score (TP / (TP + FP))
        recall: Recall score (TP / (TP + FN))
        f1: F1 score (harmonic mean of precision and recall)
        ece: Expected Calibration Error (calibration quality)
    """
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1: Optional[float] = None
    ece: Optional[float] = None

    def to_dict(self) -> Dict[str, Optional[float]]:
        """Convert to dictionary representation."""
        return {
            "precision": self.precision,
            "recall": self.recall,
            "F1": self.f1,
            "ECE": self.ece,
        }


@dataclass
class ComparisonResult:
    """
    Result of comparing blue (current) and green (candidate) models.

    Attributes:
        current_version: Version identifier of current model
        candidate_version: Version identifier of candidate model
        current_metrics: Metrics for current model
        candidate_metrics: Metrics for candidate model
        delta_precision: Change in precision (candidate - current)
        delta_recall: Change in recall (candidate - current)
        delta_f1: Change in F1 score (candidate - current)
        delta_ece: Change in ECE (candidate - current, negative is better)
        classification: Promotion decision (promote_ready, rollback_risk, observe)
        confidence: Confidence level for promotion (0.0-1.0)
        reasons: List of reasons supporting the decision
        window_hours: Time window used for comparison
        generated_at: Timestamp of comparison
    """
    current_version: str
    candidate_version: str
    current_metrics: ModelMetrics
    candidate_metrics: ModelMetrics
    delta_precision: Optional[float] = None
    delta_recall: Optional[float] = None
    delta_f1: Optional[float] = None
    delta_ece: Optional[float] = None
    classification: str = "observe"
    confidence: float = 0.0
    reasons: List[str] = field(default_factory=list)
    window_hours: int = 72
    generated_at: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "generated_at": self.generated_at.isoformat() if self.generated_at else None,
            "current_model": self.current_version,
            "candidate": self.candidate_version,
            "current_metrics": self.current_metrics.to_dict(),
            "candidate_metrics": self.candidate_metrics.to_dict(),
            "delta": {
                "precision": self.delta_precision,
                "recall": self.delta_recall,
                "F1": self.delta_f1,
                "ECE": self.delta_ece,
            },
            "classification": self.classification,
            "confidence": round(self.confidence, 2),
            "notes": self.reasons,
            "window_hours": self.window_hours,
        }


# -----------------------
# Core Logic
# -----------------------

def compute_delta(
    current_value: Optional[float],
    candidate_value: Optional[float]
) -> Optional[float]:
    """
    Compute delta between candidate and current metric values.

    Args:
        current_value: Current model metric value
        candidate_value: Candidate model metric value

    Returns:
        Delta (candidate - current) or None if either value is missing

    Example:
        >>> compute_delta(0.74, 0.76)
        0.02
        >>> compute_delta(None, 0.76)
        None
    """
    if current_value is None or candidate_value is None:
        return None
    try:
        return float(candidate_value) - float(current_value)
    except (ValueError, TypeError):
        return None


def classify_promotion(
    delta_f1: Optional[float],
    delta_ece: Optional[float],
    confidence: float,
    config: BlueGreenConfig
) -> str:
    """
    Classify whether to promote, rollback, or observe based on metrics.

    Classification logic:
    1. rollback_risk: Significant performance degradation detected
       - F1 drops by more than 2% OR
       - ECE increases by more than 1%

    2. promote_ready: Candidate shows clear improvement
       - (F1 improves by >= delta_threshold OR ECE improves by >= delta_threshold) AND
       - Confidence >= confidence_threshold

    3. observe: Default - insufficient evidence for action
       - Metrics missing OR
       - Improvements below threshold OR
       - Confidence too low

    Args:
        delta_f1: Change in F1 score (candidate - current)
        delta_ece: Change in ECE (candidate - current, negative is better)
        confidence: Confidence level for promotion (0.0-1.0)
        config: Blue-green configuration with thresholds

    Returns:
        Classification string: "promote_ready", "rollback_risk", or "observe"

    Example:
        >>> config = BlueGreenConfig(delta_threshold=0.02, confidence_threshold=0.8)
        >>> classify_promotion(delta_f1=0.03, delta_ece=-0.01, confidence=0.85, config=config)
        'promote_ready'
        >>> classify_promotion(delta_f1=-0.05, delta_ece=None, confidence=0.85, config=config)
        'rollback_risk'
    """
    # Safe defaults when metrics missing
    if delta_f1 is None and delta_ece is None:
        return "observe"

    # Check for rollback risk first (safety)
    if (delta_f1 is not None and delta_f1 < -0.02) or \
       (delta_ece is not None and delta_ece > 0.01):
        return "rollback_risk"

    # Check for promotion readiness
    # Note: For ECE, negative delta is improvement (lower is better)
    f1_improved = delta_f1 is not None and delta_f1 >= config.delta_threshold
    ece_improved = delta_ece is not None and -delta_ece >= config.delta_threshold

    improvement_ok = f1_improved or ece_improved
    confidence_ok = confidence >= config.confidence_threshold

    if improvement_ok and confidence_ok:
        return "promote_ready"

    return "observe"


def compare_models(
    current_version: str,
    candidate_version: str,
    current_metrics: ModelMetrics,
    candidate_metrics: ModelMetrics,
    confidence: float,
    config: Optional[BlueGreenConfig] = None,
    reasons: Optional[List[str]] = None,
) -> ComparisonResult:
    """
    Compare blue (current) and green (candidate) models and recommend action.

    This is the main entry point for blue-green deployment decisions.

    Args:
        current_version: Version identifier of current model (e.g., "v0.7.7")
        candidate_version: Version identifier of candidate model (e.g., "v0.7.9")
        current_metrics: Performance metrics for current model
        candidate_metrics: Performance metrics for candidate model
        confidence: Confidence level for promotion (0.0-1.0)
        config: Configuration for promotion thresholds (uses defaults if None)
        reasons: List of reasons supporting the promotion decision

    Returns:
        ComparisonResult with deltas, classification, and metadata

    Example:
        >>> current = ModelMetrics(precision=0.76, recall=0.71, f1=0.74, ece=0.06)
        >>> candidate = ModelMetrics(precision=0.78, recall=0.73, f1=0.76, ece=0.05)
        >>> result = compare_models(
        ...     current_version="v0.7.7",
        ...     candidate_version="v0.7.9",
        ...     current_metrics=current,
        ...     candidate_metrics=candidate,
        ...     confidence=0.85,
        ...     reasons=["calibration_improved"]
        ... )
        >>> result.classification
        'promote_ready'
        >>> result.delta_f1
        0.02
    """
    if config is None:
        config = BlueGreenConfig()

    if reasons is None:
        reasons = []

    # Compute deltas
    delta_p = compute_delta(current_metrics.precision, candidate_metrics.precision)
    delta_r = compute_delta(current_metrics.recall, candidate_metrics.recall)
    delta_f1 = compute_delta(current_metrics.f1, candidate_metrics.f1)
    delta_ece = compute_delta(current_metrics.ece, candidate_metrics.ece)

    # Classify promotion decision
    classification = classify_promotion(delta_f1, delta_ece, confidence, config)

    # Build result
    return ComparisonResult(
        current_version=current_version,
        candidate_version=candidate_version,
        current_metrics=current_metrics,
        candidate_metrics=candidate_metrics,
        delta_precision=delta_p,
        delta_recall=delta_r,
        delta_f1=delta_f1,
        delta_ece=delta_ece,
        classification=classification,
        confidence=confidence,
        reasons=list(reasons),
        window_hours=config.lookback_hours,
        generated_at=datetime.now(timezone.utc),
    )


def find_promotion_candidate(
    governance_actions: List[Dict[str, Any]]
) -> Optional[tuple[str, float, List[str]]]:
    """
    Find the best promotion candidate from governance actions.

    Scans a list of governance actions and selects the 'promote' action
    with the highest confidence level.

    Args:
        governance_actions: List of governance action dictionaries
            Each action should have:
            - action: str (e.g., "promote", "rollback", "observe")
            - version: str (model version identifier)
            - confidence: float (0.0-1.0)
            - reasons: List[str] or str

    Returns:
        Tuple of (version, confidence, reasons) for best candidate,
        or None if no promote actions found

    Example:
        >>> actions = [
        ...     {"action": "promote", "version": "v0.7.9", "confidence": 0.85, "reasons": ["improved_f1"]},
        ...     {"action": "observe", "version": "v0.7.8", "confidence": 0.60},
        ...     {"action": "promote", "version": "v0.7.10", "confidence": 0.90, "reasons": ["best_ece"]},
        ... ]
        >>> find_promotion_candidate(actions)
        ('v0.7.10', 0.90, ['best_ece'])
    """
    best = None

    for action in governance_actions:
        if str(action.get("action")) != "promote":
            continue

        version = action.get("version")
        conf = float(action.get("confidence", 0.0) or 0.0)
        reasons = action.get("reasons") or action.get("reason") or []

        # Normalize reasons to list
        if not isinstance(reasons, list):
            reasons = [str(reasons)]

        if version:
            if not best or conf > best[1]:
                best = (version, conf, reasons)

    return best


# -----------------------
# Formatting Utilities
# -----------------------

def format_delta(delta: Optional[float]) -> str:
    """
    Format delta value for display.

    Args:
        delta: Delta value to format

    Returns:
        Formatted string with sign (e.g., "+0.02", "-0.01", "n/a")

    Example:
        >>> format_delta(0.023)
        '+0.02'
        >>> format_delta(-0.015)
        '-0.02'
        >>> format_delta(None)
        'n/a'
    """
    if delta is None:
        return "n/a"

    try:
        import math
        if math.isnan(delta):
            return "n/a"
    except (ImportError, TypeError):
        pass

    return f"{delta:+.2f}"


__all__ = [
    "BlueGreenConfig",
    "ModelMetrics",
    "ComparisonResult",
    "compute_delta",
    "classify_promotion",
    "compare_models",
    "find_promotion_candidate",
    "format_delta",
]
