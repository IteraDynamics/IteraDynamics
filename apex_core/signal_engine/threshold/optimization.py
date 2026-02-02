"""
Threshold Optimization and Analysis

Provides tools for optimizing classification thresholds based on precision/recall trade-offs,
threshold sweeping, and multi-objective optimization.

Key concepts:
- Threshold sweeping: Evaluate metrics across all candidate thresholds
- Precision-constrained optimization: Maximize recall subject to minimum precision
- F1 optimization: Find threshold that maximizes F1 score
- Multi-objective: Balance precision, recall, and other metrics

Example:
    ```python
    from . import (
        sweep_thresholds,
        optimize_classification_threshold,
        ThresholdRecommendation
    )

    # Predictions and labels
    scores = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2]
    labels = [True, True, True, False, True, False, False, False]

    # Find optimal threshold
    result = optimize_classification_threshold(
        scores=scores,
        labels=labels,
        strategy="precision_constrained",
        min_precision=0.75
    )

    print(f"Recommended threshold: {result.threshold:.3f}")
    print(f"Precision: {result.precision:.3f}")
    print(f"Recall: {result.recall:.3f}")
    print(f"F1: {result.f1:.3f}")

    # Sweep all thresholds
    all_results = sweep_thresholds(scores, labels)
    for r in all_results[:5]:  # Top 5
        print(f"Threshold {r.threshold:.3f}: F1={r.f1:.3f}")
    ```
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Literal, Optional, Tuple


# -----------------------
# Data Models
# -----------------------

@dataclass
class ThresholdMetrics:
    """
    Metrics for a single threshold value.

    Attributes:
        threshold: Threshold value
        tp: True positives
        fp: False positives
        fn: False negatives
        tn: True negatives
        precision: Precision score
        recall: Recall score
        f1: F1 score
        n_samples: Total number of samples
    """
    threshold: float
    tp: int
    fp: int
    fn: int
    tn: int = 0
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0

    def __post_init__(self):
        """Auto-compute derived metrics."""
        self.n_samples = self.tp + self.fp + self.fn + self.tn
        self.precision = _safe_div(self.tp, self.tp + self.fp)
        self.recall = _safe_div(self.tp, self.tp + self.fn)
        self.f1 = _safe_div(2 * self.precision * self.recall,
                           self.precision + self.recall)

    def to_dict(self) -> dict:
        """Convert to dictionary representation."""
        return {
            "threshold": self.threshold,
            "tp": self.tp,
            "fp": self.fp,
            "fn": self.fn,
            "tn": self.tn,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "n_samples": self.n_samples,
        }

    @property
    def specificity(self) -> float:
        """Specificity (true negative rate)."""
        return _safe_div(self.tn, self.tn + self.fp)

    @property
    def fpr(self) -> float:
        """False positive rate."""
        return _safe_div(self.fp, self.fp + self.tn)


@dataclass
class ThresholdRecommendation:
    """
    Recommended threshold with justification.

    Attributes:
        threshold: Recommended threshold value
        metrics: Metrics at this threshold
        strategy: Strategy used for optimization
        status: Status of recommendation ("ok", "guardrail_violated", etc.)
        reasons: List of reasons for this recommendation
    """
    threshold: float
    metrics: ThresholdMetrics
    strategy: str
    status: str = "ok"
    reasons: List[str] = None

    def __post_init__(self):
        if self.reasons is None:
            self.reasons = []

    def to_dict(self) -> dict:
        """Convert to dictionary representation."""
        return {
            "threshold": self.threshold,
            "metrics": self.metrics.to_dict(),
            "strategy": self.strategy,
            "status": self.status,
            "reasons": self.reasons,
        }

    @property
    def precision(self) -> float:
        """Precision at recommended threshold."""
        return self.metrics.precision

    @property
    def recall(self) -> float:
        """Recall at recommended threshold."""
        return self.metrics.recall

    @property
    def f1(self) -> float:
        """F1 score at recommended threshold."""
        return self.metrics.f1


# -----------------------
# Core Functions
# -----------------------

def _safe_div(numerator: float, denominator: float) -> float:
    """Safe division returning 0.0 if denominator is 0."""
    return (numerator / denominator) if denominator != 0 else 0.0


def compute_confusion_matrix(
    scores: List[float],
    labels: List[bool],
    threshold: float
) -> Tuple[int, int, int, int]:
    """
    Compute confusion matrix at a given threshold.

    Args:
        scores: Predicted scores/probabilities
        labels: True labels (True/False or 1/0)
        threshold: Classification threshold

    Returns:
        Tuple of (TP, FP, FN, TN)

    Example:
        >>> scores = [0.9, 0.7, 0.6, 0.4, 0.2]
        >>> labels = [True, True, False, False, False]
        >>> tp, fp, fn, tn = compute_confusion_matrix(scores, labels, 0.5)
        >>> (tp, fp, fn, tn)
        (2, 1, 0, 2)
    """
    tp = fp = fn = tn = 0

    for score, label in zip(scores, labels):
        prediction = score >= threshold
        true_label = bool(label)

        if true_label and prediction:
            tp += 1
        elif not true_label and prediction:
            fp += 1
        elif true_label and not prediction:
            fn += 1
        else:  # not true_label and not prediction
            tn += 1

    return tp, fp, fn, tn


def sweep_thresholds(
    scores: List[float],
    labels: List[bool],
    include_extremes: bool = False
) -> List[ThresholdMetrics]:
    """
    Sweep through all unique score values as candidate thresholds.

    Args:
        scores: Predicted scores/probabilities
        labels: True labels
        include_extremes: Whether to include 0.0 and 1.0 as thresholds

    Returns:
        List of ThresholdMetrics sorted by F1 score (descending)

    Example:
        >>> scores = [0.9, 0.8, 0.7, 0.6, 0.5]
        >>> labels = [True, True, True, False, False]
        >>> results = sweep_thresholds(scores, labels)
        >>> len(results) > 0
        True
        >>> results[0].f1 >= results[-1].f1  # Sorted by F1
        True
    """
    if not scores or not labels:
        return []

    if len(scores) != len(labels):
        raise ValueError(f"Length mismatch: scores={len(scores)}, labels={len(labels)}")

    # Get unique sorted thresholds
    candidate_thresholds = sorted(set(scores))

    if include_extremes:
        if 0.0 not in candidate_thresholds:
            candidate_thresholds = [0.0] + candidate_thresholds
        if 1.0 not in candidate_thresholds:
            candidate_thresholds.append(1.0)

    # Evaluate each threshold
    results = []
    for threshold in candidate_thresholds:
        tp, fp, fn, tn = compute_confusion_matrix(scores, labels, threshold)
        metrics = ThresholdMetrics(
            threshold=threshold,
            tp=tp,
            fp=fp,
            fn=fn,
            tn=tn
        )

        # Only include thresholds with samples
        if metrics.n_samples > 0:
            results.append(metrics)

    # Sort by F1 score (descending), then recall, then precision
    results.sort(
        key=lambda m: (-m.f1, -m.recall, -m.precision, m.threshold)
    )

    return results


def optimize_classification_threshold(
    scores: List[float],
    labels: List[bool],
    strategy: Literal["max_f1", "precision_constrained"] = "max_f1",
    min_precision: float = 0.75,
    min_recall: Optional[float] = None
) -> Optional[ThresholdRecommendation]:
    """
    Find optimal classification threshold using specified strategy.

    Strategies:
    - "max_f1": Maximize F1 score
    - "precision_constrained": Maximize recall subject to minimum precision constraint

    Args:
        scores: Predicted scores/probabilities
        labels: True labels
        strategy: Optimization strategy
        min_precision: Minimum precision constraint (for precision_constrained)
        min_recall: Optional minimum recall constraint

    Returns:
        ThresholdRecommendation or None if no valid threshold found

    Example:
        >>> scores = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4]
        >>> labels = [True, True, True, False, True, False]
        >>> result = optimize_classification_threshold(scores, labels, "max_f1")
        >>> result.threshold >= 0.0
        True
    """
    # Sweep all thresholds
    candidates = sweep_thresholds(scores, labels)

    if not candidates:
        return None

    if strategy == "max_f1":
        # Best F1 is already first after sweep_thresholds sorting
        best = candidates[0]

        # Apply optional recall constraint
        if min_recall is not None:
            feasible = [c for c in candidates if c.recall >= min_recall]
            if feasible:
                best = feasible[0]

        return ThresholdRecommendation(
            threshold=best.threshold,
            metrics=best,
            strategy=strategy,
            reasons=["max_f1"]
        )

    elif strategy == "precision_constrained":
        # Find feasible thresholds (precision >= min_precision)
        feasible = [c for c in candidates if c.precision >= min_precision]

        if feasible:
            # Maximize recall among feasible, tie-break by F1
            feasible.sort(key=lambda m: (-m.recall, -m.f1, m.threshold))
            best = feasible[0]

            return ThresholdRecommendation(
                threshold=best.threshold,
                metrics=best,
                strategy=strategy,
                reasons=["precision_constrained", f"min_precision={min_precision:.2f}"]
            )
        else:
            # No feasible threshold - fallback to max F1
            best = candidates[0]
            return ThresholdRecommendation(
                threshold=best.threshold,
                metrics=best,
                strategy="max_f1_fallback",
                status="precision_constraint_violated",
                reasons=[f"no_threshold_meets_min_precision={min_precision:.2f}",
                         "fallback_to_max_f1"]
            )

    else:
        raise ValueError(f"Unknown strategy: {strategy}")


def compare_thresholds(
    threshold_a: float,
    threshold_b: float,
    scores: List[float],
    labels: List[bool]
) -> dict:
    """
    Compare metrics between two thresholds.

    Args:
        threshold_a: First threshold
        threshold_b: Second threshold
        scores: Predicted scores
        labels: True labels

    Returns:
        Dictionary with comparison results

    Example:
        >>> scores = [0.9, 0.8, 0.7, 0.6, 0.5]
        >>> labels = [True, True, True, False, False]
        >>> comparison = compare_thresholds(0.6, 0.7, scores, labels)
        >>> "delta_f1" in comparison
        True
    """
    tp_a, fp_a, fn_a, tn_a = compute_confusion_matrix(scores, labels, threshold_a)
    tp_b, fp_b, fn_b, tn_b = compute_confusion_matrix(scores, labels, threshold_b)

    metrics_a = ThresholdMetrics(threshold_a, tp_a, fp_a, fn_a, tn_a)
    metrics_b = ThresholdMetrics(threshold_b, tp_b, fp_b, fn_b, tn_b)

    return {
        "threshold_a": threshold_a,
        "threshold_b": threshold_b,
        "metrics_a": metrics_a.to_dict(),
        "metrics_b": metrics_b.to_dict(),
        "delta_precision": metrics_b.precision - metrics_a.precision,
        "delta_recall": metrics_b.recall - metrics_a.recall,
        "delta_f1": metrics_b.f1 - metrics_a.f1,
        "improvement": "better" if metrics_b.f1 > metrics_a.f1 else
                      "worse" if metrics_b.f1 < metrics_a.f1 else "same"
    }


def apply_guardrails(
    current_threshold: float,
    recommended_threshold: float,
    max_delta: float = 0.10,
    allow_large_changes: bool = False
) -> Tuple[float, str, List[str]]:
    """
    Apply guardrails to threshold changes.

    Prevents large sudden changes in thresholds that could destabilize the system.

    Args:
        current_threshold: Current threshold in use
        recommended_threshold: Recommended new threshold
        max_delta: Maximum allowed change
        allow_large_changes: Whether to allow changes exceeding max_delta

    Returns:
        Tuple of (final_threshold, status, reasons)

    Example:
        >>> final, status, reasons = apply_guardrails(0.5, 0.7, max_delta=0.10)
        >>> status
        'clamped'
        >>> final
        0.6
    """
    delta = recommended_threshold - current_threshold
    abs_delta = abs(delta)

    if abs_delta <= max_delta:
        # Within guardrails
        status = "ok" if abs_delta > 0.001 else "no_change"
        reasons = ["within_guardrails"] if abs_delta > 0.001 else ["no_change"]
        return recommended_threshold, status, reasons

    elif allow_large_changes:
        # Large change allowed
        return recommended_threshold, "large_change_allowed", ["large_change"]

    else:
        # Clamp to guardrails
        if delta > 0:
            clamped = current_threshold + max_delta
        else:
            clamped = current_threshold - max_delta

        return clamped, "clamped", ["exceeds_guardrails", "clamped_to_limit"]


__all__ = [
    "ThresholdMetrics",
    "ThresholdRecommendation",
    "compute_confusion_matrix",
    "sweep_thresholds",
    "optimize_classification_threshold",
    "compare_thresholds",
    "apply_guardrails",
]
