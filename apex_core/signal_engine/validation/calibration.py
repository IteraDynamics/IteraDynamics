"""
Model Calibration Analysis

Provides tools for analyzing model calibration including Expected Calibration Error (ECE),
Brier score, and reliability diagram computation.

Key concepts:
- Calibration: How well predicted probabilities match observed frequencies
- ECE (Expected Calibration Error): Weighted average of calibration errors across bins
- Brier Score: Mean squared error between predictions and outcomes
- Reliability Diagram: Visualization showing predicted vs. empirical probabilities

Example:
    ```python
    from . import (
        compute_calibration_metrics,
        compute_ece_and_bins,
        compute_brier_score
    )

    # Binary predictions and outcomes
    y_true = [1, 0, 1, 1, 0, 1, 0, 0, 1, 1]
    y_prob = [0.9, 0.2, 0.8, 0.7, 0.3, 0.85, 0.15, 0.25, 0.75, 0.95]

    # Compute calibration metrics
    result = compute_calibration_metrics(y_true, y_prob, n_bins=10)

    print(f"ECE: {result.ece:.4f}")
    print(f"Brier: {result.brier:.4f}")

    # Check calibration bins
    for bin in result.bins:
        if bin.count > 0:
            print(f"Bin [{bin.bin_low:.2f}, {bin.bin_high:.2f}]: "
                  f"Predicted={bin.avg_confidence:.2f}, "
                  f"Empirical={bin.empirical_rate:.2f}, "
                  f"Count={bin.count}")
    ```
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from typing import List, Optional, Tuple


# -----------------------
# Data Models
# -----------------------

@dataclass
class CalibrationBin:
    """
    Single bin in a calibration/reliability diagram.

    Attributes:
        bin_low: Lower edge of bin (inclusive)
        bin_high: Upper edge of bin (exclusive, except last bin includes 1.0)
        avg_confidence: Average predicted probability in this bin
        empirical_rate: Empirical outcome rate (fraction of positives)
        count: Number of samples in this bin
    """
    bin_low: float
    bin_high: float
    avg_confidence: Optional[float] = None
    empirical_rate: Optional[float] = None
    count: int = 0

    def to_dict(self) -> dict:
        """Convert to dictionary representation."""
        return {
            "bin_low": self.bin_low,
            "bin_high": self.bin_high,
            "avg_conf": self.avg_confidence,
            "empirical": self.empirical_rate,
            "count": self.count,
        }

    @property
    def calibration_error(self) -> Optional[float]:
        """
        Absolute calibration error for this bin.

        Returns:
            |empirical_rate - avg_confidence| or None if bin is empty
        """
        if self.avg_confidence is None or self.empirical_rate is None:
            return None
        return abs(self.empirical_rate - self.avg_confidence)


@dataclass
class CalibrationMetrics:
    """
    Complete calibration analysis results.

    Attributes:
        ece: Expected Calibration Error (weighted mean absolute error)
        brier: Brier score (mean squared error)
        bins: List of calibration bins
        n_samples: Total number of samples
        n_bins: Number of bins used
    """
    ece: float
    brier: float
    bins: List[CalibrationBin]
    n_samples: int
    n_bins: int

    def to_dict(self) -> dict:
        """Convert to dictionary representation."""
        return {
            "ece": self.ece,
            "brier": self.brier,
            "n_samples": self.n_samples,
            "n_bins": self.n_bins,
            "bins": [b.to_dict() for b in self.bins],
        }

    @property
    def max_calibration_error(self) -> float:
        """
        Maximum calibration error across all bins.

        Returns:
            Maximum absolute calibration error
        """
        errors = [b.calibration_error for b in self.bins if b.calibration_error is not None]
        return max(errors) if errors else 0.0


# -----------------------
# Core Calibration Functions
# -----------------------

def compute_brier_score(
    y_true: List[int],
    y_prob: List[float]
) -> float:
    """
    Compute Brier score (mean squared error between predictions and outcomes).

    Args:
        y_true: True binary labels (0 or 1)
        y_prob: Predicted probabilities (0.0 to 1.0)

    Returns:
        Brier score (lower is better, range [0, 1])

    Example:
        >>> y_true = [1, 0, 1, 1, 0]
        >>> y_prob = [0.9, 0.2, 0.8, 0.7, 0.3]
        >>> brier = compute_brier_score(y_true, y_prob)
        >>> brier < 0.1  # Well-calibrated
        True
    """
    if not y_prob or not y_true:
        return 0.0

    if len(y_true) != len(y_prob):
        raise ValueError(f"Length mismatch: y_true={len(y_true)}, y_prob={len(y_prob)}")

    # Clamp probabilities to [0, 1]
    probs = [min(1.0, max(0.0, float(p))) for p in y_prob]

    # Convert labels to 0/1
    trues = [1 if int(y) else 0 for y in y_true]

    # Mean squared error
    return float(statistics.fmean((p - t) ** 2 for p, t in zip(probs, trues)))


def create_bin_edges(n_bins: int) -> List[Tuple[float, float]]:
    """
    Create equal-width bin edges over [0, 1].

    Args:
        n_bins: Number of bins to create

    Returns:
        List of (low, high) tuples for each bin

    Example:
        >>> edges = create_bin_edges(4)
        >>> edges
        [(0.0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.0)]
    """
    if n_bins <= 0:
        raise ValueError(f"n_bins must be positive, got {n_bins}")

    step = 1.0 / n_bins
    return [(i * step, (i + 1) * step) for i in range(n_bins)]


def assign_to_bin(probability: float, n_bins: int) -> int:
    """
    Assign a probability to a bin index.

    Args:
        probability: Predicted probability (0.0 to 1.0)
        n_bins: Total number of bins

    Returns:
        Bin index (0 to n_bins-1)

    Example:
        >>> assign_to_bin(0.75, n_bins=10)
        7
        >>> assign_to_bin(1.0, n_bins=10)  # Edge case: 1.0 goes to last bin
        9
    """
    # Clamp to [0, 1]
    p = min(1.0, max(0.0, float(probability)))

    # Assign to bin (last bin includes 1.0)
    return min(int(p * n_bins), n_bins - 1)


def compute_ece_and_bins(
    y_true: List[int],
    y_prob: List[float],
    n_bins: int = 10
) -> Tuple[float, List[CalibrationBin]]:
    """
    Compute Expected Calibration Error and reliability diagram bins.

    Uses equal-width bins over [0, 1] interval. ECE is the weighted average
    of absolute calibration errors across bins.

    Args:
        y_true: True binary labels (0 or 1)
        y_prob: Predicted probabilities (0.0 to 1.0)
        n_bins: Number of bins for reliability diagram (default: 10)

    Returns:
        Tuple of (ECE, list of CalibrationBin objects)

    Example:
        >>> y_true = [1, 0, 1, 1, 0, 1, 0, 0, 1, 1]
        >>> y_prob = [0.9, 0.2, 0.8, 0.7, 0.3, 0.85, 0.15, 0.25, 0.75, 0.95]
        >>> ece, bins = compute_ece_and_bins(y_true, y_prob, n_bins=5)
        >>> ece < 0.2  # Reasonably calibrated
        True
        >>> len(bins)
        5
    """
    if not y_prob or not y_true:
        # Return empty bins
        edges = create_bin_edges(n_bins)
        empty_bins = [
            CalibrationBin(low, high, None, None, 0)
            for low, high in edges
        ]
        return 0.0, empty_bins

    if len(y_true) != len(y_prob):
        raise ValueError(f"Length mismatch: y_true={len(y_true)}, y_prob={len(y_prob)}")

    n = len(y_true)

    # Create bin edges
    edges = create_bin_edges(n_bins)

    # Accumulate samples per bin
    bin_accumulators = [{
        "sum_prob": 0.0,
        "sum_label": 0.0,
        "count": 0
    } for _ in range(n_bins)]

    for y, p in zip(y_true, y_prob):
        bin_idx = assign_to_bin(p, n_bins)
        acc = bin_accumulators[bin_idx]
        acc["sum_prob"] += min(1.0, max(0.0, float(p)))
        acc["sum_label"] += 1 if int(y) else 0
        acc["count"] += 1

    # Compute bins and ECE
    bins = []
    ece = 0.0

    for (low, high), acc in zip(edges, bin_accumulators):
        count = acc["count"]

        if count > 0:
            avg_conf = acc["sum_prob"] / count
            emp_rate = acc["sum_label"] / count
            # Contribution to ECE
            ece += (count / n) * abs(emp_rate - avg_conf)
        else:
            avg_conf = None
            emp_rate = None

        bin = CalibrationBin(
            bin_low=low,
            bin_high=high,
            avg_confidence=avg_conf,
            empirical_rate=emp_rate,
            count=count
        )
        bins.append(bin)

    return float(ece), bins


def compute_calibration_metrics(
    y_true: List[int],
    y_prob: List[float],
    n_bins: int = 10
) -> CalibrationMetrics:
    """
    Compute comprehensive calibration metrics.

    Computes both ECE and Brier score along with reliability diagram bins.

    Args:
        y_true: True binary labels (0 or 1)
        y_prob: Predicted probabilities (0.0 to 1.0)
        n_bins: Number of bins for reliability diagram (default: 10)

    Returns:
        CalibrationMetrics with all calibration analysis results

    Example:
        >>> y_true = [1, 0, 1, 1, 0, 1, 0, 0, 1, 1]
        >>> y_prob = [0.9, 0.2, 0.8, 0.7, 0.3, 0.85, 0.15, 0.25, 0.75, 0.95]
        >>> metrics = compute_calibration_metrics(y_true, y_prob, n_bins=10)
        >>> metrics.ece < 0.2
        True
        >>> metrics.brier < 0.2
        True
        >>> metrics.n_samples
        10
    """
    if not y_true or not y_prob:
        edges = create_bin_edges(n_bins)
        empty_bins = [CalibrationBin(low, high, None, None, 0) for low, high in edges]
        return CalibrationMetrics(
            ece=0.0,
            brier=0.0,
            bins=empty_bins,
            n_samples=0,
            n_bins=n_bins
        )

    ece, bins = compute_ece_and_bins(y_true, y_prob, n_bins)
    brier = compute_brier_score(y_true, y_prob)

    return CalibrationMetrics(
        ece=ece,
        brier=brier,
        bins=bins,
        n_samples=len(y_true),
        n_bins=n_bins
    )


def is_well_calibrated(
    metrics: CalibrationMetrics,
    ece_threshold: float = 0.1,
    brier_threshold: float = 0.25
) -> bool:
    """
    Check if model is well-calibrated based on thresholds.

    Args:
        metrics: CalibrationMetrics to evaluate
        ece_threshold: Maximum acceptable ECE (default: 0.1)
        brier_threshold: Maximum acceptable Brier score (default: 0.25)

    Returns:
        True if model is well-calibrated (both metrics below thresholds)

    Example:
        >>> y_true = [1, 0, 1, 1, 0]
        >>> y_prob = [0.95, 0.05, 0.85, 0.75, 0.15]  # Well-calibrated
        >>> metrics = compute_calibration_metrics(y_true, y_prob)
        >>> is_well_calibrated(metrics)
        True
    """
    return metrics.ece <= ece_threshold and metrics.brier <= brier_threshold


def compute_calibration_by_group(
    y_true: List[int],
    y_prob: List[float],
    groups: List[str],
    n_bins: int = 10
) -> dict[str, CalibrationMetrics]:
    """
    Compute calibration metrics separately for each group.

    Useful for per-origin, per-model, or per-feature calibration analysis.

    Args:
        y_true: True binary labels
        y_prob: Predicted probabilities
        groups: Group identifier for each sample
        n_bins: Number of bins for reliability diagrams

    Returns:
        Dictionary mapping group name to CalibrationMetrics

    Example:
        >>> y_true = [1, 0, 1, 1, 0, 1]
        >>> y_prob = [0.9, 0.2, 0.8, 0.7, 0.3, 0.85]
        >>> groups = ["A", "A", "A", "B", "B", "B"]
        >>> results = compute_calibration_by_group(y_true, y_prob, groups)
        >>> "A" in results and "B" in results
        True
    """
    if len(y_true) != len(y_prob) or len(y_true) != len(groups):
        raise ValueError("y_true, y_prob, and groups must have same length")

    # Group samples
    grouped = {}
    for y, p, g in zip(y_true, y_prob, groups):
        if g not in grouped:
            grouped[g] = {"y_true": [], "y_prob": []}
        grouped[g]["y_true"].append(y)
        grouped[g]["y_prob"].append(p)

    # Compute metrics per group
    results = {}
    for group, data in grouped.items():
        results[group] = compute_calibration_metrics(
            data["y_true"],
            data["y_prob"],
            n_bins=n_bins
        )

    return results


__all__ = [
    "CalibrationBin",
    "CalibrationMetrics",
    "compute_brier_score",
    "create_bin_edges",
    "assign_to_bin",
    "compute_ece_and_bins",
    "compute_calibration_metrics",
    "is_well_calibrated",
    "compute_calibration_by_group",
]
