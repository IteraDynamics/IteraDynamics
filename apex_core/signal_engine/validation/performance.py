"""
Performance Validation and Statistical Testing

Provides statistical tests and validation methods for model performance metrics,
including confidence intervals, significance testing, and performance comparisons.

Key concepts:
- Confidence intervals: Uncertainty bounds for performance metrics
- Statistical significance: Whether observed differences are likely real
- Bootstrap resampling: Non-parametric confidence interval estimation
- Sharpe ratio testing: Statistical tests for risk-adjusted returns

Example:
    ```python
    from . import (
        compute_confidence_interval,
        is_statistically_significant,
        compare_models_performance
    )

    # Model A results
    returns_a = [0.02, -0.01, 0.03, 0.01, -0.005, 0.015, 0.02, -0.008]

    # Compute confidence interval for mean return
    ci = compute_confidence_interval(returns_a, confidence=0.95)
    print(f"95% CI for mean: [{ci.lower:.4f}, {ci.upper:.4f}]")

    # Compare two models
    returns_b = [0.015, -0.005, 0.025, 0.008, -0.003, 0.012, 0.018, -0.006]
    result = compare_models_performance(returns_a, returns_b)
    print(f"Statistically significant: {result.is_significant}")
    ```
"""

from __future__ import annotations

import math
import random
import statistics
from dataclasses import dataclass
from typing import List, Optional, Tuple


# -----------------------
# Data Models
# -----------------------

@dataclass
class ConfidenceInterval:
    """
    Confidence interval for a statistic.

    Attributes:
        point_estimate: Point estimate (e.g., mean, median)
        lower: Lower bound of confidence interval
        upper: Upper bound of confidence interval
        confidence_level: Confidence level (e.g., 0.95 for 95%)
        method: Method used ("bootstrap", "normal", "t")
    """
    point_estimate: float
    lower: float
    upper: float
    confidence_level: float
    method: str

    @property
    def width(self) -> float:
        """Width of the confidence interval."""
        return self.upper - self.lower

    @property
    def margin_of_error(self) -> float:
        """Margin of error (half the width)."""
        return self.width / 2.0

    def contains(self, value: float) -> bool:
        """Check if value is within the confidence interval."""
        return self.lower <= value <= self.upper


@dataclass
class StatisticalTest:
    """
    Result of a statistical hypothesis test.

    Attributes:
        statistic: Test statistic value
        p_value: P-value of the test
        is_significant: Whether result is statistically significant
        alpha: Significance level used
        test_name: Name of the test performed
    """
    statistic: float
    p_value: float
    is_significant: bool
    alpha: float
    test_name: str


@dataclass
class PerformanceComparison:
    """
    Result of comparing two model performances.

    Attributes:
        mean_a: Mean performance of model A
        mean_b: Mean performance of model B
        difference: Difference in means (B - A)
        test_result: Statistical test result
        confidence_interval: CI for the difference
    """
    mean_a: float
    mean_b: float
    difference: float
    test_result: StatisticalTest
    confidence_interval: Optional[ConfidenceInterval] = None

    @property
    def is_significant(self) -> bool:
        """Whether difference is statistically significant."""
        return self.test_result.is_significant

    @property
    def improvement_pct(self) -> float:
        """Percentage improvement from A to B."""
        if self.mean_a == 0:
            return 0.0
        return (self.difference / abs(self.mean_a)) * 100.0


# -----------------------
# Statistical Functions
# -----------------------

def compute_confidence_interval(
    data: List[float],
    confidence: float = 0.95,
    method: str = "bootstrap",
    n_bootstrap: int = 1000
) -> ConfidenceInterval:
    """
    Compute confidence interval for the mean of data.

    Args:
        data: List of numeric values
        confidence: Confidence level (e.g., 0.95 for 95%)
        method: Method to use ("bootstrap", "normal", "t")
        n_bootstrap: Number of bootstrap samples (for bootstrap method)

    Returns:
        ConfidenceInterval object

    Example:
        >>> data = [0.02, -0.01, 0.03, 0.01, -0.005, 0.015]
        >>> ci = compute_confidence_interval(data, confidence=0.95)
        >>> ci.point_estimate > 0
        True
    """
    if not data:
        return ConfidenceInterval(0.0, 0.0, 0.0, confidence, method)

    mean = statistics.mean(data)

    if method == "bootstrap":
        return _bootstrap_confidence_interval(data, mean, confidence, n_bootstrap)
    elif method == "normal":
        return _normal_confidence_interval(data, mean, confidence)
    elif method == "t":
        return _t_confidence_interval(data, mean, confidence)
    else:
        raise ValueError(f"Unknown method: {method}")


def _bootstrap_confidence_interval(
    data: List[float],
    mean: float,
    confidence: float,
    n_bootstrap: int
) -> ConfidenceInterval:
    """Bootstrap confidence interval for mean."""
    n = len(data)
    bootstrap_means = []

    for _ in range(n_bootstrap):
        # Resample with replacement
        sample = [random.choice(data) for _ in range(n)]
        bootstrap_means.append(statistics.mean(sample))

    # Percentile method
    alpha = 1 - confidence
    lower_pct = alpha / 2
    upper_pct = 1 - (alpha / 2)

    sorted_means = sorted(bootstrap_means)
    lower_idx = int(lower_pct * n_bootstrap)
    upper_idx = int(upper_pct * n_bootstrap)

    return ConfidenceInterval(
        point_estimate=mean,
        lower=sorted_means[lower_idx],
        upper=sorted_means[upper_idx],
        confidence_level=confidence,
        method="bootstrap"
    )


def _normal_confidence_interval(
    data: List[float],
    mean: float,
    confidence: float
) -> ConfidenceInterval:
    """Normal distribution confidence interval for mean."""
    n = len(data)
    if n < 2:
        return ConfidenceInterval(mean, mean, mean, confidence, "normal")

    std = statistics.stdev(data)
    se = std / math.sqrt(n)

    # Z-score for confidence level (approximation)
    z = {
        0.90: 1.645,
        0.95: 1.960,
        0.99: 2.576,
        0.999: 3.291,
    }.get(confidence, 1.960)

    margin = z * se

    return ConfidenceInterval(
        point_estimate=mean,
        lower=mean - margin,
        upper=mean + margin,
        confidence_level=confidence,
        method="normal"
    )


def _t_confidence_interval(
    data: List[float],
    mean: float,
    confidence: float
) -> ConfidenceInterval:
    """Student's t-distribution confidence interval for mean."""
    n = len(data)
    if n < 2:
        return ConfidenceInterval(mean, mean, mean, confidence, "t")

    std = statistics.stdev(data)
    se = std / math.sqrt(n)

    # T-score approximation (for common confidence levels)
    # This is a simplified version; full implementation would use scipy
    df = n - 1
    t_approx = {
        (0.95, 5): 2.571,
        (0.95, 10): 2.228,
        (0.95, 20): 2.086,
        (0.95, 30): 2.042,
        (0.95, float('inf')): 1.960,
    }

    # Find closest df
    t = t_approx.get((confidence, df))
    if t is None:
        # Use approximation for large df
        if df > 30:
            t = t_approx[(confidence, float('inf'))]
        else:
            # Fallback to normal
            t = 1.960

    margin = t * se

    return ConfidenceInterval(
        point_estimate=mean,
        lower=mean - margin,
        upper=mean + margin,
        confidence_level=confidence,
        method="t"
    )


def is_statistically_significant(
    p_value: float,
    alpha: float = 0.05
) -> bool:
    """
    Check if p-value indicates statistical significance.

    Args:
        p_value: P-value from statistical test
        alpha: Significance level (default: 0.05)

    Returns:
        True if p-value < alpha

    Example:
        >>> is_statistically_significant(0.03, alpha=0.05)
        True
        >>> is_statistically_significant(0.10, alpha=0.05)
        False
    """
    return p_value < alpha


def compute_sharpe_ratio(
    returns: List[float],
    risk_free_rate: float = 0.0
) -> float:
    """
    Compute Sharpe ratio for a series of returns.

    Args:
        returns: List of period returns
        risk_free_rate: Risk-free rate per period (default: 0.0)

    Returns:
        Sharpe ratio (annualized if returns are daily)

    Example:
        >>> returns = [0.02, -0.01, 0.03, 0.01, -0.005]
        >>> sharpe = compute_sharpe_ratio(returns)
        >>> sharpe > 0
        True
    """
    if not returns or len(returns) < 2:
        return 0.0

    excess_returns = [r - risk_free_rate for r in returns]
    mean_excess = statistics.mean(excess_returns)
    std_excess = statistics.stdev(excess_returns)

    if std_excess == 0:
        return 0.0

    return mean_excess / std_excess


def compute_sortino_ratio(
    returns: List[float],
    risk_free_rate: float = 0.0,
    target_return: float = 0.0
) -> float:
    """
    Compute Sortino ratio (uses downside deviation instead of total volatility).

    Args:
        returns: List of period returns
        risk_free_rate: Risk-free rate per period
        target_return: Target return for downside calculation

    Returns:
        Sortino ratio

    Example:
        >>> returns = [0.02, -0.01, 0.03, 0.01, -0.005]
        >>> sortino = compute_sortino_ratio(returns)
        >>> sortino > 0
        True
    """
    if not returns or len(returns) < 2:
        return 0.0

    excess_returns = [r - risk_free_rate for r in returns]
    mean_excess = statistics.mean(excess_returns)

    # Downside deviation (only negative deviations from target)
    downside_diffs = [min(0, r - target_return) for r in returns]
    downside_var = statistics.fmean([d ** 2 for d in downside_diffs])
    downside_std = math.sqrt(downside_var)

    if downside_std == 0:
        return 0.0

    return mean_excess / downside_std


def compare_models_performance(
    returns_a: List[float],
    returns_b: List[float],
    alpha: float = 0.05
) -> PerformanceComparison:
    """
    Compare performance of two models using permutation test.

    Tests whether the difference in mean returns is statistically significant.

    Args:
        returns_a: Returns from model A
        returns_b: Returns from model B
        alpha: Significance level

    Returns:
        PerformanceComparison with test results

    Example:
        >>> returns_a = [0.02, -0.01, 0.03]
        >>> returns_b = [0.03, 0.01, 0.04]  # Better
        >>> result = compare_models_performance(returns_a, returns_b)
        >>> result.difference > 0
        True
    """
    if not returns_a or not returns_b:
        raise ValueError("Both return series must be non-empty")

    mean_a = statistics.mean(returns_a)
    mean_b = statistics.mean(returns_b)
    observed_diff = mean_b - mean_a

    # Permutation test
    p_value = _permutation_test(returns_a, returns_b, observed_diff)

    test_result = StatisticalTest(
        statistic=observed_diff,
        p_value=p_value,
        is_significant=p_value < alpha,
        alpha=alpha,
        test_name="permutation_test"
    )

    # Compute CI for difference using bootstrap
    combined = returns_a + returns_b
    n_a = len(returns_a)

    bootstrap_diffs = []
    for _ in range(1000):
        sample = random.sample(combined, len(combined))
        boot_a = sample[:n_a]
        boot_b = sample[n_a:]
        boot_diff = statistics.mean(boot_b) - statistics.mean(boot_a)
        bootstrap_diffs.append(boot_diff)

    sorted_diffs = sorted(bootstrap_diffs)
    ci = ConfidenceInterval(
        point_estimate=observed_diff,
        lower=sorted_diffs[25],  # 2.5th percentile
        upper=sorted_diffs[975],  # 97.5th percentile
        confidence_level=0.95,
        method="bootstrap"
    )

    return PerformanceComparison(
        mean_a=mean_a,
        mean_b=mean_b,
        difference=observed_diff,
        test_result=test_result,
        confidence_interval=ci
    )


def _permutation_test(
    group_a: List[float],
    group_b: List[float],
    observed_diff: float,
    n_permutations: int = 1000
) -> float:
    """
    Permutation test for difference in means.

    Args:
        group_a: First group of values
        group_b: Second group of values
        observed_diff: Observed difference in means (B - A)
        n_permutations: Number of permutations

    Returns:
        P-value
    """
    combined = group_a + group_b
    n_a = len(group_a)
    count_extreme = 0

    for _ in range(n_permutations):
        # Randomly shuffle and split
        shuffled = random.sample(combined, len(combined))
        perm_a = shuffled[:n_a]
        perm_b = shuffled[n_a:]

        perm_diff = statistics.mean(perm_b) - statistics.mean(perm_a)

        # Two-tailed test
        if abs(perm_diff) >= abs(observed_diff):
            count_extreme += 1

    return count_extreme / n_permutations


__all__ = [
    "ConfidenceInterval",
    "StatisticalTest",
    "PerformanceComparison",
    "compute_confidence_interval",
    "is_statistically_significant",
    "compute_sharpe_ratio",
    "compute_sortino_ratio",
    "compare_models_performance",
]
