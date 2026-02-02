"""
Signal Engine Threshold Module

Provides comprehensive threshold optimization and analysis including:
- Classification threshold optimization (precision/recall trade-offs)
- Feedback-based threshold simulation
- Threshold sweeping and comparison
- Guardrail application

Example:
    ```python
    # Classification threshold optimization
    from . import optimize_classification_threshold

    scores = [0.9, 0.8, 0.7, 0.6, 0.5]
    labels = [True, True, True, False, False]

    result = optimize_classification_threshold(
        scores, labels,
        strategy="precision_constrained",
        min_precision=0.75
    )
    print(f"Optimal threshold: {result.threshold:.3f}")

    # Feedback-based simulation
    from . import simulate_thresholds

    feedback = [{"confidence": 0.8, "agrees_with_signal": True}, ...]
    results = simulate_thresholds(feedback, min_confidence=0.7)
    ```
"""

# Feedback-based simulation (from simulator.py)
from .simulator import (
    simulate_thresholds,
    analyze_threshold_range,
    find_optimal_threshold,  # For feedback disagreement rate
)

# Classification threshold optimization (from optimization.py)
from .optimization import (
    ThresholdMetrics,
    ThresholdRecommendation,
    compute_confusion_matrix,
    sweep_thresholds,
    optimize_classification_threshold,
    compare_thresholds,
    apply_guardrails,
)

__all__ = [
    # Simulator (feedback-based)
    "simulate_thresholds",
    "analyze_threshold_range",
    "find_optimal_threshold",
    # Optimization (classification-based)
    "ThresholdMetrics",
    "ThresholdRecommendation",
    "compute_confusion_matrix",
    "sweep_thresholds",
    "optimize_classification_threshold",
    "compare_thresholds",
    "apply_guardrails",
]
