"""
Automated Retrain Triggering

Automatically decides when to retrain models based on drift detection,
calibration degradation, and performance metrics.

Key concepts:
- Trigger conditions: Persistent drift, calibration degradation, performance decay
- Retrain planning: Dry-run mode for testing before actual retraining
- Data requirements: Labels, lookback period, training set size
- Decision modes: plan, hold, execute

Example:
    ```python
    from . import (
        RetrainConfig,
        DriftCandidate,
        CalibrationPoint,
        plan_retraining
    )

    # Configure retrain automation
    config = RetrainConfig(
        lookback_days=30,
        min_labels=1000,
        action_mode="plan"
    )

    # Drift candidates from drift detection
    drift_candidates = [
        DriftCandidate(
            origin="news_sentiment",
            model_version="v0.7.7",
            current_threshold=0.50,
            proposed_threshold=0.54,
            delta=0.04,
            reasons=["high_ece_persistent"],
            decision="proceed"
        )
    ]

    # Recent calibration points
    calibration_points = [
        CalibrationPoint(
            bucket_start="2025-10-30T12:00:00Z",
            ece=0.09,
            n=60
        )
    ]

    # Plan retraining
    plan = plan_retraining(
        drift_candidates=drift_candidates,
        calibration_points=calibration_points,
        ece_threshold=0.06,
        config=config
    )

    for candidate in plan.candidates:
        print(f"{candidate.origin}: {candidate.decision}")
        print(f"  Reasons: {', '.join(candidate.reasons)}")
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
class RetrainConfig:
    """
    Configuration for retrain automation.

    Attributes:
        lookback_days: Number of days of historical data to use for retraining
        min_labels: Minimum number of labeled examples required to retrain
        action_mode: Mode of operation ("dryrun", "plan", "execute")
        ece_threshold: ECE threshold for considering calibration degraded
    """
    lookback_days: int = 30
    min_labels: int = 1000
    action_mode: str = "dryrun"
    ece_threshold: float = 0.06


# -----------------------
# Data Models
# -----------------------

@dataclass
class RetrainCandidate:
    """
    A model/origin candidate for retraining.

    Attributes:
        origin: Data origin or model identifier
        current_version: Current model version
        reasons: List of reasons triggering retrain
        window_days: Lookback window for training data
        estimated_labels: Estimated number of labels available
        dataset_info: Information about training dataset
        expected_impact: Expected performance impact from retraining
        decision: Retrain decision ("plan", "hold", "execute")
        new_version: Proposed new version identifier
    """
    origin: str
    current_version: str
    reasons: List[str] = field(default_factory=list)
    window_days: int = 30
    estimated_labels: int = 0
    dataset_info: Optional[Dict[str, Any]] = None
    expected_impact: Optional[Dict[str, float]] = None
    decision: str = "hold"
    new_version: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        result = {
            "origin": self.origin,
            "current_version": self.current_version,
            "reason": self.reasons,  # Note: using "reason" for backwards compatibility
            "window_days": self.window_days,
            "labels": self.estimated_labels,
            "decision": self.decision,
            "new_version": self.new_version,
        }
        if self.dataset_info:
            result["datasets"] = self.dataset_info
        if self.expected_impact:
            result["eval"] = self.expected_impact
        return result

    @classmethod
    def from_drift_candidate(
        cls,
        drift_candidate: Any,  # DriftCandidate from drift_response module
        window_days: int = 30,
        still_high_ece: bool = False
    ) -> RetrainCandidate:
        """
        Create retrain candidate from drift candidate.

        Args:
            drift_candidate: DriftCandidate from drift detection
            window_days: Lookback window for training data
            still_high_ece: Whether calibration is still showing high ECE

        Returns:
            RetrainCandidate with retrain decision
        """
        # Combine reasons
        reasons = list(drift_candidate.reasons)
        if still_high_ece and "still_high" not in reasons:
            reasons.append("still_high")

        # Deduplicate while preserving order
        seen = set()
        unique_reasons = []
        for r in reasons:
            if r not in seen:
                seen.add(r)
                unique_reasons.append(r)

        # Decision logic
        decision = "plan" if still_high_ece else "hold"

        return cls(
            origin=drift_candidate.origin,
            current_version=drift_candidate.model_version,
            reasons=unique_reasons,
            window_days=window_days,
            estimated_labels=0,  # Would be filled by data analysis
            dataset_info={"path": ""},
            expected_impact={
                "precision_delta": 0.0,
                "ece_delta": -0.01,  # Expected calibration improvement
                "f1_delta": 0.0,
            },
            decision=decision,
            new_version=None,
        )


@dataclass
class RetrainPlan:
    """
    Complete retraining plan.

    Attributes:
        generated_at: Timestamp of plan generation
        action_mode: Mode of operation
        candidates: List of retrain candidates
        total_candidates: Total number of candidates
    """
    generated_at: datetime
    action_mode: str
    candidates: List[RetrainCandidate] = field(default_factory=list)

    @property
    def total_candidates(self) -> int:
        """Get total number of candidates."""
        return len(self.candidates)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "generated_at": self.generated_at.isoformat(),
            "action_mode": self.action_mode,
            "candidates": [c.to_dict() for c in self.candidates],
        }


# -----------------------
# Core Logic
# -----------------------

def check_calibration_still_high(
    calibration_points: List[Any],  # List[CalibrationPoint]
    ece_threshold: float
) -> bool:
    """
    Check if the most recent calibration point still shows high ECE.

    Args:
        calibration_points: Time series of calibration measurements
        ece_threshold: ECE threshold for "high"

    Returns:
        True if most recent point has ECE > threshold

    Example:
        >>> from .drift_response import CalibrationPoint
        >>> points = [
        ...     CalibrationPoint("2025-10-30T10:00:00Z", ece=0.05, n=50),
        ...     CalibrationPoint("2025-10-30T12:00:00Z", ece=0.09, n=60)
        ... ]
        >>> check_calibration_still_high(points, ece_threshold=0.06)
        True
    """
    if not calibration_points:
        return False

    try:
        last_point = calibration_points[-1]
        return last_point.ece > ece_threshold
    except (AttributeError, TypeError, KeyError):
        return False


def should_retrain_candidate(
    drift_candidate: Any,  # DriftCandidate
    still_high_ece: bool
) -> bool:
    """
    Decide if a drift candidate should trigger retraining.

    Args:
        drift_candidate: Candidate from drift detection
        still_high_ece: Whether calibration is still showing high ECE

    Returns:
        True if retraining should be planned

    Example:
        >>> from .drift_response import DriftCandidate
        >>> candidate = DriftCandidate(
        ...     origin="news",
        ...     model_version="v1",
        ...     current_threshold=0.5,
        ...     proposed_threshold=0.54,
        ...     delta=0.04,
        ...     reasons=["high_ece_persistent"],
        ...     decision="proceed"
        ... )
        >>> should_retrain_candidate(candidate, still_high_ece=True)
        True
    """
    reasons = getattr(drift_candidate, "reasons", [])
    decision = getattr(drift_candidate, "decision", "")

    # Check if high ECE is persistent
    has_high_ece_reason = "high_ece_persistent" in reasons

    # Check if decision is to proceed
    should_proceed = decision == "proceed"

    # Retrain if either condition is met AND calibration is still high
    return (has_high_ece_reason or should_proceed) and still_high_ece


def plan_retraining(
    drift_candidates: List[Any],  # List[DriftCandidate]
    calibration_points: Optional[List[Any]] = None,  # List[CalibrationPoint]
    ece_threshold: float = 0.06,
    config: Optional[RetrainConfig] = None,
) -> RetrainPlan:
    """
    Create retraining plan based on drift candidates and calibration data.

    Args:
        drift_candidates: Candidates from drift detection
        calibration_points: Recent calibration measurements
        ece_threshold: ECE threshold for considering calibration degraded
        config: Retrain configuration (uses defaults if None)

    Returns:
        RetrainPlan with candidates and decisions

    Example:
        >>> from .drift_response import DriftCandidate, CalibrationPoint
        >>> drift_cands = [
        ...     DriftCandidate(
        ...         origin="news",
        ...         model_version="v1",
        ...         current_threshold=0.5,
        ...         proposed_threshold=0.54,
        ...         delta=0.04,
        ...         reasons=["high_ece_persistent"],
        ...         decision="proceed"
        ...     )
        ... ]
        >>> cal_points = [CalibrationPoint("2025-10-30T12:00:00Z", ece=0.09, n=60)]
        >>> plan = plan_retraining(drift_cands, cal_points, ece_threshold=0.06)
        >>> len(plan.candidates)
        1
    """
    if config is None:
        config = RetrainConfig()

    if calibration_points is None:
        calibration_points = []

    # Check if calibration is still showing high ECE
    still_high = check_calibration_still_high(calibration_points, ece_threshold)

    # Build retrain candidates
    retrain_candidates = []

    for drift_candidate in drift_candidates:
        # Check if this candidate should trigger retrain
        if should_retrain_candidate(drift_candidate, still_high):
            retrain_candidate = RetrainCandidate.from_drift_candidate(
                drift_candidate=drift_candidate,
                window_days=config.lookback_days,
                still_high_ece=still_high
            )
            retrain_candidates.append(retrain_candidate)

    return RetrainPlan(
        generated_at=datetime.now(timezone.utc),
        action_mode=config.action_mode,
        candidates=retrain_candidates,
    )


def plan_retraining_from_json(
    calibration_data: Optional[Dict[str, Any]] = None,
    drift_plan_data: Optional[Dict[str, Any]] = None,
    config: Optional[RetrainConfig] = None,
) -> RetrainPlan:
    """
    Create retraining plan from JSON-like dictionaries.

    This is a convenience method for integrating with existing JSON-based systems.

    Args:
        calibration_data: Calibration trend data with "series" key
        drift_plan_data: Drift response plan with "candidates" key
        config: Retrain configuration

    Returns:
        RetrainPlan with candidates and decisions

    Example:
        >>> cal_data = {
        ...     "series": [{
        ...         "points": [{"bucket_start": "2025-10-30T12:00:00Z", "ece": 0.09, "n": 60}]
        ...     }]
        ... }
        >>> drift_data = {
        ...     "candidates": [{
        ...         "origin": "news",
        ...         "model_version": "v1",
        ...         "current_threshold": 0.5,
        ...         "proposed_threshold": 0.54,
        ...         "delta": 0.04,
        ...         "reasons": ["high_ece_persistent"],
        ...         "decision": "proceed"
        ...     }],
        ...     "ece_threshold": 0.06
        ... }
        >>> plan = plan_retraining_from_json(cal_data, drift_data)
        >>> len(plan.candidates) > 0
        True
    """
    if config is None:
        config = RetrainConfig()

    if not calibration_data or not drift_plan_data:
        # Return empty plan
        return RetrainPlan(
            generated_at=datetime.now(timezone.utc),
            action_mode=config.action_mode,
            candidates=[]
        )

    # Import here to avoid circular dependency
    from .drift_response import (
        DriftCandidate,
        CalibrationPoint
    )

    # Extract calibration points
    calibration_points = []
    series = calibration_data.get("series", [])
    for s in series:
        for point_dict in s.get("points", []):
            try:
                point = CalibrationPoint.from_dict(point_dict)
                calibration_points.append(point)
            except Exception:
                pass

    # Extract drift candidates
    drift_candidates = []
    for cand_dict in drift_plan_data.get("candidates", []):
        try:
            # Reconstruct DriftCandidate
            candidate = DriftCandidate(
                origin=cand_dict.get("origin", "unknown"),
                model_version=cand_dict.get("model_version", "v0"),
                current_threshold=float(cand_dict.get("current_threshold", 0.5)),
                proposed_threshold=float(cand_dict.get("proposed_threshold", 0.5)),
                delta=float(cand_dict.get("delta", 0.0)),
                reasons=cand_dict.get("reasons", []),
                decision=cand_dict.get("decision", "observe"),
                backtest_impact=cand_dict.get("backtest")
            )
            drift_candidates.append(candidate)
        except Exception:
            pass

    # Get ECE threshold from drift plan
    ece_threshold = float(drift_plan_data.get("ece_threshold", 0.06))

    return plan_retraining(
        drift_candidates=drift_candidates,
        calibration_points=calibration_points,
        ece_threshold=ece_threshold,
        config=config
    )


__all__ = [
    "RetrainConfig",
    "RetrainCandidate",
    "RetrainPlan",
    "check_calibration_still_high",
    "should_retrain_candidate",
    "plan_retraining",
    "plan_retraining_from_json",
]
