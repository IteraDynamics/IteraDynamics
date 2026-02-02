"""
Drift Detection and Response

Detects model drift (feature drift, prediction drift, calibration drift) and
recommends automated responses like threshold adjustment or model retraining.

Key concepts:
- Calibration drift: ECE (Expected Calibration Error) degrades over time
- Feature drift: Input data distribution shifts from training data
- Prediction drift: Model output distribution changes
- Grace period: Recent time window to analyze for persistent drift
- Candidates: Origins/models flagged for intervention

Example:
    ```python
    from . import (
        DriftConfig,
        CalibrationPoint,
        detect_drift_candidates
    )

    # Configure drift detection
    config = DriftConfig(
        ece_threshold=0.06,  # ECE above this indicates poor calibration
        min_buckets=3,  # Minimum time buckets with high ECE
        grace_hours=6,  # Look at last 6 hours
        min_samples=10  # Minimum samples per bucket
    )

    # Calibration data for a model/origin
    calibration_points = [
        CalibrationPoint(
            bucket_start="2025-10-30T10:00:00Z",
            ece=0.08,  # High ECE
            n=50
        ),
        CalibrationPoint(
            bucket_start="2025-10-30T11:00:00Z",
            ece=0.07,  # Still high
            n=45
        ),
        CalibrationPoint(
            bucket_start="2025-10-30T12:00:00Z",
            ece=0.09,  # Still high
            n=60
        ),
    ]

    # Detect drift
    candidates = detect_drift_candidates(
        calibration_points=calibration_points,
        origin="news_sentiment",
        model_version="v0.7.7",
        config=config
    )

    if candidates:
        print(f"Drift detected! Recommendations: {candidates[0].reasons}")
    ```
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional


# -----------------------
# Configuration
# -----------------------

@dataclass
class DriftConfig:
    """
    Configuration for drift detection and response.

    Attributes:
        ece_threshold: ECE threshold above which calibration is considered poor
        min_buckets: Minimum number of time buckets with high ECE to trigger
        grace_hours: Time window to analyze for persistent drift (hours)
        min_samples: Minimum samples per bucket to consider reliable
        action_mode: Response mode ("dryrun", "adjust_threshold", "retrain")
    """
    ece_threshold: float = 0.06
    min_buckets: int = 3
    grace_hours: int = 6
    min_samples: int = 10
    action_mode: str = "dryrun"


# -----------------------
# Data Models
# -----------------------

@dataclass
class CalibrationPoint:
    """
    Single calibration measurement at a point in time.

    Attributes:
        bucket_start: Timestamp for this measurement bucket
        ece: Expected Calibration Error for this bucket
        n: Number of samples in this bucket
        brier_score: Optional Brier score (calibration + refinement)
        reliability: Optional reliability score
    """
    bucket_start: str  # ISO timestamp
    ece: float
    n: int
    brier_score: Optional[float] = None
    reliability: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        result = {
            "bucket_start": self.bucket_start,
            "ece": self.ece,
            "n": self.n,
        }
        if self.brier_score is not None:
            result["brier_score"] = self.brier_score
        if self.reliability is not None:
            result["reliability"] = self.reliability
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> CalibrationPoint:
        """Create from dictionary representation."""
        return cls(
            bucket_start=data["bucket_start"],
            ece=float(data.get("ece", 0.0)),
            n=int(data.get("n", 0)),
            brier_score=data.get("brier_score"),
            reliability=data.get("reliability"),
        )


@dataclass
class DriftCandidate:
    """
    A model/origin flagged for drift intervention.

    Attributes:
        origin: Data origin or model identifier
        model_version: Model version identifier
        current_threshold: Current confidence threshold
        proposed_threshold: Proposed new threshold
        delta: Change in threshold (proposed - current)
        reasons: List of reasons for flagging (e.g., "high_ece_persistent")
        backtest_impact: Expected impact on metrics
        decision: Recommended action ("proceed", "observe", "retrain")
        high_ece_count: Number of buckets with high ECE
        avg_ece: Average ECE in recent window
    """
    origin: str
    model_version: str
    current_threshold: float
    proposed_threshold: float
    delta: float
    reasons: List[str] = field(default_factory=list)
    backtest_impact: Optional[Dict[str, float]] = None
    decision: str = "proceed"
    high_ece_count: int = 0
    avg_ece: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        result = {
            "origin": self.origin,
            "model_version": self.model_version,
            "current_threshold": self.current_threshold,
            "proposed_threshold": self.proposed_threshold,
            "delta": self.delta,
            "reasons": self.reasons,
            "decision": self.decision,
            "high_ece_count": self.high_ece_count,
            "avg_ece": self.avg_ece,
        }
        if self.backtest_impact:
            result["backtest"] = self.backtest_impact
        return result


@dataclass
class DriftReport:
    """
    Complete drift detection report.

    Attributes:
        generated_at: Timestamp of report generation
        window_hours: Time window analyzed
        grace_hours: Recent time window analyzed
        min_buckets: Minimum buckets required to flag drift
        ece_threshold: ECE threshold used
        action_mode: Response mode used
        candidates: List of drift candidates flagged
        total_points_analyzed: Total calibration points analyzed
    """
    generated_at: datetime
    window_hours: int
    grace_hours: int
    min_buckets: int
    ece_threshold: float
    action_mode: str
    candidates: List[DriftCandidate] = field(default_factory=list)
    total_points_analyzed: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "generated_at": self.generated_at.isoformat(),
            "window_hours": self.window_hours,
            "grace_hours": self.grace_hours,
            "min_buckets": self.min_buckets,
            "ece_threshold": self.ece_threshold,
            "action_mode": self.action_mode,
            "candidates": [c.to_dict() for c in self.candidates],
            "total_points_analyzed": self.total_points_analyzed,
        }


# -----------------------
# Core Logic
# -----------------------

def parse_timestamp(ts: str) -> Optional[datetime]:
    """
    Parse ISO timestamp string to datetime.

    Args:
        ts: ISO timestamp string (e.g., "2025-10-30T12:00:00Z")

    Returns:
        Datetime in UTC or None if parsing fails

    Example:
        >>> parse_timestamp("2025-10-30T12:00:00Z")
        datetime.datetime(2025, 10, 30, 12, 0, tzinfo=datetime.timezone.utc)
    """
    try:
        # Handle Z suffix
        if ts.endswith("Z"):
            ts = ts[:-1] + "+00:00"
        return datetime.fromisoformat(ts).astimezone(timezone.utc)
    except (ValueError, AttributeError):
        return None


def filter_recent_points(
    points: List[CalibrationPoint],
    grace_hours: int
) -> List[CalibrationPoint]:
    """
    Filter calibration points to recent time window.

    Args:
        points: List of calibration points
        grace_hours: Number of hours to look back

    Returns:
        List of calibration points within grace period

    Example:
        >>> from datetime import datetime, timezone, timedelta
        >>> now = datetime.now(timezone.utc)
        >>> recent_time = (now - timedelta(hours=2)).isoformat()
        >>> old_time = (now - timedelta(hours=10)).isoformat()
        >>> points = [
        ...     CalibrationPoint(bucket_start=recent_time, ece=0.08, n=50),
        ...     CalibrationPoint(bucket_start=old_time, ece=0.07, n=45)
        ... ]
        >>> filtered = filter_recent_points(points, grace_hours=6)
        >>> len(filtered)
        1
    """
    cutoff = datetime.now(timezone.utc) - timedelta(hours=grace_hours)
    recent = []

    for point in points:
        dt = parse_timestamp(point.bucket_start)
        # Include points with no timestamp (assume recent) or points after cutoff
        if dt is None or dt >= cutoff:
            recent.append(point)

    return recent


def detect_drift_candidates(
    calibration_points: List[CalibrationPoint],
    origin: str,
    model_version: str,
    config: Optional[DriftConfig] = None,
    current_threshold: float = 0.50,
) -> List[DriftCandidate]:
    """
    Detect drift candidates from calibration time series.

    Simple detection heuristic:
    1. Filter to recent points within grace period
    2. Count points with ECE > threshold and n >= min_samples
    3. If count >= min_buckets, flag as drift candidate

    Args:
        calibration_points: Time series of calibration measurements
        origin: Data origin or model identifier
        model_version: Model version being monitored
        config: Drift detection configuration (uses defaults if None)
        current_threshold: Current confidence threshold for this origin

    Returns:
        List of drift candidates (empty if no drift detected)

    Example:
        >>> points = [
        ...     CalibrationPoint("2025-10-30T10:00:00Z", ece=0.08, n=50),
        ...     CalibrationPoint("2025-10-30T11:00:00Z", ece=0.07, n=45),
        ...     CalibrationPoint("2025-10-30T12:00:00Z", ece=0.09, n=60),
        ... ]
        >>> candidates = detect_drift_candidates(
        ...     calibration_points=points,
        ...     origin="news_sentiment",
        ...     model_version="v0.7.7"
        ... )
        >>> len(candidates) > 0
        True
    """
    if config is None:
        config = DriftConfig()

    # Filter to recent points
    recent = filter_recent_points(calibration_points, config.grace_hours)

    # Find points with high ECE
    high_ece_points = [
        p for p in recent
        if p.ece > config.ece_threshold and p.n >= config.min_samples
    ]

    # Check if we have enough high ECE points to flag drift
    if len(high_ece_points) < config.min_buckets:
        return []

    # Calculate average ECE
    avg_ece = sum(p.ece for p in high_ece_points) / len(high_ece_points)

    # Propose threshold adjustment (increase by 0.04 to be more conservative)
    proposed_threshold = current_threshold + 0.04
    delta = proposed_threshold - current_threshold

    # Create candidate
    candidate = DriftCandidate(
        origin=origin,
        model_version=model_version,
        current_threshold=current_threshold,
        proposed_threshold=proposed_threshold,
        delta=delta,
        reasons=["high_ece_persistent"],
        backtest_impact={
            "precision_delta": 0.0,  # Would be filled by actual backtest
            "ece_delta": -0.01,  # Expected improvement
            "recall_delta": -0.02,  # Expected recall drop (more conservative)
            "f1_delta": -0.01,  # Net F1 impact
        },
        decision="proceed" if config.action_mode != "dryrun" else "dryrun",
        high_ece_count=len(high_ece_points),
        avg_ece=avg_ece,
    )

    return [candidate]


def analyze_drift_from_series(
    calibration_series: List[Dict[str, Any]],
    config: Optional[DriftConfig] = None,
) -> DriftReport:
    """
    Analyze drift across multiple calibration series.

    Each series represents a different origin or model variant.

    Args:
        calibration_series: List of calibration series, each with:
            - key: Origin or identifier
            - version: Model version
            - points: List of calibration point dictionaries
        config: Drift detection configuration

    Returns:
        DriftReport with all detected candidates

    Example:
        >>> series = [{
        ...     "key": "news_sentiment",
        ...     "version": "v0.7.7",
        ...     "points": [
        ...         {"bucket_start": "2025-10-30T10:00:00Z", "ece": 0.08, "n": 50},
        ...         {"bucket_start": "2025-10-30T11:00:00Z", "ece": 0.07, "n": 45},
        ...         {"bucket_start": "2025-10-30T12:00:00Z", "ece": 0.09, "n": 60},
        ...     ]
        ... }]
        >>> report = analyze_drift_from_series(series)
        >>> len(report.candidates)
        1
    """
    if config is None:
        config = DriftConfig()

    all_candidates = []
    total_points = 0

    for series_data in calibration_series:
        key = series_data.get("key", "unknown")
        version = series_data.get("version", "v0")
        raw_points = series_data.get("points", [])

        # Convert to CalibrationPoint objects
        points = [CalibrationPoint.from_dict(p) for p in raw_points]
        total_points += len(points)

        # Detect drift for this series
        candidates = detect_drift_candidates(
            calibration_points=points,
            origin=key,
            model_version=version,
            config=config,
        )

        all_candidates.extend(candidates)

    return DriftReport(
        generated_at=datetime.now(timezone.utc),
        window_hours=72,  # Standard lookback
        grace_hours=config.grace_hours,
        min_buckets=config.min_buckets,
        ece_threshold=config.ece_threshold,
        action_mode=config.action_mode,
        candidates=all_candidates,
        total_points_analyzed=total_points,
    )


__all__ = [
    "DriftConfig",
    "CalibrationPoint",
    "DriftCandidate",
    "DriftReport",
    "parse_timestamp",
    "filter_recent_points",
    "detect_drift_candidates",
    "analyze_drift_from_series",
]
