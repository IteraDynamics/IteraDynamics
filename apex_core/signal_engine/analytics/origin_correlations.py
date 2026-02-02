"""
Origin Correlations Module

Compute pairwise Pearson correlations between origin time-series activity patterns.
Useful for understanding which origins tend to spike together.

Example:
    ```python
    from . import compute_origin_correlations

    events = [
        {"timestamp": "2025-01-01T10:00:00Z", "origin": "twitter"},
        {"timestamp": "2025-01-01T10:00:00Z", "origin": "reddit"},
        {"timestamp": "2025-01-01T11:00:00Z", "origin": "twitter"},
        {"timestamp": "2025-01-01T11:00:00Z", "origin": "reddit"},
        ...
    ]

    result = compute_origin_correlations(
        events=events,
        days=7,
        interval="day"
    )

    for pair in result["pairs"]:
        print(f"{pair['a']} <-> {pair['b']}: {pair['correlation']:.2f}")
    ```
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timedelta, timezone
from itertools import combinations
from math import sqrt
from typing import Dict, Any, List, Tuple, DefaultDict

from .origin_utils import normalize_origin, parse_timestamp


def _bucket_start(dt: datetime, interval: str) -> datetime:
    """Truncate datetime to bucket start based on interval."""
    if interval == "day":
        return dt.replace(hour=0, minute=0, second=0, microsecond=0)
    elif interval == "hour":
        return dt.replace(minute=0, second=0, microsecond=0)
    else:
        raise ValueError('interval must be "day" or "hour"')


def _pearson(x: List[float], y: List[float]) -> float:
    """Compute Pearson correlation coefficient between two series."""
    n = len(x)
    if n < 2:
        return 0.0

    mx = sum(x) / n
    my = sum(y) / n
    sxx = sum((xi - mx) ** 2 for xi in x)
    syy = sum((yi - my) ** 2 for yi in y)

    if sxx == 0.0 or syy == 0.0:
        # Constant series -> undefined correlation; treat as 0
        return 0.0

    sxy = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y))
    return sxy / (sqrt(sxx) * sqrt(syy))


def compute_origin_correlations(
    events: List[Dict[str, Any]],
    days: int = 7,
    interval: str = "day",
    now: datetime | None = None,
    origin_field: str = "origin",
    timestamp_field: str = "timestamp"
) -> Dict[str, Any]:
    """
    Compute pairwise Pearson correlations between origin activity time-series.

    Buckets events by time (day|hour), builds aligned time series per origin
    over observed buckets only, then computes pairwise Pearson correlations.

    Args:
        events: List of event dictionaries with timestamp and origin fields
        days: Number of days to look back
        interval: Time bucket size ("day" or "hour")
        now: Reference time (defaults to datetime.now(timezone.utc))
        origin_field: Field name for origin/source (default: "origin")
        timestamp_field: Field name for timestamp (default: "timestamp")

    Returns:
        Dictionary with:
        {
          "window_days": 7,
          "interval": "day",
          "origins": ["reddit", "rss_news", "twitter"],
          "pairs": [
            {"a": "reddit", "b": "twitter", "correlation": 0.82},
            ...
          ]
        }

    Example:
        >>> events = [
        ...     {"timestamp": "2025-01-01T10:00:00Z", "origin": "twitter"},
        ...     {"timestamp": "2025-01-01T10:00:00Z", "origin": "reddit"},
        ... ]
        >>> result = compute_origin_correlations(events, days=7, interval="day")
        >>> "origins" in result and "pairs" in result
        True
    """
    if days <= 0:
        return {"window_days": days, "interval": interval, "origins": [], "pairs": []}
    if interval not in ("day", "hour"):
        raise ValueError('interval must be "day" or "hour"')

    if now is None:
        now = datetime.now(timezone.utc)

    cutoff = now - timedelta(days=days)

    # Collect counts and the set of actually observed buckets in-window
    counts: DefaultDict[Tuple[str, datetime], int] = defaultdict(int)
    origins_set: set[str] = set()
    observed_buckets: set[datetime] = set()

    for event in events:
        ts_str = event.get(timestamp_field)
        if not ts_str:
            continue

        ts = parse_timestamp(ts_str)
        if ts is None or ts < cutoff:
            continue

        bts = _bucket_start(ts, interval)
        origin = normalize_origin(
            event.get(origin_field) or event.get("source") or
            event.get("meta", {}).get("origin") or
            event.get("metadata", {}).get("source")
        )

        counts[(origin, bts)] += 1
        origins_set.add(origin)
        observed_buckets.add(bts)

    # No data → no pairs
    if not origins_set or not observed_buckets:
        return {"window_days": days, "interval": interval, "origins": [], "pairs": []}

    buckets_sorted = sorted(observed_buckets)
    origins_sorted = sorted(origins_set)

    # Build aligned vectors per origin across observed buckets only; skip all-zero series
    vectors: Dict[str, List[float]] = {}
    for o in origins_sorted:
        vec = [float(counts.get((o, b), 0)) for b in buckets_sorted]
        if any(v != 0.0 for v in vec):
            vectors[o] = vec

    origins = sorted(vectors.keys())
    pairs: List[Dict[str, Any]] = []
    for a, b in combinations(origins, 2):
        r = _pearson(vectors[a], vectors[b])
        pairs.append({"a": a, "b": b, "correlation": round(r, 3)})

    # Sort strongest first
    pairs.sort(key=lambda p: p["correlation"], reverse=True)

    return {"window_days": days, "interval": interval, "origins": origins, "pairs": pairs}


__all__ = ["compute_origin_correlations"]
