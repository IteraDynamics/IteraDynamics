"""
Burst Detection Module

Detect bursty time periods using z-score analysis against baseline activity levels.
Useful for identifying sudden spikes in event activity per origin/source.

Example:
    ```python
    from . import compute_bursts
    from datetime import datetime, timezone

    events = [
        {"timestamp": "2025-01-01T10:00:00Z", "origin": "twitter"},
        {"timestamp": "2025-01-01T10:00:00Z", "origin": "twitter"},
        {"timestamp": "2025-01-01T10:00:00Z", "origin": "twitter"},  # Burst!
        {"timestamp": "2025-01-01T11:00:00Z", "origin": "reddit"},
        ...
    ]

    result = compute_bursts(
        events=events,
        days=7,
        interval="hour",
        z_thresh=2.0
    )

    for origin_data in result["origins"]:
        if origin_data["bursts"]:
            print(f"{origin_data['origin']} has {len(origin_data['bursts'])} bursts")
    ```
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timedelta, timezone
from math import sqrt
from typing import Dict, Any, List, DefaultDict

from .origin_utils import normalize_origin, parse_timestamp


def _bucket_start(dt: datetime, interval: str) -> datetime:
    """Truncate datetime to bucket start based on interval."""
    if interval == "day":
        return dt.replace(hour=0, minute=0, second=0, microsecond=0)
    elif interval == "hour":
        return dt.replace(minute=0, second=0, microsecond=0)
    else:
        raise ValueError('interval must be "hour" or "day"')


def _bucket_range(now: datetime, days: int, interval: str) -> List[datetime]:
    """Generate forward-ordered list of bucket starts covering the full window."""
    if interval == "day":
        start = (now - timedelta(days=days - 1)).replace(hour=0, minute=0, second=0, microsecond=0)
        step = timedelta(days=1)
        count = days
    else:  # hour
        hours = days * 24
        start = now.replace(minute=0, second=0, microsecond=0) - timedelta(hours=hours - 1)
        step = timedelta(hours=1)
        count = hours

    out = []
    cur = start
    for _ in range(count):
        out.append(cur)
        cur = cur + step
    return out


def _mean_std(values: List[float]) -> tuple[float, float]:
    """Compute mean and population standard deviation."""
    n = len(values)
    if n == 0:
        return 0.0, 0.0
    m = sum(values) / n
    var = sum((v - m) ** 2 for v in values) / n  # population std
    return m, sqrt(var)


def compute_bursts(
    events: List[Dict[str, Any]],
    days: int = 7,
    interval: str = "hour",
    z_thresh: float = 2.0,
    now: datetime | None = None,
    origin_field: str = "origin",
    timestamp_field: str = "timestamp"
) -> Dict[str, Any]:
    """
    Detect bursty buckets per origin using z-score against the window baseline.

    Args:
        events: List of event dictionaries with timestamp and origin fields
        days: Number of days to look back
        interval: Time bucket size ("hour" or "day")
        z_thresh: Z-score threshold for burst detection (default: 2.0)
        now: Reference time (defaults to datetime.now(timezone.utc))
        origin_field: Field name for origin/source (default: "origin")
        timestamp_field: Field name for timestamp (default: "timestamp")

    Returns:
        Dictionary with:
        {
          "window_days": 7,
          "interval": "hour",
          "origins": [
            {
              "origin": "twitter",
              "bursts": [
                {"timestamp_bucket": "2025-01-01T10:00:00Z", "count": 42, "z_score": 3.1},
                ...
              ]
            },
            ...
          ]
        }

    Example:
        >>> from datetime import datetime, timezone
        >>> events = [
        ...     {"timestamp": "2025-01-01T10:00:00Z", "origin": "twitter"},
        ...     {"timestamp": "2025-01-01T10:00:00Z", "origin": "twitter"},
        ...     {"timestamp": "2025-01-01T10:00:00Z", "origin": "twitter"},
        ... ]
        >>> result = compute_bursts(events, days=7, interval="hour", z_thresh=2.0)
        >>> len(result["origins"]) > 0
        True
    """
    if days <= 0:
        return {"window_days": days, "interval": interval, "origins": []}
    if interval not in ("hour", "day"):
        raise ValueError('interval must be "hour" or "day"')
    if z_thresh < 0:
        z_thresh = 0.0

    if now is None:
        now = datetime.now(timezone.utc)

    cutoff = now - timedelta(days=days)
    buckets = _bucket_range(now, days, interval)

    # (origin -> bucket -> count)
    per_origin: DefaultDict[str, Dict[datetime, int]] = defaultdict(lambda: defaultdict(int))

    # Process events
    for event in events:
        ts_str = event.get(timestamp_field)
        if not ts_str:
            continue

        ts = parse_timestamp(ts_str)
        if ts is None or ts < cutoff:
            continue

        bts = _bucket_start(ts, interval)
        origin = normalize_origin(event.get(origin_field) or event.get("source") or
                                 event.get("meta", {}).get("origin") or
                                 event.get("metadata", {}).get("source"))
        per_origin[origin][bts] += 1

    # Build burst lists
    origins_out: List[Dict[str, Any]] = []
    for origin in sorted(per_origin.keys()):
        # Include zeros for real baseline
        series = [float(per_origin[origin].get(b, 0)) for b in buckets]
        mean, std = _mean_std(series)

        if std == 0.0:
            bursts = []
        else:
            bursts = []
            for b in buckets:
                c = per_origin[origin].get(b, 0)
                z = (c - mean) / std if std > 0 else 0.0
                if z >= z_thresh:
                    bursts.append({
                        "timestamp_bucket": b.replace(tzinfo=timezone.utc).isoformat().replace("+00:00", "Z"),
                        "count": int(c),
                        "z_score": round(float(z), 3)
                    })

        origins_out.append({"origin": origin, "bursts": bursts})

    return {"window_days": days, "interval": interval, "origins": origins_out}


__all__ = ["compute_bursts"]
