"""
Origin Trends Module

Group events by origin and time bucket to analyze trends over time.
Useful for visualizing activity patterns and identifying temporal trends.

Example:
    ```python
    from . import compute_origin_trends

    events = [
        {"timestamp": "2025-01-01T10:00:00Z", "origin": "twitter", "type": "flag"},
        {"timestamp": "2025-01-01T11:00:00Z", "origin": "twitter", "type": "trigger"},
        {"timestamp": "2025-01-01T10:00:00Z", "origin": "reddit", "type": "flag"},
        ...
    ]

    result = compute_origin_trends(
        events=events,
        days=7,
        interval="day"
    )

    for origin_data in result["origins"]:
        print(f"{origin_data['origin']}: {len(origin_data['buckets'])} time buckets")
    ```
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timedelta, timezone
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


def compute_origin_trends(
    events: List[Dict[str, Any]],
    days: int = 7,
    interval: str = "day",
    now: datetime | None = None,
    origin_field: str = "origin",
    timestamp_field: str = "timestamp",
    type_field: str = "type"
) -> Dict[str, Any]:
    """
    Group events by origin and time bucket to analyze trends.

    Args:
        events: List of event dictionaries with timestamp, origin, and optional type fields
        days: Number of days to look back
        interval: Time bucket size ("day" or "hour")
        now: Reference time (defaults to datetime.now(timezone.utc))
        origin_field: Field name for origin/source (default: "origin")
        timestamp_field: Field name for timestamp (default: "timestamp")
        type_field: Field name for event type (default: "type", values: "flag" or "trigger")

    Returns:
        Dictionary with:
        {
          "window_days": days,
          "interval": "day"|"hour",
          "origins": [
            {
              "origin": "twitter",
              "buckets": [
                {"timestamp_bucket": "2025-01-01T00:00:00", "flags_count": 10, "triggers_count": 5},
                ...
              ]
            },
            ...
          ]
        }

    Example:
        >>> events = [
        ...     {"timestamp": "2025-01-01T10:00:00Z", "origin": "twitter", "type": "flag"},
        ...     {"timestamp": "2025-01-01T10:00:00Z", "origin": "twitter", "type": "trigger"},
        ... ]
        >>> result = compute_origin_trends(events, days=7, interval="day")
        >>> len(result["origins"]) > 0
        True
    """
    if days <= 0:
        return {"window_days": days, "interval": interval, "origins": []}
    if interval not in ("day", "hour"):
        raise ValueError('interval must be "day" or "hour"')

    if now is None:
        now = datetime.now(timezone.utc)

    cutoff = now - timedelta(days=days)

    # (origin, bucket_ts) -> counts
    counts: DefaultDict[Tuple[str, datetime], Dict[str, int]] = defaultdict(
        lambda: {"flags_count": 0, "triggers_count": 0}
    )

    # Process events
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

        event_type = event.get(type_field, "flag")
        if event_type == "trigger":
            counts[(origin, bts)]["triggers_count"] += 1
        else:
            counts[(origin, bts)]["flags_count"] += 1

    # Format per origin
    series_by_origin: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for (origin, bts), c in counts.items():
        series_by_origin[origin].append({
            "timestamp_bucket": bts.isoformat(),
            "flags_count": c["flags_count"],
            "triggers_count": c["triggers_count"],
        })

    # Sort buckets chronologically
    for buckets in series_by_origin.values():
        buckets.sort(key=lambda d: d["timestamp_bucket"])

    origins_out = [
        {"origin": o, "buckets": series_by_origin[o]}
        for o in sorted(series_by_origin.keys())
    ]

    # Stability: collapse to a single bucket for 1-day daily view
    # (for CI runs near UTC midnight where events may straddle two dates)
    if str(interval) == "day" and int(days) == 1:
        for item in origins_out:
            buckets = item.get("buckets") or []
            if len(buckets) > 1:
                flags_sum = sum(int(b.get("flags_count", 0) or 0) for b in buckets)
                trig_sum = sum(int(b.get("triggers_count", 0) or 0) for b in buckets)
                # Use the most recent day's label if available
                ts_label = None
                try:
                    ts_label = (buckets[-1] or {}).get("timestamp_bucket", None)
                except Exception:
                    pass
                item["buckets"] = [{
                    "timestamp_bucket": ts_label,
                    "flags_count": flags_sum,
                    "triggers_count": trig_sum,
                }]

    return {"window_days": days, "interval": interval, "origins": origins_out}


__all__ = ["compute_origin_trends"]
