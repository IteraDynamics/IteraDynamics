"""
Source Metrics Module

Compute source quality metrics including flags, triggers, precision, and recall per origin.
Useful for evaluating which sources generate high-quality signals.

Example:
    ```python
    from . import compute_source_metrics

    events = [
        {"timestamp": "2025-01-01T10:00:00Z", "origin": "twitter", "type": "flag"},
        {"timestamp": "2025-01-01T10:00:00Z", "origin": "twitter", "type": "trigger"},
        {"timestamp": "2025-01-01T11:00:00Z", "origin": "reddit", "type": "flag"},
        ...
    ]

    result = compute_source_metrics(
        events=events,
        days=7,
        min_count=1
    )

    for origin in result["origins"]:
        print(f"{origin['origin']}: precision={origin['precision']:.2f}, recall={origin['recall']:.2f}")
    ```
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Any

from .origin_utils import normalize_origin, parse_timestamp


def compute_source_metrics(
    events: List[Dict[str, Any]],
    days: int = 7,
    min_count: int = 1,
    now: datetime | None = None,
    origin_field: str = "origin",
    timestamp_field: str = "timestamp",
    type_field: str = "type"
) -> Dict[str, Any]:
    """
    Compute source quality metrics per origin.

    Args:
        events: List of event dictionaries with timestamp, origin, and type fields
        days: Number of days to look back
        min_count: Minimum total events for an origin to be included
        now: Reference time (defaults to datetime.now(timezone.utc))
        origin_field: Field name for origin/source (default: "origin")
        timestamp_field: Field name for timestamp (default: "timestamp")
        type_field: Field name for event type (default: "type", values: "flag" or "trigger")

    Returns:
        Dictionary with:
        {
          "window_days": 7,
          "total_triggers": 42,
          "origins": [
            {
              "origin": "twitter",
              "flags": 100,
              "triggers": 30,
              "total": 130,
              "precision": 0.30,  # triggers / flags
              "recall": 0.71      # triggers / total_triggers
            },
            ...
          ]
        }

    Example:
        >>> events = [
        ...     {"timestamp": "2025-01-01T10:00:00Z", "origin": "twitter", "type": "flag"},
        ...     {"timestamp": "2025-01-01T10:00:00Z", "origin": "twitter", "type": "trigger"},
        ... ]
        >>> result = compute_source_metrics(events, days=7, min_count=1)
        >>> result["total_triggers"] >= 0
        True
    """
    if now is None:
        now = datetime.now(timezone.utc)

    cutoff = now - timedelta(days=days)

    flags_by_origin = defaultdict(int)
    triggers_by_origin = defaultdict(int)

    # Process events
    for event in events:
        ts_str = event.get(timestamp_field)
        if not ts_str:
            continue

        ts = parse_timestamp(ts_str)
        if ts is None or ts < cutoff:
            continue

        origin = normalize_origin(
            event.get(origin_field) or event.get("source") or
            event.get("meta", {}).get("origin") or
            event.get("metadata", {}).get("source")
        )

        event_type = event.get(type_field, "flag")
        if event_type == "trigger":
            triggers_by_origin[origin] += 1
        else:
            flags_by_origin[origin] += 1

    all_origins = set(flags_by_origin) | set(triggers_by_origin)
    total_triggers = sum(triggers_by_origin.values())
    origins = []

    for origin in sorted(all_origins):
        flags = flags_by_origin.get(origin, 0)
        triggers = triggers_by_origin.get(origin, 0)
        total = flags + triggers
        if total < min_count:
            continue

        precision = round(triggers / flags, 2) if flags > 0 else 0.0
        recall = round(triggers / total_triggers, 2) if total_triggers > 0 else 0.0

        origins.append({
            "origin": origin,
            "flags": flags,
            "triggers": triggers,
            "total": total,
            "precision": precision,
            "recall": recall
        })

    # Sort: total desc, then origin asc
    origins.sort(key=lambda x: (-x["total"], x["origin"]))

    return {
        "window_days": days,
        "total_triggers": total_triggers,
        "origins": origins
    }


__all__ = ["compute_source_metrics"]
