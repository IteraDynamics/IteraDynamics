"""
Source Yield Module

Compute source yield scores blending trigger rate and volume share.
Useful for resource allocation and budget planning across sources.

Example:
    ```python
    from . import compute_source_yield

    events = [
        {"timestamp": "2025-01-01T10:00:00Z", "origin": "twitter", "type": "flag"},
        {"timestamp": "2025-01-01T10:00:00Z", "origin": "twitter", "type": "trigger"},
        {"timestamp": "2025-01-01T11:00:00Z", "origin": "reddit", "type": "flag"},
        ...
    ]

    result = compute_source_yield(
        events=events,
        days=7,
        min_events=5,
        alpha=0.7
    )

    for origin in result["origins"]:
        print(f"{origin['origin']}: yield_score={origin['yield_score']:.3f}")

    for plan in result["budget_plan"]:
        print(f"{plan['origin']}: {plan['pct']}% of budget")
    ```
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Any

from .origin_utils import normalize_origin, parse_timestamp


def compute_source_yield(
    events: List[Dict[str, Any]],
    days: int = 7,
    min_events: int = 5,
    alpha: float = 0.7,
    now: datetime | None = None,
    origin_field: str = "origin",
    timestamp_field: str = "timestamp",
    type_field: str = "type"
) -> Dict[str, Any]:
    """
    Compute source yield scores and budget allocation plan.

    Yield score blends:
    - Trigger rate (triggers / flags): Measures conversion quality
    - Volume share (flags / total_flags): Measures activity contribution

    Formula: yield_score = alpha * trigger_rate + (1-alpha) * volume_share

    Args:
        events: List of event dictionaries with timestamp, origin, and type fields
        days: Number of days to look back
        min_events: Minimum flags for an origin to be eligible for budget plan
        alpha: Weight for trigger rate (0-1), remaining weight for volume share (default: 0.7)
        now: Reference time (defaults to datetime.now(timezone.utc))
        origin_field: Field name for origin/source (default: "origin")
        timestamp_field: Field name for timestamp (default: "timestamp")
        type_field: Field name for event type (default: "type", values: "flag" or "trigger")

    Returns:
        Dictionary with:
        {
          "window_days": 7,
          "totals": {"flags": 500, "triggers": 150},
          "origins": [
            {
              "origin": "twitter",
              "flags": 200,
              "triggers": 80,
              "trigger_rate": 0.4,
              "yield_score": 0.38,
              "eligible": true
            },
            ...
          ],
          "budget_plan": [
            {"origin": "twitter", "pct": 35.2},
            ...
          ]
        }

    Example:
        >>> events = [
        ...     {"timestamp": "2025-01-01T10:00:00Z", "origin": "twitter", "type": "flag"},
        ...     {"timestamp": "2025-01-01T10:00:00Z", "origin": "twitter", "type": "trigger"},
        ... ]
        >>> result = compute_source_yield(events, days=7, min_events=1, alpha=0.7)
        >>> "budget_plan" in result
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

    total_flags = sum(flags_by_origin.values())

    # Build per-origin stats
    origins = []
    for origin, flags in flags_by_origin.items():
        triggers = triggers_by_origin.get(origin, 0)
        trigger_rate = triggers / max(flags, 1)
        volume_share = flags / max(total_flags, 1)
        yield_score = round(alpha * trigger_rate + (1 - alpha) * volume_share, 6)

        origins.append({
            "origin": origin,
            "flags": flags,
            "triggers": triggers,
            "trigger_rate": round(trigger_rate, 6),
            "yield_score": yield_score,
            "eligible": flags >= min_events
        })

    # Compute budget plan
    eligible = [o for o in origins if o["eligible"]]
    total_yield = sum(o["yield_score"] for o in eligible) or 1
    budget_plan = [
        {
            "origin": o["origin"],
            "pct": round(100 * o["yield_score"] / total_yield, 1)
        }
        for o in sorted(eligible, key=lambda o: o["yield_score"], reverse=True)
    ]

    return {
        "window_days": days,
        "totals": {
            "flags": total_flags,
            "triggers": sum(triggers_by_origin.values())
        },
        "origins": origins,
        "budget_plan": budget_plan,
        "notes": [
            f"Origins with < {min_events} events are excluded from budget_plan.",
            f"yield_score blends conversion (trigger_rate) and volume."
        ]
    }


__all__ = ["compute_source_yield"]
