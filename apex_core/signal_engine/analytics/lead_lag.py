"""
Lead-Lag Analysis Module

Compute lead/lag relationships between origins using cross-correlation.
Identifies which origins tend to spike before others.

Example:
    ```python
    from . import compute_lead_lag

    events = [
        {"timestamp": "2025-01-01T10:00:00Z", "origin": "twitter"},
        {"timestamp": "2025-01-01T11:00:00Z", "origin": "reddit"},
        {"timestamp": "2025-01-01T12:00:00Z", "origin": "twitter"},
        ...
    ]

    result = compute_lead_lag(
        events=events,
        days=7,
        interval="hour",
        max_lag=24
    )

    for pair in result["pairs"]:
        if pair["leader"] != "tie":
            print(f"{pair['leader']} leads by {abs(pair['best_lag'])} {result['interval']}s")
    ```
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timedelta, timezone
from itertools import permutations
from math import sqrt
from typing import Dict, Any, List, Tuple, DefaultDict

from .origin_utils import normalize_origin, parse_timestamp


def _bucket_start(dt: datetime, interval: str) -> datetime:
    """Truncate datetime to bucket start based on interval."""
    if interval == "day":
        return dt.replace(hour=0, minute=0, second=0, microsecond=0)
    elif interval == "hour":
        return dt.replace(minute=0, second=0, microsecond=0)
    raise ValueError('interval must be "hour" or "day"')


def _pearson(x: List[float], y: List[float]) -> float:
    """Compute Pearson correlation coefficient."""
    n = len(x)
    if n < 3:
        return 0.0
    mx = sum(x) / n
    my = sum(y) / n
    sxx = sum((xi - mx) ** 2 for xi in x)
    syy = sum((yi - my) ** 2 for yi in y)
    if sxx == 0.0 or syy == 0.0:
        return 0.0
    sxy = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y))
    return sxy / (sqrt(sxx) * sqrt(syy))


def _aligned_vectors(
    a_map: Dict[datetime, int],
    b_map: Dict[datetime, int],
    lag: int,
    step: timedelta
) -> Tuple[List[float], List[float]]:
    """
    For lag L: compare A[t] with B[t+L*step]. Only keep timestamps where both exist.
    """
    if lag == 0:
        common = sorted(set(a_map.keys()) & set(b_map.keys()))
        return [float(a_map[t]) for t in common], [float(b_map[t]) for t in common]

    out_x: List[float] = []
    out_y: List[float] = []
    for t in a_map.keys():
        t2 = t + lag * step
        if t2 in b_map:
            out_x.append(float(a_map[t]))
            out_y.append(float(b_map[t2]))
    return out_x, out_y


def compute_lead_lag(
    events: List[Dict[str, Any]],
    days: int = 7,
    interval: str = "hour",
    max_lag: int = 24,
    top_n: int = 20,
    now: datetime | None = None,
    origin_field: str = "origin",
    timestamp_field: str = "timestamp"
) -> Dict[str, Any]:
    """
    Compute lead/lag relationships across ordered origin pairs using cross-correlation.

    A positive best_lag means A leads B by best_lag intervals.
    A negative best_lag means B leads A by |best_lag| intervals.

    Args:
        events: List of event dictionaries with timestamp and origin fields
        days: Number of days to look back
        interval: Time bucket size ("hour" or "day")
        max_lag: Maximum number of intervals to test for lag
        top_n: Limit to top N origins by activity (default: 20)
        now: Reference time (defaults to datetime.now(timezone.utc))
        origin_field: Field name for origin/source (default: "origin")
        timestamp_field: Field name for timestamp (default: "timestamp")

    Returns:
        Dictionary with:
        {
          "window_days": 7,
          "interval": "hour",
          "max_lag": 24,
          "origins": ["twitter", "reddit", ...],
          "pairs": [
            {"a": "twitter", "b": "reddit", "best_lag": 3, "correlation": 0.75, "leader": "twitter"},
            ...
          ]
        }

    Example:
        >>> events = [
        ...     {"timestamp": "2025-01-01T10:00:00Z", "origin": "twitter"},
        ...     {"timestamp": "2025-01-01T11:00:00Z", "origin": "reddit"},
        ... ]
        >>> result = compute_lead_lag(events, days=7, interval="hour", max_lag=24)
        >>> "pairs" in result
        True
    """
    if interval not in ("hour", "day"):
        raise ValueError('interval must be "hour" or "day"')
    if days <= 0:
        return {"window_days": days, "interval": interval, "max_lag": max_lag, "origins": [], "pairs": []}
    if max_lag < 0:
        max_lag = 0

    if now is None:
        now = datetime.now(timezone.utc)

    cutoff = now - timedelta(days=days)

    per_origin: DefaultDict[str, Dict[datetime, int]] = defaultdict(lambda: defaultdict(int))
    totals: DefaultDict[str, int] = defaultdict(int)

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
        per_origin[origin][bts] += 1
        totals[origin] += 1

    if not per_origin:
        return {"window_days": days, "interval": interval, "max_lag": max_lag, "origins": [], "pairs": []}

    # Limit to top N origins by total activity
    origins_sorted = sorted(totals.keys(), key=lambda o: (-totals[o], o))[:top_n]

    step = timedelta(hours=1) if interval == "hour" else timedelta(days=1)

    pairs: List[Dict[str, Any]] = []
    for a, b in permutations(origins_sorted, 2):  # ordered pairs
        a_map = per_origin.get(a, {})
        b_map = per_origin.get(b, {})
        if not a_map or not b_map:
            continue

        best_r = 0.0
        best_L = None

        for L in range(-max_lag, max_lag + 1):
            x, y = _aligned_vectors(a_map, b_map, L, step)
            if len(x) < 3:
                continue
            r = _pearson(x, y)
            if abs(r) > abs(best_r):
                best_r = r
                best_L = L

        if best_L is None or best_r == 0.0:
            continue

        leader = a if best_L > 0 else (b if best_L < 0 else "tie")
        pairs.append({
            "a": a,
            "b": b,
            "best_lag": int(best_L),
            "correlation": round(float(best_r), 3),
            "leader": leader
        })

    # Sort by |correlation| desc, then by names
    pairs.sort(key=lambda p: (-abs(p["correlation"]), p["a"], p["b"]))

    return {
        "window_days": days,
        "interval": interval,
        "max_lag": max_lag,
        "origins": origins_sorted,
        "pairs": pairs
    }


__all__ = ["compute_lead_lag"]
