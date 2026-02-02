"""
Volatility Regimes Module

Classify origin activity into volatility regimes (calm, normal, turbulent) based on
rolling standard deviation. Useful for adaptive threshold policies.

Example:
    ```python
    from . import compute_volatility_regimes

    events = [
        {"timestamp": "2025-01-01T10:00:00Z", "origin": "twitter"},
        {"timestamp": "2025-01-01T11:00:00Z", "origin": "twitter"},
        {"timestamp": "2025-01-01T12:00:00Z", "origin": "reddit"},
        ...
    ]

    result = compute_volatility_regimes(
        events=events,
        days=30,
        interval="hour",
        lookback=72
    )

    for origin_data in result["origins"]:
        print(f"{origin_data['origin']}: {origin_data['regime']} regime")
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
    """Generate forward-ordered, full-window bucket starts (includes zeros for true baseline)."""
    if interval == "day":
        start = (now - timedelta(days=days - 1)).replace(hour=0, minute=0, second=0, microsecond=0)
        step = timedelta(days=1)
        count = days
    else:
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


def _mean_std(vals: List[float]) -> tuple[float, float]:
    """Compute mean and population standard deviation."""
    n = len(vals)
    if n == 0:
        return 0.0, 0.0
    m = sum(vals) / n
    var = sum((v - m) ** 2 for v in vals) / n  # population std
    return m, sqrt(var)


def _percentile(sorted_vals: List[float], q: float) -> float:
    """Compute percentile using nearest-rank method. Safe for small n."""
    n = len(sorted_vals)
    if n == 0:
        return 0.0
    if q <= 0:
        return sorted_vals[0]
    if q >= 1:
        return sorted_vals[-1]
    # index in [0, n-1]
    idx = int(round(q * (n - 1)))
    return sorted_vals[idx]


def compute_volatility_regimes(
    events: List[Dict[str, Any]],
    days: int = 30,
    interval: str = "hour",
    lookback: int = 72,
    q_calm: float = 0.33,
    q_turb: float = 0.80,
    now: datetime | None = None,
    origin_field: str = "origin",
    timestamp_field: str = "timestamp"
) -> Dict[str, Any]:
    """
    Classify origins into volatility regimes based on rolling standard deviation.

    Builds per-origin counts, computes current rolling std over the last `lookback`
    buckets as vol_metric, then labels each origin via cross-origin quantiles:
    - <= q_calm => calm
    - >= q_turb => turbulent
    - else => normal

    Args:
        events: List of event dictionaries with timestamp and origin fields
        days: Number of days to look back
        interval: Time bucket size ("hour" or "day")
        lookback: Number of buckets for rolling stats (default: 72)
        q_calm: Quantile threshold for calm regime (default: 0.33)
        q_turb: Quantile threshold for turbulent regime (default: 0.80)
        now: Reference time (defaults to datetime.now(timezone.utc))
        origin_field: Field name for origin/source (default: "origin")
        timestamp_field: Field name for timestamp (default: "timestamp")

    Returns:
        Dictionary with:
        {
          "window_days": 30,
          "interval": "hour",
          "origins": [
            {
              "origin": "twitter",
              "vol_metric": 2.45,
              "regime": "turbulent",
              "stats": {"mean": 10.5, "std": 2.45}
            },
            ...
          ]
        }

    Example:
        >>> events = [
        ...     {"timestamp": "2025-01-01T10:00:00Z", "origin": "twitter"},
        ...     {"timestamp": "2025-01-01T11:00:00Z", "origin": "twitter"},
        ... ]
        >>> result = compute_volatility_regimes(events, days=30, interval="hour", lookback=72)
        >>> len(result["origins"]) >= 0
        True
    """
    if days <= 0:
        return {"window_days": days, "interval": interval, "origins": []}
    if interval not in ("hour", "day"):
        raise ValueError('interval must be "hour" or "day"')
    if lookback <= 1:
        raise ValueError("lookback must be >= 2")
    if not (0 <= q_calm <= 1 and 0 <= q_turb <= 1):
        raise ValueError("quantiles must be in [0,1]")

    if now is None:
        now = datetime.now(timezone.utc)

    cutoff = now - timedelta(days=days)
    buckets = _bucket_range(now, days, interval)

    # (origin -> bucket -> count), combining all events
    per_origin: DefaultDict[str, Dict[datetime, int]] = defaultdict(lambda: defaultdict(int))

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

    # Compute current rolling stats per origin
    results: List[Dict[str, Any]] = []
    vol_values: List[float] = []

    for origin in sorted(per_origin.keys()):
        series = [float(per_origin[origin].get(b, 0)) for b in buckets]
        window = series[-lookback:] if len(series) >= lookback else series
        mean, std = _mean_std(window)
        vol_metric = std  # rolling std choice for simplicity & determinism
        vol_values.append(vol_metric)
        results.append({
            "origin": origin,
            "vol_metric": round(vol_metric, 6),
            "regime": "normal",  # to be assigned
            "stats": {"mean": round(mean, 6), "std": round(std, 6)},
        })

    # Label regimes via cross-origin quantiles of current vol_metric
    vol_sorted = sorted(vol_values)
    calm_cut = _percentile(vol_sorted, q_calm)
    turb_cut = _percentile(vol_sorted, q_turb)

    for row in results:
        v = row["vol_metric"]
        if v <= calm_cut:
            row["regime"] = "calm"
        elif v >= turb_cut:
            row["regime"] = "turbulent"
        else:
            row["regime"] = "normal"

    return {"window_days": days, "interval": interval, "origins": results}


__all__ = ["compute_volatility_regimes"]
