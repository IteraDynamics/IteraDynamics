"""
Nowcast Attention Module

Blend multiple components per origin into a single attention score for the most recent bucket.
Components include burst z-score, volatility regime, precision, and lead-lag leadership.

This module combines insights from multiple analytics modules to provide a comprehensive
attention ranking system for prioritizing origins/sources.

Example:
    ```python
    from . import compute_nowcast_attention

    events = [
        {"timestamp": "2025-01-01T10:00:00Z", "origin": "twitter", "type": "flag"},
        {"timestamp": "2025-01-01T10:00:00Z", "origin": "twitter", "type": "trigger"},
        ...
    ]

    result = compute_nowcast_attention(
        events=events,
        days=7,
        interval="hour",
        lookback=72,
        top=10
    )

    for origin_data in result["origins"]:
        print(f"Rank {origin_data['rank']}: {origin_data['origin']} (score: {origin_data['score']})")
    ```
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, List, DefaultDict

from .origin_utils import normalize_origin, parse_timestamp
from .volatility_regimes import compute_volatility_regimes
from .threshold_policy import threshold_for_regime
from .source_metrics import compute_source_metrics
from .lead_lag import compute_lead_lag


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
    else:
        hours = days * 24
        start = now.replace(minute=0, second=0, microsecond=0) - timedelta(hours=hours - 1)
        step = timedelta(hours=1)
        count = hours

    out, cur = [], start
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
    var = sum((v - m) ** 2 for v in vals) / n
    return m, var ** 0.5


def compute_nowcast_attention(
    events: List[Dict[str, Any]],
    days: int = 7,
    interval: str = "hour",
    lookback: int = 72,
    z_cap: float = 5.0,
    top: int = 10,
    now: datetime | None = None,
    origin_field: str = "origin",
    timestamp_field: str = "timestamp",
    type_field: str = "type"
) -> Dict[str, Any]:
    """
    Blend multiple analytics components into a single attention score per origin.

    Components:
    - Burst z-score (current bucket vs window baseline, includes zeros)
    - Volatility regime (mapped to factor via threshold policy)
    - Precision prior (triggers/flags)
    - Lead-lag leadership (max |r| where origin is currently the leader)

    Args:
        events: List of event dictionaries with timestamp, origin, and type fields
        days: Number of days to look back
        interval: Time bucket size ("hour" or "day")
        lookback: Number of buckets for rolling stats (used for regimes)
        z_cap: Cap for z normalization (default: 5.0)
        top: Limit results to top N origins (default: 10)
        now: Reference time (defaults to datetime.now(timezone.utc))
        origin_field: Field name for origin/source (default: "origin")
        timestamp_field: Field name for timestamp (default: "timestamp")
        type_field: Field name for event type (default: "type", values: "flag" or "trigger")

    Returns:
        Dictionary with:
        {
          "window_days": days,
          "interval": "hour",
          "as_of": "ISO bucket",
          "origins": [
            {
              "origin": "twitter",
              "score": 87.4,
              "rank": 1,
              "components": {
                "z": 2.1, "z_norm": 0.42,
                "precision": 0.71, "leadership": 0.58,
                "regime": "turbulent", "regime_factor": 1.1, "threshold": 3.0,
                "mean": 10.5, "std": 2.3, "current": 15
              }
            },
            ...
          ]
        }

    Example:
        >>> events = [
        ...     {"timestamp": "2025-01-01T10:00:00Z", "origin": "twitter", "type": "flag"},
        ... ]
        >>> result = compute_nowcast_attention(events, days=7, interval="hour", lookback=72, top=10)
        >>> "origins" in result
        True
    """
    if days <= 0:
        return {"window_days": days, "interval": interval, "as_of": None, "origins": []}
    if interval not in ("hour", "day"):
        raise ValueError('interval must be "hour" or "day"')
    if lookback <= 1:
        raise ValueError("lookback must be >= 2")
    if z_cap <= 0:
        z_cap = 1.0

    if now is None:
        now = datetime.now(timezone.utc)

    cutoff = now - timedelta(days=days)
    buckets = _bucket_range(now, days, interval)
    last_bucket = buckets[-1]

    # Build combined counts per origin per bucket
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

    origins = sorted(per_origin.keys())
    if not origins:
        return {
            "window_days": days,
            "interval": interval,
            "as_of": last_bucket.isoformat().replace("+00:00", "Z"),
            "origins": []
        }

    # Components from other analytics
    regimes = compute_volatility_regimes(events, days=days, interval=interval, lookback=lookback,
                                        origin_field=origin_field, timestamp_field=timestamp_field)
    regime_map = {r["origin"]: r.get("regime", "normal") for r in regimes.get("origins", [])}

    metrics = compute_source_metrics(events, days=days, min_count=1,
                                    origin_field=origin_field, timestamp_field=timestamp_field, type_field=type_field)
    precision_map = {
        o["origin"]: float(o.get("precision", 0) if "precision" in o else (o.get("triggers", 0) / max(o.get("flags", 0), 1)))
        for o in metrics.get("origins", [])
    }

    ll = compute_lead_lag(events, days=days, interval=interval, max_lag=24,
                         origin_field=origin_field, timestamp_field=timestamp_field)
    leadership_map: Dict[str, float] = defaultdict(float)
    for p in ll.get("pairs", []):
        # best_lag > 0 => a leads b, <0 => b leads a
        r = abs(float(p.get("correlation", 0)))
        if p.get("best_lag", 0) > 0:
            leadership_map[p["a"]] = max(leadership_map[p["a"]], r)
        elif p.get("best_lag", 0) < 0:
            leadership_map[p["b"]] = max(leadership_map[p["b"]], r)

    # Compute z-score for the last bucket and blend components
    out_rows: List[Dict[str, Any]] = []

    for origin in origins:
        series = [float(per_origin[origin].get(b, 0)) for b in buckets]  # include zeros for baseline
        mean, std = _mean_std(series)
        current = series[-1]
        z = (current - mean) / std if std > 0 else 0.0
        z_norm = max(0.0, min(z / z_cap, 1.0))  # clamp to [0,1]

        regime = regime_map.get(origin, "normal")
        threshold = threshold_for_regime(regime)
        # Regime factor: slight emphasis on turbulent, small de-emphasis on calm
        regime_factor = {"calm": 0.95, "normal": 1.0, "turbulent": 1.05}.get(regime, 1.0)

        precision = max(0.0, min(float(precision_map.get(origin, 0.0)), 1.0))
        leadership = max(0.0, min(float(leadership_map.get(origin, 0.0)), 1.0))

        # Blend (weights sum to 1.0)
        w_z, w_prec, w_lead, w_reg = 0.5, 0.25, 0.15, 0.10
        base = (w_z * z_norm) + (w_prec * precision) + (w_lead * leadership) + (w_reg * (regime_factor - 0.9) / 0.2)
        score = round(100.0 * base, 2)

        out_rows.append({
            "origin": origin,
            "score": score,
            "components": {
                "z": round(float(z), 3), "z_norm": round(float(z_norm), 3),
                "precision": round(precision, 3), "leadership": round(leadership, 3),
                "regime": regime, "regime_factor": regime_factor, "threshold": threshold,
                "mean": round(mean, 3), "std": round(std, 3), "current": int(current),
            }
        })

    # Rank and trim
    out_rows.sort(key=lambda r: r["score"], reverse=True)
    for i, r in enumerate(out_rows, 1):
        r["rank"] = i
    if top and top > 0:
        out_rows = out_rows[:top]

    return {
        "window_days": days,
        "interval": interval,
        "as_of": last_bucket.replace(tzinfo=timezone.utc).isoformat().replace("+00:00", "Z"),
        "origins": out_rows
    }


__all__ = ["compute_nowcast_attention"]
