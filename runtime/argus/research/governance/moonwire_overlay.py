"""
MoonWire exposure overlay for Core BTC sleeve (experiment only).
Mirrors research/portfolio/moonwire_overlay.py for backtest import (argus path).

Modifies Core desired_exposure_frac after Layer 2 intent is generated.
Input: timestamp-aligned probability feed (JSONL). Variants A/B/C map state -> multiplier.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Literal, Optional, Tuple, Union


def _ts_to_unix(bar_ts: Union[str, int, float]) -> int:
    if isinstance(bar_ts, (int, float)):
        return int(bar_ts)
    import pandas as pd
    return int(pd.to_datetime(bar_ts, utc=True).timestamp())


def load_feed(filepath: Union[str, Path]) -> Dict[int, float]:
    """Load MoonWire signal feed from JSONL. Returns dict unix_ts -> probability.
    Accepts canonical schema (timestamp ISO8601, p_long) or legacy (timestamp unix, probability).
    """
    feed: Dict[int, float] = {}
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            ts = _ts_to_unix(rec["timestamp"])
            prob = float(rec.get("p_long", rec.get("probability", 0.5)))
            feed[ts] = prob
    return feed


VARIANT_A: Dict[str, float] = {"bullish": 1.00, "neutral": 0.50, "bearish": 0.00}
VARIANT_B: Dict[str, float] = {"bullish": 1.00, "neutral": 0.75, "bearish": 0.25}
VARIANT_C: Dict[str, float] = {"bullish": 1.00, "neutral": 1.00, "bearish": 0.00}
VARIANT_MAP: Dict[str, Dict[str, float]] = {"A": VARIANT_A, "B": VARIANT_B, "C": VARIANT_C}


def _probability_to_state(prob: float, bull_thresh: float, bear_thresh: float) -> Literal["bullish", "neutral", "bearish"]:
    if prob >= bull_thresh:
        return "bullish"
    if prob <= bear_thresh:
        return "bearish"
    return "neutral"


def is_overlay_enabled() -> bool:
    return os.environ.get("MOONWIRE_OVERLAY_ENABLED", "").strip() == "1"


def get_overlay_multiplier(
    bar_ts: Union[str, int, float],
    feed: Dict[int, float],
    *,
    variant: Optional[str] = None,
    bull_thresh: Optional[float] = None,
    bear_thresh: Optional[float] = None,
    strict_ts: Optional[bool] = None,
) -> Tuple[float, str, Dict[str, Any]]:
    v = (variant or os.environ.get("MOONWIRE_OVERLAY_VARIANT", "A")).strip().upper()
    if v not in VARIANT_MAP:
        v = "A"
    mult_map = VARIANT_MAP[v]
    bull = bull_thresh if bull_thresh is not None else float(os.environ.get("MOONWIRE_OVERLAY_BULL_THRESH", "0.55"))
    bear = bear_thresh if bear_thresh is not None else float(os.environ.get("MOONWIRE_OVERLAY_BEAR_THRESH", "0.45"))
    strict = strict_ts if strict_ts is not None else (os.environ.get("MOONWIRE_OVERLAY_STRICT_TS", "0").strip() == "1")
    ts_unix = _ts_to_unix(bar_ts)
    if ts_unix not in feed:
        if strict:
            raise KeyError(f"Bar timestamp {bar_ts} (unix={ts_unix}) not in MoonWire feed (MOONWIRE_OVERLAY_STRICT_TS=1).")
        state = "neutral"
        multiplier = mult_map[state]
        meta = {"moonwire_state": state, "moonwire_multiplier": multiplier, "moonwire_ts_missing": True}
        return multiplier, state, meta
    prob = feed[ts_unix]
    state = _probability_to_state(prob, bull, bear)
    multiplier = mult_map[state]
    meta = {"moonwire_state": state, "moonwire_multiplier": multiplier, "moonwire_probability": prob}
    return multiplier, state, meta


def apply_overlay(
    core_desired_exposure_frac: float,
    bar_ts: Union[str, int, float],
    feed: Dict[int, float],
    *,
    variant: Optional[str] = None,
    bull_thresh: Optional[float] = None,
    bear_thresh: Optional[float] = None,
    strict_ts: Optional[bool] = None,
) -> Tuple[float, Dict[str, Any]]:
    mult, state, meta = get_overlay_multiplier(
        bar_ts, feed, variant=variant, bull_thresh=bull_thresh, bear_thresh=bear_thresh, strict_ts=strict_ts
    )
    final = max(0.0, min(1.0, core_desired_exposure_frac * mult))
    meta["core_desired_exposure_frac"] = core_desired_exposure_frac
    meta["final_desired_exposure_frac"] = final
    return final, meta
