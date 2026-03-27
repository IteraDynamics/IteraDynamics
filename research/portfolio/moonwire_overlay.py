"""
MoonWire exposure overlay for Core BTC sleeve (experiment only).

Modifies Core desired_exposure_frac after Layer 2 intent is generated.
MoonWire is an intelligence layer: it does not replace Core entry/exit logic;
it scales or filters exposure by a timestamp-aligned state (bullish/neutral/bearish).

Input: timestamp-aligned probability feed (JSONL: timestamp, probability).
State derivation: prob >= bull_thresh -> bullish; prob <= bear_thresh -> bearish; else neutral.
Variants A/B/C map state -> multiplier; final_exposure = core_desired_exposure_frac * multiplier.

Env (all optional when overlay disabled):
- MOONWIRE_OVERLAY_ENABLED: "1" to enable, else disabled (Core unchanged).
- MOONWIRE_OVERLAY_VARIANT: "A" | "B" | "C" (multiplier mapping).
- MOONWIRE_SIGNAL_FILE: path to JSONL feed (timestamp, probability).
- MOONWIRE_OVERLAY_BULL_THRESH: prob >= this -> bullish (default 0.55).
- MOONWIRE_OVERLAY_BEAR_THRESH: prob <= this -> bearish (default 0.45).
- MOONWIRE_OVERLAY_STRICT_TS: "1" = raise if bar ts missing in feed; "0" = treat as neutral (default 0).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Literal, Optional, Tuple, Union


def load_feed(filepath: Union[str, Path]) -> Dict[int, float]:
    """
    Load MoonWire signal feed from JSONL. Each line: {"timestamp": <unix_sec>, "probability": <0-1>}.
    Returns dict mapping timestamp (int, Unix seconds) -> probability (float).
    """
    feed: Dict[int, float] = {}
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            ts = int(rec["timestamp"])
            feed[ts] = float(rec["probability"])
    return feed

# Variant multiplier maps: state -> multiplier
VARIANT_A: Dict[str, float] = {"bullish": 1.00, "neutral": 0.50, "bearish": 0.00}
VARIANT_B: Dict[str, float] = {"bullish": 1.00, "neutral": 0.75, "bearish": 0.25}
VARIANT_C: Dict[str, float] = {"bullish": 1.00, "neutral": 1.00, "bearish": 0.00}

VARIANT_MAP: Dict[str, Dict[str, float]] = {
    "A": VARIANT_A,
    "B": VARIANT_B,
    "C": VARIANT_C,
}


def _ts_to_unix(bar_ts: Union[str, int, float]) -> int:
    """Convert bar timestamp (str ISO / unix sec) to unix int seconds."""
    if isinstance(bar_ts, (int, float)):
        return int(bar_ts)
    import pandas as pd
    t = pd.to_datetime(bar_ts, utc=True)
    return int(t.timestamp())


def _probability_to_state(
    prob: float,
    bull_thresh: float,
    bear_thresh: float,
) -> Literal["bullish", "neutral", "bearish"]:
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
    """
    Get MoonWire overlay multiplier for a bar timestamp.

    feed: dict mapping unix_ts (int) -> probability (float).
    Returns (multiplier, state_str, meta).
    When bar_ts not in feed: if strict_ts -> raise KeyError; else state="neutral", multiplier=variant neutral.
    """
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
    """
    Apply MoonWire overlay: final = core_desired_exposure_frac * multiplier.
    Returns (final_desired_exposure_frac, meta) for logging/trace.
    """
    mult, state, meta = get_overlay_multiplier(
        bar_ts, feed, variant=variant, bull_thresh=bull_thresh, bear_thresh=bear_thresh, strict_ts=strict_ts
    )
    final = max(0.0, min(1.0, core_desired_exposure_frac * mult))
    meta["core_desired_exposure_frac"] = core_desired_exposure_frac
    meta["final_desired_exposure_frac"] = final
    return final, meta
