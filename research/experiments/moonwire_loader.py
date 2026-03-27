"""
MoonWire signal loader for portfolio geometry (Experiment 1 — BTC_MOONWIRE_ONLY).

Deterministic: reads MOONWIRE_SIGNAL_FILE JSONL (timestamp, probability), produces
per-bar weights aligned to the BTC merged timeline. No dependency on sg_moonwire_intent_v1.

Env vars (os.environ):
- MOONWIRE_SIGNAL_FILE (required when running MoonWire scenario)
- MOONWIRE_LONG_THRESH (default 0.65)
- MOONWIRE_ALLOW_SHORT (default 0)
- MOONWIRE_MAX_EXPOSURE (default 0.985)
- MOONWIRE_STRICT_TS_MATCH (default 1): 1 = raise if bar ts missing; 0 = HOLD (w_btc=0, prob=NaN)

Weight rule:
- allow_short=0: prob >= long_thresh -> w_btc = max_exposure, else w_btc = 0
- allow_short=1: prob >= long_thresh -> w_btc = +max_exposure;
                 prob <= (1 - long_thresh) -> w_btc = -max_exposure;
                 else w_btc = 0
- w_cash = 1 - abs(w_btc) (no leverage)
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np


def config_from_env() -> Dict[str, Any]:
    """Read MoonWire config from os.environ."""
    long_thresh = float(os.getenv("MOONWIRE_LONG_THRESH", "0.65"))
    allow_short = os.getenv("MOONWIRE_ALLOW_SHORT", "0").strip() == "1"
    max_exposure = float(os.getenv("MOONWIRE_MAX_EXPOSURE", "0.985"))
    strict_ts_match = os.getenv("MOONWIRE_STRICT_TS_MATCH", "1").strip() == "1"
    return {
        "long_thresh": long_thresh,
        "allow_short": allow_short,
        "max_exposure": max_exposure,
        "strict_ts_match": strict_ts_match,
    }


def load_feed(filepath: Path) -> Dict[int, float]:
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
            prob = float(rec["probability"])
            feed[ts] = prob
    return feed


def apply_weight_rule(
    prob: float,
    long_thresh: float,
    allow_short: bool,
    max_exposure: float,
) -> Tuple[float, float]:
    """
    Map probability to (w_btc, prob_used). prob_used = prob for trace; caller uses NaN for HOLD.
    - allow_short=0: prob >= long_thresh -> w_btc = max_exposure, else 0
    - allow_short=1: prob >= long_thresh -> +max_exposure; prob <= (1-long_thresh) -> -max_exposure; else 0
    """
    if allow_short:
        if prob >= long_thresh:
            return max_exposure, prob
        if prob <= (1.0 - long_thresh):
            return -max_exposure, prob
        return 0.0, prob
    if prob >= long_thresh:
        return max_exposure, prob
    return 0.0, prob


def get_series_for_timeline(
    bar_timestamps_unix: np.ndarray,
    feed: Dict[int, float],
    config: Dict[str, Any],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    For each bar timestamp, look up probability in feed and compute w_btc.
    Returns (w_btc, prob_used) arrays. prob_used is NaN where missing (HOLD) when strict=False.
    Raises KeyError if strict=True and any bar timestamp is missing from feed.
    """
    long_thresh = config["long_thresh"]
    allow_short = config["allow_short"]
    max_exposure = config["max_exposure"]
    strict = config["strict_ts_match"]

    n = len(bar_timestamps_unix)
    w_btc = np.zeros(n, dtype=float)
    prob_used = np.full(n, np.nan, dtype=float)

    for i in range(n):
        ts = int(bar_timestamps_unix[i])
        if ts not in feed:
            if strict:
                raise KeyError(
                    "Bar timestamp %s not found in MoonWire feed (MOONWIRE_STRICT_TS_MATCH=1)."
                    % ts
                )
            w_btc[i] = 0.0
            prob_used[i] = np.nan
            continue
        prob = feed[ts]
        w_btc[i], prob_used[i] = apply_weight_rule(
            prob, long_thresh, allow_short, max_exposure
        )
    return w_btc, prob_used
