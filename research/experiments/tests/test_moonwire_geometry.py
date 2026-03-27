"""
Unit tests for MoonWire loader and BTC_MOONWIRE_ONLY geometry (Experiment 1).

Uses tiny synthetic JSONL feed + 5-bar price series. Verifies:
- Deterministic alignment: same feed + timeline -> same w_btc, prob_used
- STRICT_TS_MATCH=1: bar timestamp missing from feed -> raise
- STRICT_TS_MATCH=0: bar timestamp missing -> HOLD (w_btc=0, prob_used=NaN)
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Add experiments to path
_EXPERIMENTS = Path(__file__).resolve().parents[1]
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(_EXPERIMENTS))
if str(_REPO_ROOT / "runtime" / "argus") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "runtime" / "argus"))

from moonwire_loader import (
    apply_weight_rule,
    config_from_env,
    get_series_for_timeline,
    load_feed,
)


# -----------------------------------------------------------------------------
# Fixtures: 5-bar timeline (hourly UTC), synthetic JSONL feed
# -----------------------------------------------------------------------------
# Bar timestamps: 2024-01-01 00:00, 01:00, 02:00, 03:00, 04:00 UTC
BASE_TS = 1704067200  # 2024-01-01 00:00:00 UTC


@pytest.fixture
def synthetic_feed(tmp_path):
    """JSONL with 5 rows: ts 0..4 hours from BASE_TS, probs 0.2, 0.5, 0.7, 0.35, 0.8."""
    path = tmp_path / "signals.jsonl"
    probs = [0.2, 0.5, 0.7, 0.35, 0.8]
    with open(path, "w") as f:
        for i in range(5):
            rec = {"timestamp": BASE_TS + i * 3600, "probability": probs[i]}
            f.write(json.dumps(rec) + "\n")
    return path


@pytest.fixture
def five_bar_timestamps_unix():
    """Unix timestamps for 5 bars (hourly from BASE_TS)."""
    return np.array([BASE_TS + i * 3600 for i in range(5)], dtype=float)


# -----------------------------------------------------------------------------
# Deterministic alignment
# -----------------------------------------------------------------------------
def test_deterministic_alignment(synthetic_feed, five_bar_timestamps_unix):
    """Same feed + same timeline -> same w_btc and prob_used every run."""
    feed = load_feed(synthetic_feed)
    config = {
        "long_thresh": 0.65,
        "allow_short": 0,
        "max_exposure": 0.985,
        "strict_ts_match": True,
    }
    w1, p1 = get_series_for_timeline(five_bar_timestamps_unix, feed, config)
    w2, p2 = get_series_for_timeline(five_bar_timestamps_unix, feed, config)
    np.testing.assert_array_almost_equal(w1, w2)
    np.testing.assert_array_almost_equal(p1, p2)
    # With long_thresh=0.65, allow_short=0: prob>=0.65 -> w_btc=0.985 else 0
    # probs: 0.2, 0.5, 0.7, 0.35, 0.8 -> w_btc: 0, 0, 0.985, 0, 0.985
    expected_w = np.array([0.0, 0.0, 0.985, 0.0, 0.985])
    np.testing.assert_array_almost_equal(w1, expected_w)
    np.testing.assert_array_almost_equal(p1, [0.2, 0.5, 0.7, 0.35, 0.8])


# -----------------------------------------------------------------------------
# Strict timestamp match: missing bar -> raise
# -----------------------------------------------------------------------------
def test_strict_ts_match_raises_on_missing(synthetic_feed):
    """STRICT_TS_MATCH=1 and a bar timestamp not in feed -> KeyError."""
    feed = load_feed(synthetic_feed)
    config = {
        "long_thresh": 0.65,
        "allow_short": 0,
        "max_exposure": 0.985,
        "strict_ts_match": True,
    }
    # Bar timestamps: one bar (2024-01-10) is not in the feed
    bar_ts = np.array([BASE_TS, BASE_TS + 3600, 1704844800, BASE_TS + 3 * 3600, BASE_TS + 4 * 3600], dtype=float)
    # 1704844800 = 2024-01-10 00:00 UTC, not in feed
    with pytest.raises(KeyError, match="not found in MoonWire feed"):
        get_series_for_timeline(bar_ts, feed, config)


# -----------------------------------------------------------------------------
# Non-strict: missing bar -> HOLD (w_btc=0, prob_used=NaN)
# -----------------------------------------------------------------------------
def test_non_strict_missing_bar_hold(synthetic_feed):
    """STRICT_TS_MATCH=0 and missing bar -> w_btc=0, prob_used=NaN."""
    feed = load_feed(synthetic_feed)
    config = {
        "long_thresh": 0.65,
        "allow_short": 0,
        "max_exposure": 0.985,
        "strict_ts_match": False,
    }
    bar_ts = np.array([BASE_TS, BASE_TS + 3600, 1704844800, BASE_TS + 3 * 3600, BASE_TS + 4 * 3600], dtype=float)
    w_btc, prob_used = get_series_for_timeline(bar_ts, feed, config)
    assert w_btc[2] == 0.0
    assert np.isnan(prob_used[2])
    assert w_btc[0] == 0.0 and w_btc[1] == 0.0 and w_btc[3] == 0.0 and w_btc[4] == 0.985


# -----------------------------------------------------------------------------
# Weight rule: allow_short=1, short thresh = 1 - long_thresh
# -----------------------------------------------------------------------------
def test_weight_rule_allow_short():
    """prob >= long_thresh -> +max_exposure; prob <= (1-long_thresh) -> -max_exposure; else 0."""
    long_thresh = 0.65
    allow_short = True
    max_exp = 0.985
    w, p = apply_weight_rule(0.7, long_thresh, allow_short, max_exp)
    assert w == max_exp and p == 0.7
    w, p = apply_weight_rule(0.3, long_thresh, allow_short, max_exp)
    assert w == -max_exp and p == 0.3  # 0.3 <= (1-0.65)=0.35
    w, p = apply_weight_rule(0.5, long_thresh, allow_short, max_exp)
    assert w == 0.0 and p == 0.5


def test_weight_rule_no_short():
    """allow_short=0: prob >= long_thresh -> max_exposure, else 0."""
    w, _ = apply_weight_rule(0.65, 0.65, False, 0.985)
    assert w == 0.985
    w, _ = apply_weight_rule(0.64, 0.65, False, 0.985)
    assert w == 0.0


# -----------------------------------------------------------------------------
# Config from env
# -----------------------------------------------------------------------------
def test_config_from_env_defaults(synthetic_feed):
    """With minimal env, config has expected defaults."""
    os.environ["MOONWIRE_SIGNAL_FILE"] = str(synthetic_feed)
    for k in ["MOONWIRE_LONG_THRESH", "MOONWIRE_ALLOW_SHORT", "MOONWIRE_MAX_EXPOSURE", "MOONWIRE_STRICT_TS_MATCH"]:
        os.environ.pop(k, None)
    cfg = config_from_env()
    assert cfg["long_thresh"] == 0.65
    assert cfg["allow_short"] is False
    assert cfg["max_exposure"] == 0.985
    assert cfg["strict_ts_match"] is True
