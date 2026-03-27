"""
Unit tests for MoonWire signal integration strategy.

Tests:
- Signal mapping (LONG/SHORT/FLAT based on thresholds)
- Timestamp alignment
- Determinism
- Missing timestamp error handling
- Config validation

Run:
    pytest tests/test_sg_moonwire_intent.py -v
"""

import json
import os
import tempfile
from pathlib import Path
import pandas as pd
import pytest
from runtime.argus.research.strategies.sg_moonwire_intent_v1 import (
    MoonWireConfig,
    map_probability_to_intent,
    generate_intent,
    validate_feed_alignment,
)


# ============================
# Fixtures
# ============================

@pytest.fixture
def sample_signal_feed(tmp_path):
    """Create a temporary signal feed JSONL file."""
    signal_file = tmp_path / "test_signals.jsonl"
    
    # Generate sample signals (hourly, 7 days)
    signals = []
    base_ts = 1704067200  # 2024-01-01 00:00:00 UTC
    for i in range(168):  # 7 days of hourly signals
        ts = base_ts + (i * 3600)
        prob = 0.5 + (i % 10) * 0.05  # Varies from 0.5 to 0.95
        signals.append({
            "timestamp": ts,
            "probability": prob,
            "symbol": "BTC-USD",
            "model_version": "v1.0.0",
        })
    
    with open(signal_file, "w") as f:
        for sig in signals:
            f.write(json.dumps(sig) + "\n")
    
    return str(signal_file)


@pytest.fixture
def sample_df():
    """Create a sample DataFrame of candles."""
    timestamps = pd.date_range("2024-01-01", periods=24, freq="h", tz="UTC")
    df = pd.DataFrame(
        {
            "Open": 40000.0,
            "High": 40100.0,
            "Low": 39900.0,
            "Close": 40000.0,
            "Volume": 100.0,
        },
        index=timestamps,
    )
    return df


@pytest.fixture
def mock_env(sample_signal_feed):
    """Set environment variables for testing."""
    original_env = os.environ.copy()
    
    os.environ["MOONWIRE_SIGNAL_FILE"] = sample_signal_feed
    os.environ["MOONWIRE_LONG_THRESH"] = "0.65"
    os.environ["MOONWIRE_SHORT_THRESH"] = "0.35"
    os.environ["MOONWIRE_ALLOW_SHORT"] = "0"
    os.environ["MOONWIRE_REQUIRE_EXACT_TS"] = "1"
    
    yield
    
    # Restore original environment
    os.environ.clear()
    os.environ.update(original_env)


# ============================
# Config Tests
# ============================

def test_config_validation_missing_file(tmp_path):
    """Test config validation fails when signal file doesn't exist."""
    os.environ["MOONWIRE_SIGNAL_FILE"] = str(tmp_path / "nonexistent.jsonl")
    
    config = MoonWireConfig()
    with pytest.raises(FileNotFoundError):
        config.validate()


def test_config_validation_no_file_path():
    """Test config validation fails when MOONWIRE_SIGNAL_FILE not set."""
    if "MOONWIRE_SIGNAL_FILE" in os.environ:
        del os.environ["MOONWIRE_SIGNAL_FILE"]
    
    config = MoonWireConfig()
    with pytest.raises(ValueError, match="MOONWIRE_SIGNAL_FILE not set"):
        config.validate()


def test_config_validation_invalid_thresholds(sample_signal_feed):
    """Test config validation fails with invalid thresholds."""
    os.environ["MOONWIRE_SIGNAL_FILE"] = sample_signal_feed
    os.environ["MOONWIRE_LONG_THRESH"] = "1.5"  # Invalid
    
    config = MoonWireConfig()
    with pytest.raises(ValueError, match="must be 0-1"):
        config.validate()


def test_config_defaults(sample_signal_feed):
    """Test default config values."""
    os.environ["MOONWIRE_SIGNAL_FILE"] = sample_signal_feed
    # Clear other env vars
    for key in ["MOONWIRE_LONG_THRESH", "MOONWIRE_SHORT_THRESH", "MOONWIRE_ALLOW_SHORT"]:
        if key in os.environ:
            del os.environ[key]
    
    config = MoonWireConfig()
    config.validate()
    
    assert config.long_thresh == 0.65
    assert config.short_thresh == 0.35
    assert config.allow_short is False
    assert config.require_exact_ts is True
    assert config.strict_ts_match is True
    assert config.max_exposure == 0.985


# ============================
# Signal Mapping Tests
# ============================

def test_map_probability_long_signal():
    """Test LONG signal generation."""
    config = MoonWireConfig()
    config.long_thresh = 0.65
    config.short_thresh = 0.35
    config.allow_short = False

    intent = map_probability_to_intent(0.75, config)

    assert intent["action"] == "ENTER_LONG"
    assert intent["confidence"] == 0.75
    assert "LONG" in intent["reason"]
    assert intent["meta"]["probability"] == 0.75


def test_map_probability_short_signal_disabled():
    """Test SHORT signal when ALLOW_SHORT=0."""
    config = MoonWireConfig()
    config.long_thresh = 0.65
    config.short_thresh = 0.35
    config.allow_short = False

    intent = map_probability_to_intent(0.25, config)

    # Should be HOLD (neutral), not EXIT, when shorts disabled
    assert intent["action"] == "HOLD"
    assert "neutral" in intent["reason"]


def test_map_probability_short_signal_enabled():
    """Test SHORT signal when ALLOW_SHORT=1."""
    config = MoonWireConfig()
    config.long_thresh = 0.65
    config.short_thresh = 0.35
    config.allow_short = True

    intent = map_probability_to_intent(0.25, config)

    assert intent["action"] == "EXIT"  # Using EXIT as SHORT proxy
    assert "SHORT" in intent["reason"]
    assert intent["meta"]["short_signal"] is True
    assert intent["confidence"] == 0.75  # Inverted (1 - 0.25)


def test_map_probability_flat_signal():
    """Test HOLD signal in neutral zone."""
    config = MoonWireConfig()
    config.long_thresh = 0.65
    config.short_thresh = 0.35
    config.allow_short = False

    intent = map_probability_to_intent(0.50, config)

    assert intent["action"] == "HOLD"
    assert "neutral" in intent["reason"]
    assert intent["meta"]["probability"] == 0.50


# ============================
# Integration Tests
# ============================

def test_generate_intent_exact_match(mock_env, sample_signal_feed):
    """Test generate_intent with exact timestamp match (last bar = 2024-01-01 00:00, prob=0.5 -> neutral)."""
    import runtime.argus.research.strategies.sg_moonwire_intent_v1 as mw
    mw._SIGNAL_CACHE = None
    mw._CONFIG_CACHE = None

    # Single bar at first feed timestamp so we get prob=0.5 (neutral)
    df = pd.DataFrame(
        {"Open": 40000.0, "High": 40100.0, "Low": 39900.0, "Close": 40000.0, "Volume": 100.0},
        index=pd.DatetimeIndex([pd.Timestamp("2024-01-01", tz="UTC")]),
    )
    df.index.name = "Timestamp"
    intent = generate_intent(df, ctx=None)

    assert intent["action"] == "HOLD"
    assert "neutral" in intent["reason"]


def test_generate_intent_missing_timestamp_fail_fast(mock_env):
    """Test generate_intent fails with missing timestamp when MOONWIRE_STRICT_TS_MATCH=1 (or REQUIRE_EXACT_TS=1)."""
    # Clear module cache
    import runtime.argus.research.strategies.sg_moonwire_intent_v1 as mw
    mw._SIGNAL_CACHE = None
    mw._CONFIG_CACHE = None
    
    # Create DataFrame with timestamp NOT in signal feed
    df = pd.DataFrame(
        {"Close": [40000.0]},
        index=pd.date_range("2025-01-01", periods=1, freq="h", tz="UTC"),
    )
    
    with pytest.raises(KeyError, match="not found in signal feed"):
        generate_intent(df, ctx=None)


def test_generate_intent_fallback_mode(sample_signal_feed):
    """Test generate_intent with fallback to nearest prior signal."""
    # Set REQUIRE_EXACT_TS=0 for fallback mode
    os.environ["MOONWIRE_SIGNAL_FILE"] = sample_signal_feed
    os.environ["MOONWIRE_REQUIRE_EXACT_TS"] = "0"
    
    # Clear module cache
    import runtime.argus.research.strategies.sg_moonwire_intent_v1 as mw
    mw._SIGNAL_CACHE = None
    mw._CONFIG_CACHE = None
    
    # Create DataFrame with timestamp slightly after feed ends
    df = pd.DataFrame(
        {"Close": [40000.0]},
        index=pd.date_range("2024-01-10", periods=1, freq="h", tz="UTC"),
    )
    
    intent = generate_intent(df, ctx=None)

    # With REQUIRE_EXACT_TS=0 we used to fall back to nearest prior; now we return HOLD/0 (non-strict)
    assert intent["action"] == "HOLD"
    assert intent["desired_exposure_frac"] == 0.0


# ============================
# Determinism Tests
# ============================

def test_determinism(mock_env, sample_df):
    """Test that same inputs produce same outputs."""
    # Clear module cache
    import runtime.argus.research.strategies.sg_moonwire_intent_v1 as mw
    mw._SIGNAL_CACHE = None
    mw._CONFIG_CACHE = None
    
    intent1 = generate_intent(sample_df, ctx=None)
    intent2 = generate_intent(sample_df, ctx=None)

    assert intent1["action"] == intent2["action"]
    assert intent1["confidence"] == intent2["confidence"]
    assert intent1["reason"] == intent2["reason"]
    assert intent1["meta"]["probability"] == intent2["meta"]["probability"]


# ============================
# Alignment Validation Tests
# ============================

def test_validate_feed_alignment_perfect(sample_signal_feed, sample_df):
    """Test alignment validation with perfect match."""
    result = validate_feed_alignment(sample_df, sample_signal_feed)
    
    assert result["total_bars"] == 24
    assert result["matched"] == 24
    assert result["missing"] == 0
    assert result["coverage"] == 1.0


def test_validate_feed_alignment_partial(sample_signal_feed):
    """Test alignment validation with partial match (feed is 2024-01-01 to 2024-01-07; df extends past)."""
    # Feed has 168 hours from 2024-01-01; use df that starts after feed ends so most bars are missing
    df = pd.DataFrame(
        {"Close": [40000.0] * 48},
        index=pd.date_range("2024-01-10", periods=48, freq="h", tz="UTC"),
    )

    result = validate_feed_alignment(df, sample_signal_feed)

    assert result["total_bars"] == 48
    assert result["matched"] < 48  # Bars in 2024-01-10+ are not in feed (2024-01-01 to 2024-01-07)
    assert result["coverage"] < 1.0


# ============================
# Edge Cases
# ============================

def test_empty_signal_feed(tmp_path):
    """Test behavior with empty signal feed (non-strict: HOLD/0 exposure)."""
    empty_file = tmp_path / "empty.jsonl"
    empty_file.write_text("")

    os.environ["MOONWIRE_SIGNAL_FILE"] = str(empty_file)
    os.environ["MOONWIRE_STRICT_TS_MATCH"] = "0"

    import runtime.argus.research.strategies.sg_moonwire_intent_v1 as mw
    mw._SIGNAL_CACHE = None
    mw._CONFIG_CACHE = None

    df = pd.DataFrame(
        {"Close": [40000.0]},
        index=pd.date_range("2024-01-01", periods=1, freq="h", tz="UTC"),
    )

    intent = generate_intent(df, ctx=None)

    assert intent["action"] == "HOLD"
    assert intent["desired_exposure_frac"] == 0.0


def test_threshold_edge_cases():
    """Test probability exactly at thresholds."""
    config = MoonWireConfig()
    config.long_thresh = 0.65
    config.short_thresh = 0.35
    config.allow_short = True

    # Exactly at LONG threshold
    intent_long = map_probability_to_intent(0.65, config)
    assert intent_long["action"] == "ENTER_LONG"

    # Exactly at SHORT threshold
    intent_short = map_probability_to_intent(0.35, config)
    assert intent_short["action"] == "EXIT"

    # Just above SHORT threshold
    intent_neutral = map_probability_to_intent(0.351, config)
    assert intent_neutral["action"] == "HOLD"


# ============================
# Experiment 1: geometry adapter tests
# ============================


def test_deterministic_mapping_same_probability_same_exposure():
    """Deterministic mapping: same feed probability -> same desired_exposure_frac."""
    config = MoonWireConfig()
    config.long_thresh = 0.65
    config.short_thresh = 0.35
    config.allow_short = False

    for prob in [0.0, 0.5, 0.65, 0.70, 0.99]:
        a = map_probability_to_intent(prob, config)
        b = map_probability_to_intent(prob, config)
        assert a["desired_exposure_frac"] == b["desired_exposure_frac"]
        assert a["action"] == b["action"]


def test_strict_timestamp_match_raises_when_missing(mock_env):
    """Strict mode: missing bar timestamp in feed raises."""
    import runtime.argus.research.strategies.sg_moonwire_intent_v1 as mw
    mw._SIGNAL_CACHE = None
    mw._CONFIG_CACHE = None
    os.environ["MOONWIRE_STRICT_TS_MATCH"] = "1"

    df = pd.DataFrame(
        {"Close": [40000.0]},
        index=pd.date_range("2025-06-01", periods=1, freq="h", tz="UTC"),
    )
    with pytest.raises(KeyError, match="not found in signal feed"):
        generate_intent(df, ctx=None)


def test_non_strict_produces_exposure_zero_when_missing(sample_signal_feed):
    """Non-strict mode: missing bar timestamp -> HOLD with desired_exposure_frac=0."""
    os.environ["MOONWIRE_SIGNAL_FILE"] = sample_signal_feed
    os.environ["MOONWIRE_STRICT_TS_MATCH"] = "0"

    import runtime.argus.research.strategies.sg_moonwire_intent_v1 as mw
    mw._SIGNAL_CACHE = None
    mw._CONFIG_CACHE = None

    df = pd.DataFrame(
        {"Close": [40000.0]},
        index=pd.date_range("2025-06-01", periods=1, freq="h", tz="UTC"),
    )
    intent = generate_intent(df, ctx=None)
    assert intent["action"] == "HOLD"
    assert intent["desired_exposure_frac"] == 0.0


def test_exposure_capped_at_moonwire_max_exposure():
    """LONG intent desired_exposure_frac is capped at MOONWIRE_MAX_EXPOSURE."""
    config = MoonWireConfig()
    config.long_thresh = 0.65
    config.short_thresh = 0.35
    config.allow_short = False
    config.max_exposure = 0.5  # cap at 50%

    intent = map_probability_to_intent(0.90, config)
    assert intent["action"] == "ENTER_LONG"
    assert intent["desired_exposure_frac"] == 0.5

    config.max_exposure = 0.985
    intent2 = map_probability_to_intent(0.90, config)
    assert intent2["desired_exposure_frac"] == 0.985


if __name__ == "__main__":
    pytest.main([__file__, "-v"])



