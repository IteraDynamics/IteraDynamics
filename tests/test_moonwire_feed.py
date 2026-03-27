"""
Tests for MoonWire feed orchestration (runtime/argus/integrations/moonwire_feed).

Itera consumes MoonWire outputs only; no inference in Itera. These tests cover:
- missing file
- stale file (when max_age_seconds set)
- valid file (schema + fresh)
- malformed schema
"""

import json
import os
import time
from pathlib import Path

import pytest

# Ensure repo root on path for runtime.argus
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(_REPO_ROOT))

from runtime.argus.integrations.moonwire_feed import (
    validate_schema,
    is_fresh,
    ensure_feed,
    REQUIRED_KEYS,
)


# ============== validate_schema ==============

def test_validate_schema_missing_file(tmp_path):
    path = tmp_path / "nonexistent.jsonl"
    valid, msg = validate_schema(path)
    assert not valid
    assert "does not exist" in msg or "exist" in msg


def test_validate_schema_valid_file(tmp_path):
    path = tmp_path / "signals.jsonl"
    with open(path, "w") as f:
        f.write('{"timestamp": 1546300800, "probability": 0.65}\n')
        f.write('{"timestamp": 1546304400, "probability": 0.58, "symbol": "BTC-USD"}\n')
    valid, msg = validate_schema(path)
    assert valid
    assert "valid" in msg


def test_validate_schema_malformed_not_json(tmp_path):
    path = tmp_path / "bad.jsonl"
    with open(path, "w") as f:
        f.write("not json\n")
    valid, msg = validate_schema(path)
    assert not valid
    assert "JSON" in msg or "invalid" in msg.lower()


def test_validate_schema_malformed_missing_timestamp(tmp_path):
    path = tmp_path / "bad.jsonl"
    with open(path, "w") as f:
        f.write('{"probability": 0.5}\n')
    valid, msg = validate_schema(path)
    assert not valid
    assert "timestamp" in msg


def test_validate_schema_malformed_missing_probability(tmp_path):
    path = tmp_path / "bad.jsonl"
    with open(path, "w") as f:
        f.write('{"timestamp": 1546300800}\n')
    valid, msg = validate_schema(path)
    assert not valid
    assert "probability" in msg


def test_validate_schema_malformed_probability_out_of_range(tmp_path):
    path = tmp_path / "bad.jsonl"
    with open(path, "w") as f:
        f.write('{"timestamp": 1546300800, "probability": 1.5}\n')
    valid, msg = validate_schema(path)
    assert not valid
    assert "probability" in msg or "0" in msg or "1" in msg


# ============== is_fresh ==============

def test_is_fresh_missing_file(tmp_path):
    path = tmp_path / "missing.jsonl"
    assert not is_fresh(path)
    assert not is_fresh(path, max_age_seconds=3600)


def test_is_fresh_no_max_age(tmp_path):
    path = tmp_path / "signals.jsonl"
    path.write_text('{"timestamp": 1, "probability": 0.5}\n')
    assert is_fresh(path) is True
    assert is_fresh(path, max_age_seconds=None) is True


def test_is_fresh_stale_file(tmp_path):
    path = tmp_path / "old.jsonl"
    path.write_text('{"timestamp": 1, "probability": 0.5}\n')
    # Set mtime to 2 hours ago (e.g. on systems where we can't easily mock)
    old = time.time() - 7200
    try:
        os.utime(path, (old, old))
    except OSError:
        pytest.skip("cannot set mtime")
    assert is_fresh(path, max_age_seconds=3600) is False


def test_is_fresh_recent_file(tmp_path):
    path = tmp_path / "new.jsonl"
    path.write_text('{"timestamp": 1, "probability": 0.5}\n')
    assert is_fresh(path, max_age_seconds=3600) is True


# ============== ensure_feed ==============

def test_ensure_feed_missing_file_no_backend(tmp_path):
    """When feed is missing and MOONWIRE_BACKEND_ROOT is not set, ensure_feed returns (False, None)."""
    feed_path = tmp_path / "feed.jsonl"
    ok, path = ensure_feed(
        signal_file=str(feed_path),
        backend_root=None,
        skip_export_if_missing_backend=True,
    )
    assert ok is False
    assert path is None


def test_ensure_feed_valid_file_returns_path(tmp_path):
    """When feed exists and is valid, ensure_feed returns (True, path)."""
    feed_path = tmp_path / "feed.jsonl"
    with open(feed_path, "w") as f:
        f.write('{"timestamp": 1546300800, "probability": 0.6}\n')
    ok, path = ensure_feed(signal_file=str(feed_path), backend_root=tmp_path / "nonexistent")
    assert ok is True
    assert path is not None
    assert Path(path).exists()


def test_ensure_feed_missing_backend_skip_export(tmp_path):
    """When feed is missing and backend path does not exist, skip_export_if_missing_backend=True returns (False, None)."""
    feed_path = tmp_path / "feed.jsonl"
    fake_backend = tmp_path / "moonwire-backend"
    ok, path = ensure_feed(
        signal_file=str(feed_path),
        backend_root=str(fake_backend),
        skip_export_if_missing_backend=True,
    )
    assert ok is False
    assert path is None


def test_ensure_feed_missing_backend_no_skip_raises(tmp_path):
    """When feed is missing and backend does not exist, skip_export_if_missing_backend=False raises."""
    feed_path = tmp_path / "feed.jsonl"
    fake_backend = tmp_path / "moonwire-backend"
    with pytest.raises(FileNotFoundError):
        ensure_feed(
            signal_file=str(feed_path),
            backend_root=str(fake_backend),
            skip_export_if_missing_backend=False,
        )


def test_required_keys():
    """Required JSONL keys are timestamp and probability."""
    assert REQUIRED_KEYS == {"timestamp", "probability"}
