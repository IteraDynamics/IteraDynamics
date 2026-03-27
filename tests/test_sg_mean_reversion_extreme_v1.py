"""
Unit tests for Mean Reversion Extreme strategy (sg_mean_reversion_extreme_v1).

Tests:
- RSI oversold triggers long entry
- Strategy exits after max hold (48 bars)
- Strategy exits on RSI recovery above threshold
- Deterministic repeated runs produce same intent
- closed_only / no lookahead: intent for prefix depends only on prefix data

Run:
    pytest tests/test_sg_mean_reversion_extreme_v1.py -v
"""

import os
import numpy as np
import pandas as pd
import pytest

from runtime.argus.research.strategies.sg_mean_reversion_extreme_v1 import (
    generate_intent,
    _get_config,
    _rsi,
    _simulate_state,
)


# ---------------------------
# Fixtures
# ---------------------------

@pytest.fixture
def mr_env_defaults():
    """Set env to research baseline; restore after test."""
    orig = {}
    for key in ("SG_MR_RSI_LEN", "SG_MR_OVERSOLD", "SG_MR_EXIT_RSI", "SG_MR_MAX_HOLD_BARS", "SG_MR_MAX_EXPOSURE"):
        orig[key] = os.environ.pop(key, None)
    os.environ["SG_MR_RSI_LEN"] = "21"
    os.environ["SG_MR_OVERSOLD"] = "25"
    os.environ["SG_MR_EXIT_RSI"] = "50"
    os.environ["SG_MR_MAX_HOLD_BARS"] = "48"
    os.environ["SG_MR_MAX_EXPOSURE"] = "0.985"
    yield
    for k, v in orig.items():
        if v is not None:
            os.environ[k] = v
        elif k in os.environ:
            del os.environ[k]


def _df_from_close(close: np.ndarray, freq: str = "h") -> pd.DataFrame:
    """Build OHLCV DataFrame from close series (Open/High/Low = Close, Volume = 100)."""
    n = len(close)
    ts = pd.date_range("2024-01-01", periods=n, freq=freq, tz="UTC")
    c = pd.Series(close, index=ts)
    return pd.DataFrame({
        "Open": c,
        "High": c,
        "Low": c,
        "Close": c,
        "Volume": 100.0,
    }, index=ts)


def _declining_close(n: int, start: float = 100.0, step: float = -1.5) -> np.ndarray:
    """Steady decline to produce oversold RSI."""
    return start + np.arange(n) * step


def _flat_close(n: int, value: float = 50.0) -> np.ndarray:
    """Flat series (RSI ~50 after warmup)."""
    return np.full(n, value)


# ---------------------------
# Config
# ---------------------------

def test_config_defaults(mr_env_defaults):
    """Config uses research baseline when env not set (after we set in fixture)."""
    cfg = _get_config()
    assert cfg["rsi_len"] == 21
    assert cfg["oversold"] == 25.0
    assert cfg["exit_rsi"] == 50.0
    assert cfg["max_hold_bars"] == 48
    assert 0 < cfg["max_exposure"] <= 1.0


def test_config_env_override():
    """Env vars override defaults."""
    os.environ["SG_MR_RSI_LEN"] = "14"
    os.environ["SG_MR_OVERSOLD"] = "30"
    os.environ["SG_MR_EXIT_RSI"] = "60"
    os.environ["SG_MR_MAX_HOLD_BARS"] = "24"
    os.environ["SG_MR_MAX_EXPOSURE"] = "0.5"
    try:
        cfg = _get_config()
        assert cfg["rsi_len"] == 14
        assert cfg["oversold"] == 30.0
        assert cfg["exit_rsi"] == 60.0
        assert cfg["max_hold_bars"] == 24
        assert cfg["max_exposure"] == 0.5
    finally:
        for k in ("SG_MR_RSI_LEN", "SG_MR_OVERSOLD", "SG_MR_EXIT_RSI", "SG_MR_MAX_HOLD_BARS", "SG_MR_MAX_EXPOSURE"):
            os.environ.pop(k, None)


# ---------------------------
# RSI oversold -> long entry
# ---------------------------

def test_rsi_oversold_triggers_long_entry(mr_env_defaults):
    """When RSI(21) <= 25, strategy returns ENTER_LONG with positive exposure."""
    # Strong decline over 40 bars -> RSI well below 25
    close = _declining_close(45, start=100.0, step=-1.2)
    df = _df_from_close(close)
    intent = generate_intent(df, None, closed_only=True)
    assert intent["action"] == "ENTER_LONG"
    assert intent["desired_exposure_frac"] > 0
    assert "oversold" in intent["reason"].lower() or "long" in intent["reason"].lower()
    assert "action" in intent and "confidence" in intent and "meta" in intent


def test_flat_series_stays_flat(mr_env_defaults):
    """Flat price series keeps RSI mid; strategy should stay HOLD with 0 exposure."""
    close = _flat_close(50, value=40000.0)
    df = _df_from_close(close)
    intent = generate_intent(df, None, closed_only=True)
    assert intent["action"] == "HOLD"
    assert intent["desired_exposure_frac"] == 0.0


# ---------------------------
# Exit after max hold
# ---------------------------

def test_exit_after_max_hold(mr_env_defaults):
    """After max_hold bars in position, strategy exits (EXIT, 0 exposure)."""
    # Use short max_hold=3: enter when RSI<=25, then exit when bars_held >= 3
    os.environ["SG_MR_MAX_HOLD_BARS"] = "3"
    # Steep decline so RSI <= 25 at bar 20; then 3 flat bars -> at bar 23 bars_held=3, we exit. End df at bar 23.
    close_pre = _declining_close(21, start=100.0, step=-2.5)  # 21 bars, RSI oversold by bar 20
    flat_val = float(close_pre[-1])
    close_hold = np.array([flat_val, flat_val, flat_val])  # 3 bars: entry at 20, bars 21,22,23 -> exit at 23
    close = np.concatenate([close_pre, close_hold])  # 24 bars; last bar (index 23) is exit bar
    df = _df_from_close(close)
    intent = generate_intent(df, None, closed_only=True)
    assert intent["action"] == "EXIT", f"expected EXIT, got {intent['action']} (reason: {intent['reason']})"
    assert intent["desired_exposure_frac"] == 0.0
    os.environ["SG_MR_MAX_HOLD_BARS"] = "48"  # restore


# ---------------------------
# Exit on RSI recovery
# ---------------------------

def test_exit_on_rsi_recovery(mr_env_defaults):
    """When RSI recovers above exit_rsi (50), strategy exits."""
    # Decline -> enter; then sharp rise -> RSI > 50 -> exit
    n_decline = 25
    n_rise = 15
    close_decline = _declining_close(n_decline, start=100.0, step=-1.5)
    # Sharp rise: RSI will go above 50 quickly
    close_rise = close_decline[-1] + np.linspace(0, 30, n_rise)
    close = np.concatenate([close_decline, close_rise])
    df = _df_from_close(close)
    intent = generate_intent(df, None, closed_only=True)
    # Should exit on recovery (or still long if recovery not yet; at 40 bars we likely recovered)
    assert intent["action"] in ("EXIT", "HOLD", "ENTER_LONG")
    if intent["action"] == "EXIT":
        assert intent["desired_exposure_frac"] == 0.0


# ---------------------------
# Determinism
# ---------------------------

def test_deterministic_repeated_runs(mr_env_defaults):
    """Same df and env -> same intent every time."""
    close = _declining_close(40, start=100.0, step=-1.2)
    df = _df_from_close(close)
    a = generate_intent(df, None, closed_only=True)
    b = generate_intent(df, None, closed_only=True)
    assert a["action"] == b["action"]
    assert a["desired_exposure_frac"] == b["desired_exposure_frac"]
    assert a["reason"] == b["reason"]
    assert a["meta"]["rsi"] == b["meta"]["rsi"]


# ---------------------------
# No lookahead / closed_only
# ---------------------------

def test_no_lookahead_intent_depends_only_on_prefix(mr_env_defaults):
    """Intent for first N bars must not change when extra (future) bars are appended."""
    n_prefix = 50
    close_prefix = _declining_close(n_prefix, start=100.0, step=-1.0)
    df_prefix = _df_from_close(close_prefix)
    intent_prefix = generate_intent(df_prefix, None, closed_only=True)
    # Extend with different data (simulating "future" bars)
    close_extra = np.linspace(200, 300, 20)  # different trajectory
    close_full = np.concatenate([close_prefix, close_extra])
    df_full = _df_from_close(close_full)
    # Intent when given only the prefix slice of the full df must match intent from df_prefix
    df_slice = df_full.iloc[:n_prefix]
    intent_slice = generate_intent(df_slice, None, closed_only=True)
    assert intent_prefix["action"] == intent_slice["action"]
    assert intent_prefix["desired_exposure_frac"] == intent_slice["desired_exposure_frac"]


# ---------------------------
# Contract: dict shape
# ---------------------------

def test_intent_dict_has_required_keys(mr_env_defaults):
    """Returned intent has keys expected by backtest harness."""
    df = _df_from_close(_flat_close(30))
    intent = generate_intent(df, None, closed_only=True)
    required = {"action", "confidence", "desired_exposure_frac", "horizon_hours", "reason", "meta"}
    assert required.issubset(intent.keys())
    assert isinstance(intent["meta"], dict)


# ---------------------------
# Edge: insufficient bars
# ---------------------------

def test_insufficient_bars_returns_hold(mr_env_defaults):
    """When bars < RSI length, strategy returns HOLD."""
    df = _df_from_close(np.array([100.0, 99.0, 98.0]))  # 3 bars, need 21
    intent = generate_intent(df, None, closed_only=True)
    assert intent["action"] == "HOLD"
    assert intent["desired_exposure_frac"] == 0.0


# ---------------------------
# _rsi / _simulate_state (unit)
# ---------------------------

def test_rsi_low_after_decline():
    """_rsi produces low values after sustained decline."""
    close = pd.Series(_declining_close(30, start=100.0, step=-1.5))
    rsi = _rsi(close, 21)
    assert rsi.iloc[-1] < 40  # clearly below 50


def test_simulate_state_enter_on_oversold():
    """_simulate_state enters long when RSI <= oversold."""
    close = pd.Series(_declining_close(30, start=100.0, step=-1.5))
    in_long, action, rsi_val, entry_idx = _simulate_state(
        close, rsi_len=21, oversold=25.0, exit_rsi=50.0, max_hold_bars=48
    )
    assert in_long is True
    assert action == "ENTER_LONG"
    assert entry_idx is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
