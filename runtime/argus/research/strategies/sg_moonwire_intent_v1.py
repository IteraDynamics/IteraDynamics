"""
MoonWire Signal Integration Strategy (Layer-2)

Consumes pre-generated MoonWire signal feeds (JSONL format) and maps
probability scores to StrategyIntent actions.

Configuration via environment variables:
- MOONWIRE_SIGNAL_FILE: Path to signals.jsonl
- MOONWIRE_LONG_THRESH: Probability threshold for LONG (default 0.65)
- MOONWIRE_SHORT_THRESH: Probability threshold for SHORT (default 0.35)
- MOONWIRE_ALLOW_SHORT: Enable SHORT signals (default 0)
- MOONWIRE_MAX_EXPOSURE: Cap on desired exposure (default 0.985)
- MOONWIRE_STRICT_TS_MATCH: 1 = raise if bar timestamp missing in feed; 0 = HOLD/0 exposure (default 1)
- MOONWIRE_REQUIRE_EXACT_TS: Legacy alias for strict timestamp (default 1)

Architecture:
- Deterministic: Same feed + same params = same intents
- Closed-bar semantics: Decision at bar t uses data through bar t
- Fail-fast: Missing timestamp errors by default (no silent fallbacks)
- Module-level caching: Feed loaded once per Python process

Author: Itera Dynamics
Date: 2026-03-04
"""

from __future__ import annotations
import os
import json
from typing import Optional, Dict, Any
from pathlib import Path
import pandas as pd
from runtime.argus.apex_core.strategy_intent import Intent, Action


# ============================
# Configuration
# ============================

class MoonWireConfig:
    """Environment-based configuration for MoonWire strategy."""

    def __init__(self):
        self.signal_file = os.getenv("MOONWIRE_SIGNAL_FILE")
        self.long_thresh = float(os.getenv("MOONWIRE_LONG_THRESH", "0.65"))
        self.short_thresh = float(os.getenv("MOONWIRE_SHORT_THRESH", "0.35"))
        self.allow_short = os.getenv("MOONWIRE_ALLOW_SHORT", "0") == "1"
        # MOONWIRE_STRICT_TS_MATCH takes precedence; fallback to legacy MOONWIRE_REQUIRE_EXACT_TS
        strict_env = os.getenv("MOONWIRE_STRICT_TS_MATCH")
        if strict_env is not None:
            self.strict_ts_match = strict_env.strip() == "1"
        else:
            self.strict_ts_match = os.getenv("MOONWIRE_REQUIRE_EXACT_TS", "1") == "1"
        self.require_exact_ts = self.strict_ts_match  # legacy alias
        self.max_exposure = float(os.getenv("MOONWIRE_MAX_EXPOSURE", "0.985"))

    def validate(self) -> None:
        """Validate configuration and raise clear errors."""
        if not self.signal_file:
            raise ValueError(
                "MOONWIRE_SIGNAL_FILE not set. "
                "Export signals from MoonWire and set env var to path."
            )

        if not Path(self.signal_file).exists():
            raise FileNotFoundError(
                f"Signal file not found: {self.signal_file}\n"
                f"Export signals using: python scripts/export_signal_feed.py"
            )

        if not (0 <= self.long_thresh <= 1):
            raise ValueError(f"MOONWIRE_LONG_THRESH must be 0-1, got {self.long_thresh}")

        if not (0 <= self.short_thresh <= 1):
            raise ValueError(f"MOONWIRE_SHORT_THRESH must be 0-1, got {self.short_thresh}")

        if not (0 < self.max_exposure <= 1.0):
            raise ValueError(f"MOONWIRE_MAX_EXPOSURE must be in (0, 1], got {self.max_exposure}")


# ============================
# Signal Feed Loader
# ============================

_SIGNAL_CACHE: Optional[Dict[int, float]] = None
_CONFIG_CACHE: Optional[MoonWireConfig] = None


def load_signal_feed(config: MoonWireConfig) -> Dict[int, float]:
    """
    Load MoonWire signal feed from JSONL.
    
    Returns:
        Dict mapping timestamp (Unix seconds) -> probability (0-1)
    """
    global _SIGNAL_CACHE
    
    if _SIGNAL_CACHE is not None:
        return _SIGNAL_CACHE
    
    signals = {}
    with open(config.signal_file, "r") as f:
        for line in f:
            record = json.loads(line)
            ts = int(record["timestamp"])
            prob = float(record["probability"])
            signals[ts] = prob
    
    _SIGNAL_CACHE = signals
    return signals


def get_config() -> MoonWireConfig:
    """Get or create cached config."""
    global _CONFIG_CACHE
    
    if _CONFIG_CACHE is None:
        _CONFIG_CACHE = MoonWireConfig()
        _CONFIG_CACHE.validate()
    
    return _CONFIG_CACHE


# ============================
# Signal Mapping
# ============================

def map_probability_to_intent(
    probability: float,
    config: MoonWireConfig,
) -> dict:
    """
    Map MoonWire probability to strategy intent dict.
    
    Thresholds:
    - probability >= LONG_THRESH → ENTER_LONG
    - probability <= SHORT_THRESH (and ALLOW_SHORT=1) → EXIT (short proxy)
    - else → HOLD
    
    Args:
        probability: ML model probability (0-1)
        config: MoonWire configuration
    
    Returns:
        Dict with action, confidence, reason
    """
    if probability >= config.long_thresh:
        exposure = min(config.max_exposure, 1.0)
        return {
            "action": "ENTER_LONG",
            "confidence": probability,
            "desired_exposure_frac": exposure,
            "horizon_hours": None,
            "reason": f"MoonWire LONG signal (p={probability:.3f} >= {config.long_thresh})",
            "meta": {"source": "moonwire", "probability": probability},
        }
    
    elif config.allow_short and probability <= config.short_thresh:
        return {
            "action": "EXIT",
            "confidence": 1.0 - probability,
            "desired_exposure_frac": 0.0,
            "horizon_hours": None,
            "reason": f"MoonWire SHORT signal (p={probability:.3f} <= {config.short_thresh})",
            "meta": {"source": "moonwire", "probability": probability, "short_signal": True},
        }
    
    else:
        return {
            "action": "HOLD",
            "confidence": 0.5,
            "desired_exposure_frac": 0.0,
            "horizon_hours": None,
            "reason": f"MoonWire neutral (p={probability:.3f} in [{config.short_thresh}, {config.long_thresh}])",
            "meta": {"source": "moonwire", "probability": probability},
        }


# ============================
# External Strategy API
# ============================

def generate_intent(df: pd.DataFrame, ctx, closed_only: bool = True) -> dict:
    """
    External Strategy API for MoonWire signals.
    
    Args:
        df: DataFrame of flight_recorder candles (tail = most recent bar)
        ctx: StrategyContext from SG (contains timestamp, product, etc.)
        closed_only: Whether to use closed bars only
    
    Returns:
        Dict with action, confidence, reason
    """
    config = get_config()
    signals = load_signal_feed(config)
    
    # Get current bar timestamp (most recent closed bar)
    last_bar = df.iloc[-1]
    
    # Handle both DatetimeIndex and Timestamp column
    if isinstance(df.index, pd.DatetimeIndex):
        bar_ts = int(pd.Timestamp(last_bar.name).timestamp())
    else:
        # Timestamp is a column, not the index
        bar_ts = int(pd.Timestamp(last_bar['Timestamp']).timestamp())
    
    # Lookup signal for this timestamp
    if bar_ts in signals:
        probability = signals[bar_ts]
        return map_probability_to_intent(probability, config)

    # Handle missing timestamp: strict mode raises; non-strict returns HOLD/0 exposure
    if config.strict_ts_match:
        raise KeyError(
            f"Timestamp {bar_ts} not found in signal feed.\n"
            f"Feed may be stale or misaligned.\n"
            f"Set MOONWIRE_STRICT_TS_MATCH=0 to treat missing bars as HOLD (exposure=0)."
        )

    return {
        "action": "HOLD",
        "confidence": 0.0,
        "desired_exposure_frac": 0.0,
        "horizon_hours": None,
        "reason": "MoonWire bar timestamp not in feed (non-strict mode: HOLD)",
        "meta": {"source": "moonwire", "missing_ts": bar_ts},
    }


# ============================
# Diagnostics
# ============================

def validate_feed_alignment(
    df: pd.DataFrame,
    signal_file: str,
) -> Dict[str, Any]:
    """
    Diagnostic tool to check timestamp alignment between strategy bars and signal feed.
    
    Args:
        df: Strategy bar DataFrame
        signal_file: Path to signals.jsonl
    
    Returns:
        Dict with alignment metrics
    """
    signals = {}
    with open(signal_file, "r") as f:
        for line in f:
            record = json.loads(line)
            signals[int(record["timestamp"])] = float(record["probability"])
    
    bar_timestamps = [int(pd.Timestamp(idx).timestamp()) for idx in df.index]
    
    matched = sum(1 for ts in bar_timestamps if ts in signals)
    missing = [ts for ts in bar_timestamps if ts not in signals]
    
    return {
        "total_bars": len(bar_timestamps),
        "matched": matched,
        "missing": len(missing),
        "coverage": matched / len(bar_timestamps) if bar_timestamps else 0,
        "missing_timestamps": missing[:10],  # First 10 for debugging
        "signal_count": len(signals),
        "first_signal_ts": min(signals.keys()) if signals else None,
        "last_signal_ts": max(signals.keys()) if signals else None,
    }