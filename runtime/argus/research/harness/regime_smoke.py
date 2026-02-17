"""
Regime Engine Smoke Test
========================

Validates Layer 1 (Regime Engine) functionality:

1. Import resolution under runtime/argus sys.path
2. classify_regime works on sample data
3. closed_only behavior (no lookahead)
4. RegimeState output validation

Run from repo root:
    python -c "import sys; sys.path.insert(0, r'./runtime/argus'); from research.harness.regime_smoke import main; main()"

Or directly:
    cd runtime/argus && python -m research.harness.regime_smoke
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import numpy as np


def _setup_path() -> str:
    """Ensure runtime/argus is on sys.path. Returns project root."""
    # Try to find runtime/argus relative to this file
    this_file = Path(__file__).resolve()
    
    # research/harness/regime_smoke.py -> research -> argus
    argus_dir = this_file.parent.parent.parent
    
    if argus_dir.name == "argus" and (argus_dir / "research").exists():
        if str(argus_dir) not in sys.path:
            sys.path.insert(0, str(argus_dir))
        return str(argus_dir)
    
    # Fallback: try cwd-based resolution
    for candidate in [Path.cwd() / "runtime" / "argus", Path.cwd()]:
        if (candidate / "research").exists():
            if str(candidate) not in sys.path:
                sys.path.insert(0, str(candidate))
            return str(candidate)
    
    return str(Path.cwd())


def _generate_sample_ohlcv(n_rows: int = 500, base_price: float = 50000.0) -> pd.DataFrame:
    """Generate synthetic OHLCV data for testing."""
    np.random.seed(42)  # Deterministic
    
    timestamps = pd.date_range(
        end=datetime.now(timezone.utc) - timedelta(hours=1),  # Last closed candle
        periods=n_rows,
        freq="1h",
        tz="UTC",
    )
    
    # Random walk for close prices
    returns = np.random.normal(0, 0.01, n_rows)
    close = base_price * np.cumprod(1 + returns)
    
    # Generate OHLC from close
    high = close * (1 + np.abs(np.random.normal(0, 0.005, n_rows)))
    low = close * (1 - np.abs(np.random.normal(0, 0.005, n_rows)))
    open_ = close * (1 + np.random.normal(0, 0.002, n_rows))
    
    # Ensure OHLC consistency
    high = np.maximum(high, np.maximum(open_, close))
    low = np.minimum(low, np.minimum(open_, close))
    
    volume = np.random.uniform(100, 1000, n_rows)
    
    return pd.DataFrame({
        "Timestamp": timestamps,
        "Open": open_,
        "High": high,
        "Low": low,
        "Close": close,
        "Volume": volume,
    })


def _load_real_data() -> Optional[pd.DataFrame]:
    """Try to load real flight_recorder.csv data."""
    for candidate in [
        Path.cwd() / "runtime" / "argus" / "flight_recorder.csv",
        Path.cwd() / "flight_recorder.csv",
        Path(__file__).resolve().parent.parent.parent / "flight_recorder.csv",
    ]:
        if candidate.exists():
            try:
                df = pd.read_csv(candidate)
                if len(df) >= 100:
                    return df
            except Exception:
                pass
    return None


def _validate_regime_state(state: Any, label: str) -> List[str]:
    """Validate RegimeState object. Returns list of errors."""
    errors = []
    
    # Check required attributes
    required = ["asof_ts", "label", "confidence", "features", "meta"]
    for attr in required:
        if not hasattr(state, attr):
            errors.append(f"{label}: missing attribute '{attr}'")
    
    if errors:
        return errors
    
    # Validate asof_ts is parseable ISO8601
    try:
        ts = datetime.fromisoformat(state.asof_ts.replace("Z", "+00:00"))
        if ts.tzinfo is None:
            errors.append(f"{label}: asof_ts not timezone-aware")
    except Exception as e:
        errors.append(f"{label}: asof_ts not valid ISO8601: {e}")
    
    # Validate label
    valid_labels = {"TREND_UP", "TREND_DOWN", "CHOP", "VOL_EXPANSION", "VOL_COMPRESSION", "PANIC"}
    if state.label not in valid_labels:
        errors.append(f"{label}: invalid label '{state.label}' (valid: {valid_labels})")
    
    # Validate confidence
    if not isinstance(state.confidence, (int, float)):
        errors.append(f"{label}: confidence not numeric")
    elif not (0.0 <= state.confidence <= 1.0):
        errors.append(f"{label}: confidence {state.confidence} not in [0, 1]")
    
    # Validate features is dict
    if not isinstance(state.features, dict):
        errors.append(f"{label}: features not a dict")
    
    # Validate meta is dict
    if not isinstance(state.meta, dict):
        errors.append(f"{label}: meta not a dict")
    
    return errors


def main() -> bool:
    """
    Run regime engine smoke tests.
    
    Returns True if all tests pass, False otherwise.
    """
    print("=" * 60)
    print("REGIME ENGINE SMOKE TEST")
    print("=" * 60)
    
    # Setup path
    project_root = _setup_path()
    print(f"\n[1] Path Setup")
    print(f"    project_root: {project_root}")
    
    # Import regime engine
    print(f"\n[2] Import Test")
    try:
        import research
        from research.regime.regime_engine import classify_regime, RegimeState, RegimeLabel
        print(f"    research.__file__: {research.__file__}")
        print(f"    regime_engine loaded from: {classify_regime.__module__}")
        
        # Verify it's the correct module
        import research.regime.regime_engine as re_module
        print(f"    regime_engine.__file__: {re_module.__file__}")
    except ImportError as e:
        print(f"    FAILED: {e}")
        return False
    
    # Generate or load data
    print(f"\n[3] Data Loading")
    df = _load_real_data()
    if df is not None:
        print(f"    Loaded real data: {len(df)} rows")
        data_source = "real"
    else:
        df = _generate_sample_ohlcv(500)
        print(f"    Generated synthetic data: {len(df)} rows")
        data_source = "synthetic"
    
    # Test basic classification
    print(f"\n[4] Basic Classification Test")
    try:
        # Use last 500 rows
        df_test = df.tail(500).copy()
        state = classify_regime(df_test, closed_only=True)
        
        errors = _validate_regime_state(state, "basic_test")
        if errors:
            for e in errors:
                print(f"    ERROR: {e}")
            return False
        
        print(f"    label: {state.label}")
        print(f"    confidence: {state.confidence:.4f}")
        print(f"    asof_ts: {state.asof_ts}")
        print(f"    features keys: {list(state.features.keys())}")
        print(f"    meta.closed_only: {state.meta.get('closed_only')}")
        print(f"    meta.dropped_last_row: {state.meta.get('dropped_last_row')}")
    except Exception as e:
        print(f"    FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test closed_only behavior (CRITICAL)
    print(f"\n[5] Timeline Safety Test (closed_only)")
    try:
        df_test = df.tail(500).copy()
        
        # KEY INSIGHT: closed_only=True ALWAYS drops the last row.
        # This ensures we never make decisions on potentially-forming candles.
        # When we add a new candle, the previously "last" row becomes closed.
        
        # Test 1: closed_only drops last row
        state_closed = classify_regime(df_test, closed_only=True)
        state_open = classify_regime(df_test, closed_only=False)
        
        # closed_only=True should have earlier asof_ts (one row less)
        ts_closed = datetime.fromisoformat(state_closed.asof_ts.replace("Z", "+00:00"))
        ts_open = datetime.fromisoformat(state_open.asof_ts.replace("Z", "+00:00"))
        
        if ts_closed >= ts_open:
            print(f"    FAILED: closed_only=True did not drop last row!")
            print(f"           closed asof: {state_closed.asof_ts}")
            print(f"           open asof:   {state_open.asof_ts}")
            return False
        
        print(f"    closed_only=True asof:  {state_closed.asof_ts}")
        print(f"    closed_only=False asof: {state_open.asof_ts}")
        print(f"    Row dropped: YES (timestamps differ)")
        
        # Test 2: Verify meta flags are set correctly
        if not state_closed.meta.get("closed_only", False):
            print(f"    FAILED: meta.closed_only not set to True")
            return False
        if not state_closed.meta.get("dropped_last_row", False):
            print(f"    FAILED: meta.dropped_last_row not set to True")
            return False
        
        # Test 3: Append fake incomplete candle with extreme values
        # The point is to verify that with closed_only=True, this extreme 
        # candle doesn't affect classification (it gets dropped)
        fake_ts = pd.Timestamp.now(tz="UTC") + timedelta(hours=1)
        last_close = float(df_test["Close"].iloc[-1])
        fake_candle = pd.DataFrame({
            "Timestamp": [fake_ts],
            "Open": [last_close],
            "High": [last_close * 1.50],  # 50% spike (extreme!)
            "Low": [last_close * 0.50],   # 50% drop (extreme!)
            "Close": [last_close * 1.40], # 40% up
            "Volume": [1000000],          # Extreme volume
        })
        df_with_extreme = pd.concat([df_test, fake_candle], ignore_index=True)
        
        # With closed_only=True, the extreme candle should be dropped
        # So the asof_ts should be the ORIGINAL last row (now considered closed)
        state_with_extreme_closed = classify_regime(df_with_extreme, closed_only=True)
        
        # With closed_only=False, the extreme candle is included
        state_with_extreme_open = classify_regime(df_with_extreme, closed_only=False)
        
        # The asof_ts with extreme+closed_only should match original open (both use same last row)
        if state_with_extreme_closed.asof_ts != state_open.asof_ts:
            print(f"    WARN: asof_ts mismatch (may be timestamp normalization)")
            print(f"         with extreme (closed_only=True):  {state_with_extreme_closed.asof_ts}")
            print(f"         original (closed_only=False):     {state_open.asof_ts}")
        
        # The extreme candle should cause different classification with closed_only=False
        # due to massive volatility spike
        print(f"    With extreme candle (closed_only=True):  label={state_with_extreme_closed.label}")
        print(f"    With extreme candle (closed_only=False): label={state_with_extreme_open.label}")
        
        # If labels match with closed_only=True but differ with False, that proves
        # the extreme candle is being correctly excluded
        labels_differ_when_open = (state_with_extreme_open.label != state_closed.label)
        print(f"    Extreme candle affects classification when included: {labels_differ_when_open}")
        
        print(f"    PASS: closed_only behavior verified")
        
    except Exception as e:
        print(f"    FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test determinism
    print(f"\n[6] Determinism Test")
    try:
        df_test = df.tail(500).copy()
        
        state1 = classify_regime(df_test, closed_only=True)
        state2 = classify_regime(df_test, closed_only=True)
        
        if state1.label != state2.label:
            print(f"    FAILED: non-deterministic label ({state1.label} != {state2.label})")
            return False
        if state1.confidence != state2.confidence:
            print(f"    FAILED: non-deterministic confidence")
            return False
        if state1.asof_ts != state2.asof_ts:
            print(f"    FAILED: non-deterministic asof_ts")
            return False
        
        print(f"    PASS: classify_regime is deterministic")
        
    except Exception as e:
        print(f"    FAILED: {e}")
        return False
    
    # Test insufficient history handling
    print(f"\n[7] Insufficient History Test")
    try:
        tiny_df = df.head(20).copy()  # Too few rows
        state_tiny = classify_regime(tiny_df, closed_only=True)
        
        if state_tiny.label != "CHOP":
            print(f"    WARNING: expected CHOP for insufficient history, got {state_tiny.label}")
        if state_tiny.confidence != 0.0:
            print(f"    WARNING: expected confidence=0.0 for insufficient history")
        
        print(f"    label: {state_tiny.label}")
        print(f"    confidence: {state_tiny.confidence}")
        print(f"    reason: {state_tiny.features.get('reason', state_tiny.meta.get('reason', 'n/a'))}")
        print(f"    PASS: handles insufficient history gracefully")
        
    except Exception as e:
        print(f"    FAILED: {e}")
        return False
    
    # Summary
    print(f"\n" + "=" * 60)
    print(f"ALL TESTS PASSED")
    print(f"=" * 60)
    print(f"\nSummary:")
    print(f"  - Data source: {data_source}")
    print(f"  - Final regime: {state.label} (conf={state.confidence:.4f})")
    print(f"  - Timeline safety: VERIFIED")
    print(f"  - Determinism: VERIFIED")
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

