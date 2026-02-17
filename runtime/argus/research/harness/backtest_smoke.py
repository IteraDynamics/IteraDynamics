"""
Backtest Smoke Test
===================

End-to-end acceptance test for backtest_runner with Layer 2 strategies.

Validates:
1. Runner executes without exceptions
2. Metrics keys exist
3. max_drawdown is between 0 and 1
4. avg_exposure is between 0 and 1
5. Determinism: running twice yields identical metrics

Run from repo root:
    python -c "import sys; sys.path.insert(0, r'./runtime/argus'); from research.harness.backtest_smoke import main; main()"

Or directly:
    cd runtime/argus && python -m research.harness.backtest_smoke
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import numpy as np


def _setup_path() -> Path:
    """Ensure runtime/argus is on sys.path. Returns argus_dir."""
    this_file = Path(__file__).resolve()
    argus_dir = this_file.parent.parent.parent

    if argus_dir.name == "argus" and (argus_dir / "research").exists():
        if str(argus_dir) not in sys.path:
            sys.path.insert(0, str(argus_dir))
        return argus_dir

    for candidate in [Path.cwd() / "runtime" / "argus", Path.cwd()]:
        if (candidate / "research").exists():
            if str(candidate) not in sys.path:
                sys.path.insert(0, str(candidate))
            return candidate

    return Path.cwd()


def _load_flight_recorder(argus_dir: Path) -> Optional[pd.DataFrame]:
    """Try to load flight_recorder.csv."""
    for candidate in [
        argus_dir / "flight_recorder.csv",
        Path.cwd() / "runtime" / "argus" / "flight_recorder.csv",
        Path.cwd() / "flight_recorder.csv",
    ]:
        if candidate.exists():
            try:
                df = pd.read_csv(candidate)
                if len(df) >= 100:
                    # Parse Timestamp column
                    if "Timestamp" in df.columns:
                        df["Timestamp"] = pd.to_datetime(df["Timestamp"], utc=True, errors="coerce")
                        df = df.dropna(subset=["Timestamp"]).reset_index(drop=True)
                    return df
            except Exception:
                pass
    return None


def _generate_synthetic_ohlcv(n_rows: int = 600, base_price: float = 50000.0) -> pd.DataFrame:
    """Generate deterministic synthetic OHLCV data."""
    np.random.seed(42)

    timestamps = pd.date_range(
        start="2025-01-01 00:00:00",
        periods=n_rows,
        freq="1h",
        tz="UTC",
    )

    # Random walk with slight upward drift
    returns = np.random.normal(0.0001, 0.008, n_rows)
    close = base_price * np.cumprod(1 + returns)

    high = close * (1 + np.abs(np.random.normal(0, 0.004, n_rows)))
    low = close * (1 - np.abs(np.random.normal(0, 0.004, n_rows)))
    open_ = close * (1 + np.random.normal(0, 0.002, n_rows))

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


def _compare_metrics(m1: Dict[str, Any], m2: Dict[str, Any], tolerance: float = 1e-9) -> List[str]:
    """Compare two metric dictionaries. Returns list of differences."""
    errors = []

    for key in m1:
        if key not in m2:
            errors.append(f"Missing key in m2: {key}")
            continue

        v1, v2 = m1[key], m2[key]

        if isinstance(v1, float) and isinstance(v2, float):
            if abs(v1 - v2) > tolerance:
                errors.append(f"Mismatch {key}: {v1} != {v2}")
        elif v1 != v2:
            errors.append(f"Mismatch {key}: {v1} != {v2}")

    for key in m2:
        if key not in m1:
            errors.append(f"Extra key in m2: {key}")

    return errors


def main() -> bool:
    """
    Run backtest smoke tests.

    Returns True if all tests pass, False otherwise.
    """
    print("=" * 60)
    print("BACKTEST SMOKE TEST")
    print("=" * 60)

    # Setup path
    argus_dir = _setup_path()
    print(f"\n[1] Path Setup")
    print(f"    argus_dir: {argus_dir}")

    # Import backtest runner
    print(f"\n[2] Import Test")
    try:
        from research.harness.backtest_runner import (
            run_backtest,
            load_strategy_func,
            load_flight_recorder,
        )
        print(f"    backtest_runner imported successfully")
    except ImportError as e:
        print(f"    FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Load or generate data
    print(f"\n[3] Data Loading")
    df = _load_flight_recorder(argus_dir)
    if df is not None:
        print(f"    Loaded real data: {len(df)} rows")
        data_source = "real"
    else:
        df = _generate_synthetic_ohlcv(600)
        print(f"    Generated synthetic data: {len(df)} rows")
        data_source = "synthetic"

    # Truncate to last 500 rows for speed
    df = df.tail(500).copy().reset_index(drop=True)
    print(f"    Using: {len(df)} rows for smoke test")

    # Load strategy
    print(f"\n[4] Strategy Loading")
    try:
        # Use stub strategy for reliable testing
        strategy_func = load_strategy_func(
            "research.strategies.sg_stub_strategy",
            "generate_intent"
        )
        print(f"    Loaded: research.strategies.sg_stub_strategy.generate_intent")
    except Exception as e:
        print(f"    FAILED: {e}")
        return False

    # Run backtest #1
    print(f"\n[5] First Backtest Run")
    try:
        equity_df1, metrics1 = run_backtest(
            df,
            strategy_func,
            lookback=100,
            initial_equity=10000.0,
            fee_bps=10.0,
            slippage_bps=5.0,
            closed_only=True,
        )
        print(f"    Completed: {metrics1['bars']} bars")
    except Exception as e:
        print(f"    FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Validate metrics keys
    print(f"\n[6] Metrics Validation")
    required_keys = [
        "total_return", "cagr", "max_drawdown", "calmar", "sortino",
        "avg_exposure", "time_in_market", "final_equity", "bars", "years"
    ]
    missing_keys = [k for k in required_keys if k not in metrics1]
    if missing_keys:
        print(f"    FAILED: Missing keys: {missing_keys}")
        return False
    print(f"    All required keys present: {len(required_keys)}")

    # Validate max_drawdown range
    print(f"\n[7] Max Drawdown Validation")
    max_dd = metrics1["max_drawdown"]
    if not (0.0 <= max_dd <= 1.0):
        print(f"    FAILED: max_drawdown={max_dd} not in [0, 1]")
        return False
    print(f"    max_drawdown={max_dd:.4f} (valid)")

    # Validate avg_exposure range
    print(f"\n[8] Avg Exposure Validation")
    avg_expo = metrics1["avg_exposure"]
    if not (0.0 <= avg_expo <= 1.0):
        print(f"    FAILED: avg_exposure={avg_expo} not in [0, 1]")
        return False
    print(f"    avg_exposure={avg_expo:.4f} (valid)")

    # Validate time_in_market range
    print(f"\n[9] Time In Market Validation")
    tim = metrics1["time_in_market"]
    if not (0.0 <= tim <= 1.0):
        print(f"    FAILED: time_in_market={tim} not in [0, 1]")
        return False
    print(f"    time_in_market={tim:.4f} (valid)")

    # Run backtest #2 for determinism
    print(f"\n[10] Determinism Test (Second Run)")
    try:
        equity_df2, metrics2 = run_backtest(
            df,
            strategy_func,
            lookback=100,
            initial_equity=10000.0,
            fee_bps=10.0,
            slippage_bps=5.0,
            closed_only=True,
        )
        print(f"    Completed: {metrics2['bars']} bars")
    except Exception as e:
        print(f"    FAILED: {e}")
        return False

    # Compare metrics for determinism
    print(f"\n[11] Determinism Comparison")
    diff_errors = _compare_metrics(metrics1, metrics2, tolerance=1e-9)
    if diff_errors:
        for err in diff_errors:
            print(f"    ERROR: {err}")
        return False
    print(f"    PASS: Metrics are identical across runs")

    # Test with a real strategy if available
    print(f"\n[12] Real Strategy Test (sg_core_exposure_v1)")
    try:
        real_strategy_func = load_strategy_func(
            "research.strategies.sg_core_exposure_v1",
            "generate_intent"
        )
        print(f"    Loaded: research.strategies.sg_core_exposure_v1.generate_intent")

        equity_df_real, metrics_real = run_backtest(
            df,
            real_strategy_func,
            lookback=100,
            initial_equity=10000.0,
            fee_bps=10.0,
            slippage_bps=5.0,
            closed_only=True,
        )
        print(f"    Completed: {metrics_real['bars']} bars")
        print(f"    total_return: {metrics_real['total_return'] * 100:.2f}%")
        print(f"    max_drawdown: {metrics_real['max_drawdown'] * 100:.2f}%")
        print(f"    avg_exposure: {metrics_real['avg_exposure'] * 100:.2f}%")
    except Exception as e:
        print(f"    WARNING: Real strategy test failed: {e}")
        print(f"    (This may be expected if regime engine has issues)")
        # Don't fail the smoke test for this - stub strategy is the contract test

    # Summary
    print(f"\n" + "=" * 60)
    print(f"ALL TESTS PASSED")
    print("=" * 60)
    print(f"\nSummary:")
    print(f"  - Data source: {data_source}")
    print(f"  - Rows used: {len(df)}")
    print(f"  - Stub strategy metrics:")
    print(f"      total_return: {metrics1['total_return'] * 100:.2f}%")
    print(f"      max_drawdown: {metrics1['max_drawdown'] * 100:.2f}%")
    print(f"      avg_exposure: {metrics1['avg_exposure'] * 100:.2f}%")
    print(f"      time_in_market: {metrics1['time_in_market'] * 100:.2f}%")
    print(f"  - Determinism: VERIFIED")

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

