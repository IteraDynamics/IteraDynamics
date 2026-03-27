"""
Tests for portfolio_geometry_validation sliced run reporting (--start/--end).

Verifies:
- Output start/end columns reflect slice boundaries (not full 2019-2025).
- period_days and total_return_pct are present; period_days ≈ actual slice duration.
- CAGR is consistent with total return annualized over the slice period.
- When slice does not overlap crash_window/post_crash, those rows are omitted.
- Manifest records run_start_date, run_end_date, run_bars.
"""

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[3]
_EXPERIMENTS = Path(__file__).resolve().parents[1]
_RUNTIME_ARGUS = _REPO_ROOT / "runtime" / "argus"
for _p in (_EXPERIMENTS, _RUNTIME_ARGUS, _REPO_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from cost_regimes import COST_REGIME_CUSTOM, resolve_cost_params
from portfolio_geometry_validation import RunConfig, run


def _make_ohlcv_csv(path: Path, n_rows: int, base_ts: str = "2019-01-01 00:00:00") -> None:
    """Write a minimal OHLCV CSV with n_rows hourly bars."""
    df = pd.DataFrame(
        {
            "Timestamp": pd.date_range(base_ts, periods=n_rows, freq="h", tz="UTC"),
            "Open": 100.0,
            "High": 101.0,
            "Low": 99.0,
            "Close": 100.0,
        }
    )
    df["Volume"] = 1000.0
    df.to_csv(path, index=False)


@pytest.fixture
def synthetic_slice_data(tmp_path):
    """BTC/ETH with hourly bars from 2025-11-01 so a Dec 1–26 slice has enough history + run bars."""
    # Need lookback (100) + run bars (26 days * 24 = 624). Nov 1 -> Dec 26 = 55 days = 55*24 = 1320 bars.
    n = 1500
    base = "2025-11-01 00:00:00"
    btc = tmp_path / "btc_slice.csv"
    eth = tmp_path / "eth_slice.csv"
    _make_ohlcv_csv(btc, n, base)
    _make_ohlcv_csv(eth, n, base)
    return btc, eth


def test_sliced_run_output_start_end_reflect_slice(synthetic_slice_data, tmp_path):
    """With --start/--end, output CSV start/end columns are the slice boundaries."""
    btc_path, eth_path = synthetic_slice_data
    fee_bps, slippage_bps = resolve_cost_params(
        cost_regime=COST_REGIME_CUSTOM, fee_bps_cli=10, slippage_bps_cli=5
    )
    cfg = RunConfig(
        mode="net",
        fee_bps=fee_bps,
        slippage_bps=slippage_bps,
        cost_regime=COST_REGIME_CUSTOM,
        out_dir=tmp_path,
        out_csv=tmp_path / "sliced.csv",
        env_file=None,
        btc_data_file=btc_path,
        eth_data_file=eth_path,
        debug_trace_max_bars=None,
        max_bars=None,
        start_date="2025-12-01",
        end_date="2025-12-26",
        initial_equity=10000.0,
        output_suffix="sliced_test",
    )
    out = run(cfg)
    assert out is not None and len(out) > 0
    # All rows should show slice boundaries, not 2019-01-01 / 2025-12-30
    assert (out["start"] == "2025-12-01").all(), "start column should be slice start"
    assert (out["end"] == "2025-12-26").all(), "end column should be slice end"


def test_sliced_run_has_period_days_and_total_return_pct(synthetic_slice_data, tmp_path):
    """Output includes period_days and total_return_pct; period_days ≈ slice duration in days."""
    btc_path, eth_path = synthetic_slice_data
    fee_bps, slippage_bps = resolve_cost_params(
        cost_regime=COST_REGIME_CUSTOM, fee_bps_cli=10, slippage_bps_cli=5
    )
    cfg = RunConfig(
        mode="net",
        fee_bps=fee_bps,
        slippage_bps=slippage_bps,
        cost_regime=COST_REGIME_CUSTOM,
        out_dir=tmp_path,
        out_csv=tmp_path / "sliced2.csv",
        env_file=None,
        btc_data_file=btc_path,
        eth_data_file=eth_path,
        debug_trace_max_bars=None,
        max_bars=None,
        start_date="2025-12-01",
        end_date="2025-12-26",
        initial_equity=10000.0,
        output_suffix="sliced_test2",
    )
    out = run(cfg)
    assert "period_days" in out.columns and "total_return_pct" in out.columns
    # Dec 1 to Dec 26 ≈ 25 days
    period_days = out["period_days"].dropna()
    assert len(period_days) > 0
    assert period_days.iloc[0] >= 24 and period_days.iloc[0] <= 27, (
        f"period_days should be ~25 for Dec 1–26 slice, got {period_days.iloc[0]}"
    )


def test_sliced_run_cagr_consistent_with_total_return(synthetic_slice_data, tmp_path):
    """CAGR (annualized) should match total_return_pct annualized over period_days."""
    btc_path, eth_path = synthetic_slice_data
    fee_bps, slippage_bps = resolve_cost_params(
        cost_regime=COST_REGIME_CUSTOM, fee_bps_cli=10, slippage_bps_cli=5
    )
    cfg = RunConfig(
        mode="net",
        fee_bps=fee_bps,
        slippage_bps=slippage_bps,
        cost_regime=COST_REGIME_CUSTOM,
        out_dir=tmp_path,
        out_csv=tmp_path / "sliced3.csv",
        env_file=None,
        btc_data_file=btc_path,
        eth_data_file=eth_path,
        debug_trace_max_bars=None,
        max_bars=None,
        start_date="2025-12-01",
        end_date="2025-12-26",
        initial_equity=10000.0,
        output_suffix="sliced_test3",
    )
    out = run(cfg)
    # For each row: (1 + total_return_pct/100)^(365.25/period_days) - 1 ≈ CAGR
    for _, row in out.iterrows():
        pd_val = row.get("period_days")
        tr_val = row.get("total_return_pct")
        cagr_val = row.get("CAGR")
        if pd.isna(pd_val) or pd_val <= 0 or pd.isna(tr_val) or pd.isna(cagr_val):
            continue
        years = pd_val / 365.25
        if years <= 0:
            continue
        implied_cagr = (1.0 + float(tr_val) / 100.0) ** (1.0 / years) - 1.0
        # Allow relative tolerance (CAGR is from same formula; small float diff ok)
        assert abs(implied_cagr - float(cagr_val)) < 0.01 or (abs(cagr_val) < 1e-6 and abs(implied_cagr) < 1e-6), (
            f"CAGR {cagr_val} vs total_return_pct annualized {implied_cagr} (period_days={pd_val})"
        )


def test_sliced_run_omits_non_overlapping_windows(synthetic_slice_data, tmp_path):
    """When slice is Dec 2025, crash_window (2021–2022) should not appear in output."""
    btc_path, eth_path = synthetic_slice_data
    fee_bps, slippage_bps = resolve_cost_params(
        cost_regime=COST_REGIME_CUSTOM, fee_bps_cli=10, slippage_bps_cli=5
    )
    cfg = RunConfig(
        mode="net",
        fee_bps=fee_bps,
        slippage_bps=slippage_bps,
        cost_regime=COST_REGIME_CUSTOM,
        out_dir=tmp_path,
        out_csv=tmp_path / "sliced4.csv",
        env_file=None,
        btc_data_file=btc_path,
        eth_data_file=eth_path,
        debug_trace_max_bars=None,
        max_bars=None,
        start_date="2025-12-01",
        end_date="2025-12-26",
        initial_equity=10000.0,
        output_suffix="sliced_test4",
    )
    out = run(cfg)
    windows = set(out["window"].tolist())
    assert "crash_window" not in windows, "Dec 2025 slice should not emit crash_window (no overlap)"


def test_sliced_run_manifest_has_slice_and_bars(synthetic_slice_data, tmp_path):
    """Manifest records run_start_date, run_end_date, run_bars."""
    btc_path, eth_path = synthetic_slice_data
    fee_bps, slippage_bps = resolve_cost_params(
        cost_regime=COST_REGIME_CUSTOM, fee_bps_cli=10, slippage_bps_cli=5
    )
    cfg = RunConfig(
        mode="net",
        fee_bps=fee_bps,
        slippage_bps=slippage_bps,
        cost_regime=COST_REGIME_CUSTOM,
        out_dir=tmp_path,
        out_csv=tmp_path / "sliced5.csv",
        env_file=None,
        btc_data_file=btc_path,
        eth_data_file=eth_path,
        debug_trace_max_bars=None,
        max_bars=None,
        start_date="2025-12-01",
        end_date="2025-12-26",
        initial_equity=10000.0,
        output_suffix="sliced_manifest",
    )
    run(cfg)
    manifest_path = tmp_path / "portfolio_geometry_run_manifest__sliced_manifest.json"
    assert manifest_path.exists()
    with open(manifest_path) as f:
        manifest = json.load(f)
    assert manifest.get("run_start_date") == "2025-12-01"
    assert manifest.get("run_end_date") == "2025-12-26"
    assert "run_bars" in manifest
    assert isinstance(manifest["run_bars"], int)
    assert manifest["run_bars"] > 0
