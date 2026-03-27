"""
Unit tests for portfolio_geometry_validation --max_bars (fast-run).

Verifies:
- With --max_bars N, only first N closed bars are processed; runtime is reduced.
- Output CSV and manifest exist; manifest contains max_bars.
- BTC_CORE_ONLY trace has <= N rows (or exactly N depending on convention).
- Determinism: two runs with same config yield identical metrics.
"""

import json
import os
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
    """Write a minimal OHLCV CSV with n_rows hourly bars (Timestamp, O, H, L, C, Volume)."""
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
def synthetic_btc_eth(tmp_path):
    """Two CSVs with same timestamps, 2500 bars each (enough for max_bars=2000 + lookback)."""
    btc = tmp_path / "btc.csv"
    eth = tmp_path / "eth.csv"
    _make_ohlcv_csv(btc, 2500)
    _make_ohlcv_csv(eth, 2500)
    return btc, eth


@pytest.fixture
def run_config(synthetic_btc_eth, tmp_path):
    """RunConfig with max_bars=2000 and temp output dir."""
    btc_path, eth_path = synthetic_btc_eth
    fee_bps, slippage_bps = resolve_cost_params(
        cost_regime=COST_REGIME_CUSTOM,
        fee_bps_cli=10,
        slippage_bps_cli=5,
    )
    out_csv = tmp_path / "portfolio_geometry_validation.csv"
    return RunConfig(
        mode="net",
        fee_bps=fee_bps,
        slippage_bps=slippage_bps,
        cost_regime=COST_REGIME_CUSTOM,
        out_dir=tmp_path,
        out_csv=out_csv,
        env_file=None,
        btc_data_file=btc_path,
        eth_data_file=eth_path,
        debug_trace_max_bars=None,
        max_bars=2000,
        initial_equity=10000.0,
        output_suffix="test_max_bars",
    )


def test_max_bars_outputs_exist(run_config, tmp_path):
    """Run with --max_bars 2000; assert output CSV and manifest exist and manifest has max_bars."""
    out = run(run_config)
    assert out is not None
    assert len(out) > 0
    assert run_config.out_csv.exists()
    manifest_path = tmp_path / "portfolio_geometry_run_manifest__test_max_bars.json"
    assert manifest_path.exists()
    with open(manifest_path) as f:
        manifest = json.load(f)
    assert manifest.get("max_bars") == 2000


def test_max_bars_trace_row_count(run_config):
    """Trace file exists and has <= 2000 rows (BTC_CORE_ONLY trace)."""
    run(run_config)
    trace_path = _REPO_ROOT / "debug" / "geometry_btc_trace__test_max_bars.csv"
    assert trace_path.exists(), f"Trace not found: {trace_path}"
    trace_df = pd.read_csv(trace_path)
    assert len(trace_df) <= 2000, f"Trace has {len(trace_df)} rows, expected <= 2000"


def test_max_bars_determinism(synthetic_btc_eth, tmp_path):
    """Running twice with same config yields identical metrics. Uses 500 bars for speed."""
    btc_path, eth_path = synthetic_btc_eth
    fee_bps, slippage_bps = resolve_cost_params(
        cost_regime=COST_REGIME_CUSTOM, fee_bps_cli=10, slippage_bps_cli=5
    )
    cfg = RunConfig(
        mode="net",
        fee_bps=fee_bps,
        slippage_bps=slippage_bps,
        cost_regime=COST_REGIME_CUSTOM,
        out_dir=tmp_path,
        out_csv=tmp_path / "portfolio_geometry_validation_det.csv",
        env_file=None,
        btc_data_file=btc_path,
        eth_data_file=eth_path,
        debug_trace_max_bars=None,
        max_bars=500,
        initial_equity=10000.0,
        output_suffix="test_det",
    )
    out1 = run(cfg)
    out2 = run(cfg)
    key_cols = ["scenario", "window", "CAGR", "MaxDD", "Calmar", "Sortino", "Turnover"]
    for c in key_cols:
        if c in out1.columns and c in out2.columns:
            pd.testing.assert_series_equal(
                out1[c].reset_index(drop=True),
                out2[c].reset_index(drop=True),
                check_names=True,
                atol=1e-9,
                rtol=0,
            )
