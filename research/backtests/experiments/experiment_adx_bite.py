"""
ADX Bite Experiment
===================

This script proves that ADX gating changes behavior when set to different modes.

Run A: ADX_MODE=hard, SGRT_ADX_REL_MIN=1.5 (strict)
Run B: ADX_MODE=off (disabled)

Compares trade count, CAGR, MDD, and total return.
Writes outputs to output/ with distinct filenames.

Usage:
    python research/backtests/experiments/experiment_adx_bite.py

The script inherits other env vars (SGRT_TREND_MIN, etc.) from the environment,
so you can set those before running if needed.
"""

from __future__ import annotations

import os
import sys
from contextlib import contextmanager
from datetime import datetime
from typing import Any, Dict

import pandas as pd

# Ensure repo root is on sys.path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from research.backtests.experiments.backtest_sg_regime_trend_v1 import (
    BTParams,
    _load_data,
    run_backtest,
)

OUT_DIR = os.path.join(REPO_ROOT, "output")
os.makedirs(OUT_DIR, exist_ok=True)


@contextmanager
def _temp_env(overrides: Dict[str, str]):
    """Temporarily override env vars."""
    prev: Dict[str, Any] = {}
    try:
        for k, v in overrides.items():
            prev[k] = os.environ.get(k, None)
            os.environ[k] = str(v)
        yield
    finally:
        for k, old in prev.items():
            if old is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = str(old)


def _env_float(name: str, default: float) -> float:
    v = os.environ.get(name, "").strip()
    if not v:
        return default
    try:
        return float(v)
    except Exception:
        return default


def _env_int(name: str, default: int) -> int:
    v = os.environ.get(name, "").strip()
    if not v:
        return default
    try:
        return int(float(v))
    except Exception:
        return default


def _env_str(name: str, default: str) -> str:
    v = os.environ.get(name, "").strip()
    return v if v else default


def _build_base_params(adx_mode: str, adx_rel_min: float) -> BTParams:
    """Build BTParams from current env, overriding ADX mode and rel_min."""
    return BTParams(
        trend_min=_env_float("SGRT_TREND_MIN", 0.30),
        vol_min=_env_float("SGRT_VOL_MIN", 0.003),
        vol_max=_env_float("SGRT_VOL_MAX", 0.016),
        horizon_h=_env_int("SGRT_HORIZON_H", 48),
        vol_exit_max=_env_float("SGRT_VOL_EXIT_MAX", 0.030),
        reentry_cooldown_h=_env_float("PRIME_REENTRY_COOLDOWN_H", 24.0),
        exit_min_hold_h=_env_float("PRIME_EXIT_MIN_HOLD_H", 24.0),
        adx_min=_env_float("SGRT_ADX_MIN", 0.0),
        adx_rel_win=_env_int("SGRT_ADX_REL_WIN", 168),
        adx_rel_min=adx_rel_min,
        adx_mode=adx_mode,
        adx_soft_penalty=_env_float("SGRT_ADX_SOFT_PENALTY", 0.10),
        vol_win=_env_int("SGRT_VOL_WIN", 24),
        trend_win=_env_int("SGRT_TREND_WIN", 24),
        spike_ratio_max=_env_float("SGRT_SPIKE_RATIO_MAX", 1.8),
    )


def run_experiment(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Run the ADX bite experiment:
      - Run A: ADX hard mode, rel_min=1.5
      - Run B: ADX off mode
    """
    results = {}

    # ========== RUN A: ADX HARD (strict) ==========
    print("\n" + "=" * 60)
    print("RUN A: ADX MODE = hard, ADX_REL_MIN = 1.5 (strict)")
    print("=" * 60)

    with _temp_env({"SGRT_ADX_MODE": "hard", "SGRT_ADX_REL_MIN": "1.5"}):
        params_a = _build_base_params(adx_mode="hard", adx_rel_min=1.5)
        print(f"  adx_mode={params_a.adx_mode}, adx_rel_min={params_a.adx_rel_min}")
        curve_a, trades_a, summary_a = run_backtest(df, params_a)

    results["A_hard"] = {
        "adx_mode": "hard",
        "adx_rel_min": 1.5,
        "trades": summary_a["trades"],
        "total_return": summary_a["total_return"],
        "cagr": summary_a["cagr"],
        "max_drawdown": summary_a["max_drawdown"],
        "enter_intents": summary_a["seen_enter_intents"],
        "exit_intents": summary_a["seen_exit_intents"],
        "hold_intents": summary_a["seen_hold_intents"],
    }

    # ========== RUN B: ADX OFF (disabled) ==========
    print("\n" + "=" * 60)
    print("RUN B: ADX MODE = off (disabled)")
    print("=" * 60)

    with _temp_env({"SGRT_ADX_MODE": "off", "SGRT_ADX_REL_MIN": "0"}):
        params_b = _build_base_params(adx_mode="off", adx_rel_min=0.0)
        print(f"  adx_mode={params_b.adx_mode}, adx_rel_min={params_b.adx_rel_min}")
        curve_b, trades_b, summary_b = run_backtest(df, params_b)

    results["B_off"] = {
        "adx_mode": "off",
        "adx_rel_min": 0.0,
        "trades": summary_b["trades"],
        "total_return": summary_b["total_return"],
        "cagr": summary_b["cagr"],
        "max_drawdown": summary_b["max_drawdown"],
        "enter_intents": summary_b["seen_enter_intents"],
        "exit_intents": summary_b["seen_exit_intents"],
        "hold_intents": summary_b["seen_hold_intents"],
    }

    # ========== RUN C (optional): ADX SOFT for comparison ==========
    print("\n" + "=" * 60)
    print("RUN C: ADX MODE = soft, ADX_REL_MIN = 1.5 (soft penalty)")
    print("=" * 60)

    with _temp_env({"SGRT_ADX_MODE": "soft", "SGRT_ADX_REL_MIN": "1.5"}):
        params_c = _build_base_params(adx_mode="soft", adx_rel_min=1.5)
        print(f"  adx_mode={params_c.adx_mode}, adx_rel_min={params_c.adx_rel_min}")
        curve_c, trades_c, summary_c = run_backtest(df, params_c)

    results["C_soft"] = {
        "adx_mode": "soft",
        "adx_rel_min": 1.5,
        "trades": summary_c["trades"],
        "total_return": summary_c["total_return"],
        "cagr": summary_c["cagr"],
        "max_drawdown": summary_c["max_drawdown"],
        "enter_intents": summary_c["seen_enter_intents"],
        "exit_intents": summary_c["seen_exit_intents"],
        "hold_intents": summary_c["seen_hold_intents"],
    }

    # Save curves with distinct filenames
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    curve_a.to_csv(os.path.join(OUT_DIR, f"adx_bite_A_hard_{ts}.csv"), index=False)
    curve_b.to_csv(os.path.join(OUT_DIR, f"adx_bite_B_off_{ts}.csv"), index=False)
    curve_c.to_csv(os.path.join(OUT_DIR, f"adx_bite_C_soft_{ts}.csv"), index=False)
    trades_a.to_csv(os.path.join(OUT_DIR, f"adx_bite_A_hard_trades_{ts}.csv"), index=False)
    trades_b.to_csv(os.path.join(OUT_DIR, f"adx_bite_B_off_trades_{ts}.csv"), index=False)
    trades_c.to_csv(os.path.join(OUT_DIR, f"adx_bite_C_soft_trades_{ts}.csv"), index=False)

    return results


def print_comparison(results: Dict[str, Any]) -> None:
    """Print a compact diff summary."""
    print("\n")
    print("=" * 70)
    print("ADX BITE EXPERIMENT - COMPARISON SUMMARY")
    print("=" * 70)

    a = results["A_hard"]
    b = results["B_off"]
    c = results["C_soft"]

    def fmt_pct(v: float) -> str:
        return f"{v * 100:+.2f}%" if v >= 0 else f"{v * 100:.2f}%"

    print(f"\n{'Metric':<25} {'A (hard)':<18} {'B (off)':<18} {'C (soft)':<18}")
    print("-" * 70)
    print(f"{'adx_mode':<25} {a['adx_mode']:<18} {b['adx_mode']:<18} {c['adx_mode']:<18}")
    print(f"{'adx_rel_min':<25} {a['adx_rel_min']:<18.2f} {b['adx_rel_min']:<18.2f} {c['adx_rel_min']:<18.2f}")
    print(f"{'Trades':<25} {a['trades']:<18} {b['trades']:<18} {c['trades']:<18}")
    print(f"{'ENTER intents':<25} {a['enter_intents']:<18,} {b['enter_intents']:<18,} {c['enter_intents']:<18,}")
    print(f"{'Total Return':<25} {fmt_pct(a['total_return']):<18} {fmt_pct(b['total_return']):<18} {fmt_pct(c['total_return']):<18}")
    print(f"{'CAGR':<25} {fmt_pct(a['cagr']):<18} {fmt_pct(b['cagr']):<18} {fmt_pct(c['cagr']):<18}")
    print(f"{'Max Drawdown':<25} {fmt_pct(a['max_drawdown']):<18} {fmt_pct(b['max_drawdown']):<18} {fmt_pct(c['max_drawdown']):<18}")

    print("\n" + "-" * 70)
    print("INTERPRETATION:")

    if a["trades"] == b["trades"] and a["cagr"] == b["cagr"]:
        print("  WARNING: A (hard) and B (off) have IDENTICAL results.")
        print("  Possible reasons:")
        print("    1. adx_rel values in your data always pass the 1.5 threshold")
        print("    2. ADX gate is only for entry; most entries pass anyway")
        print("    3. Config not propagating correctly (ghost config)")
        print("  -> Run debug_sg_regime_trend_v1_intents.py with SGRT_ADX_MODE=hard to inspect.")
    else:
        trade_diff = b["trades"] - a["trades"]
        cagr_diff = b["cagr"] - a["cagr"]
        mdd_diff = b["max_drawdown"] - a["max_drawdown"]
        print(f"  ADX gating IS biting! Differences (B_off - A_hard):")
        print(f"    Trades:       {trade_diff:+d}")
        print(f"    CAGR:         {cagr_diff * 100:+.2f}%")
        print(f"    Max Drawdown: {mdd_diff * 100:+.2f}%")

        if a["trades"] < b["trades"]:
            print("  -> Hard ADX mode is blocking entries (fewer trades than off mode).")
        if c["trades"] == b["trades"]:
            print("  -> Soft mode with same threshold has SAME trade count as off (expected: penalty only).")

    print("=" * 70 + "\n")


def main() -> None:
    print("[ADX_BITE] Loading data...")
    df = _load_data()
    print(f"[ADX_BITE] Data rows: {len(df):,}")
    print(f"[ADX_BITE] Range: {df['Timestamp'].iloc[0]} -> {df['Timestamp'].iloc[-1]}")

    # Print current env snapshot for reproducibility
    print("\n[ADX_BITE] Current env snapshot (SGRT_* and PRIME_*):")
    keys = sorted([k for k in os.environ.keys() if k.startswith("SGRT_") or k.startswith("PRIME_")])
    for k in keys:
        print(f"  {k}={os.environ.get(k)}")

    results = run_experiment(df)
    print_comparison(results)

    # Save summary
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_df = pd.DataFrame([
        results["A_hard"],
        results["B_off"],
        results["C_soft"],
    ])
    summary_path = os.path.join(OUT_DIR, f"adx_bite_summary_{ts}.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"[ADX_BITE] Wrote summary: {summary_path}")


if __name__ == "__main__":
    main()

