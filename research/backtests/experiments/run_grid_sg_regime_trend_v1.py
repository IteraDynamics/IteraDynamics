# ======================================================================
# FILE: research/backtests/experiments/run_grid_sg_regime_trend_v1.py
# FULL REPLACEMENT
# ======================================================================

from __future__ import annotations

import os
import sys
from contextlib import contextmanager
from typing import Any, Dict, List, Tuple

import pandas as pd


# -----------------------------------------------------------------------------
# Ensure repo root is on sys.path so `import research...` works when run as a file
# -----------------------------------------------------------------------------
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


# -----------------------------------------------------------------------------
# Imports (now safe)
# -----------------------------------------------------------------------------
from research.backtests.experiments.backtest_sg_regime_trend_v1 import (  # noqa: E402
    BTParams,
    _load_data,
    run_backtest,
)


OUT_DIR = os.path.join(REPO_ROOT, "output")
os.makedirs(OUT_DIR, exist_ok=True)


def _slice_is_oos(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    ts = df["Timestamp"]

    is_mask = (ts >= pd.Timestamp("2019-01-01", tz="UTC")) & (
        ts <= pd.Timestamp("2023-12-31 23:00:00", tz="UTC")
    )
    oos_mask = ts >= pd.Timestamp("2024-01-01", tz="UTC")

    df_is = df.loc[is_mask].reset_index(drop=True)
    df_oos = df.loc[oos_mask].reset_index(drop=True)
    return df_is, df_oos


def _score(oos_cagr: float, oos_mdd: float, oos_trades: int) -> float:
    """
    Simple holdout score:
      + reward CAGR
      - penalize drawdown (primary)
      - lightly penalize churn
    """
    dd_penalty = abs(oos_mdd) * 1.0
    churn_penalty = (oos_trades / 1000.0) * 0.03
    return float(oos_cagr - dd_penalty - churn_penalty)


def _parse_csv_floats(env_name: str, default: List[float]) -> List[float]:
    raw = os.getenv(env_name, "").strip()
    if not raw:
        return list(default)
    out: List[float] = []
    for part in raw.split(","):
        p = part.strip()
        if not p:
            continue
        try:
            out.append(float(p))
        except Exception:
            pass
    return out if out else list(default)


@contextmanager
def _temp_env(overrides: Dict[str, str]):
    """
    Temporarily set os.environ keys for the duration of the context.
    Restores previous values after.
    """
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


def _run_one(df_is: pd.DataFrame, df_oos: pd.DataFrame, params: BTParams) -> Dict[str, Any]:
    _, _, s_is = run_backtest(df_is, params)
    _, _, s_oos = run_backtest(df_oos, params)

    row = {
        "trend_min": params.trend_min,
        "vol_max": params.vol_max,
        "spike_ratio_max": params.spike_ratio_max,
        "adx_rel_min": float(os.getenv("SGRT_ADX_REL_MIN", "1.0")),
        "adx_mode": str(os.getenv("SGRT_ADX_MODE", "soft")),
        "reentry_cooldown_h": params.reentry_cooldown_h,
        "exit_min_hold_h": params.exit_min_hold_h,
        "is_cagr": s_is["cagr"],
        "is_mdd": s_is["max_drawdown"],
        "is_trades": s_is["trades"],
        "oos_cagr": s_oos["cagr"],
        "oos_mdd": s_oos["max_drawdown"],
        "oos_trades": s_oos["trades"],
    }
    row["score_oos"] = _score(row["oos_cagr"], row["oos_mdd"], int(row["oos_trades"]))
    return row


def main() -> None:
    df = _load_data()
    print(f"[GRID] Data rows total: {len(df):,}")
    df_is, df_oos = _slice_is_oos(df)
    print(
        f"[GRID] In-sample: {df_is['Timestamp'].iloc[0]} -> {df_is['Timestamp'].iloc[-1]} | rows={len(df_is):,}"
    )
    print(
        f"[GRID] Holdout  : {df_oos['Timestamp'].iloc[0]} -> {df_oos['Timestamp'].iloc[-1]} | rows={len(df_oos):,}"
    )

    # -----------------------------
    # Fixed "churn killers" (executor owned)
    # -----------------------------
    cd_fixed = float(os.getenv("PRIME_REENTRY_COOLDOWN_H", "24"))
    mh_fixed = float(os.getenv("PRIME_EXIT_MIN_HOLD_H", "24"))

    # -----------------------------
    # Grid ranges (override via env if you want)
    # Example overrides:
    #   $env:GRID_TREND_MINS="0.2,0.3,0.4,0.55"
    #   $env:GRID_VOL_MAXES="0.012,0.016,0.02,0.03"
    #   $env:GRID_SPIKE_MAXES="1.2,1.4,1.6,1.8"
    #   $env:GRID_ADX_REL_MINS="0.9,1.0,1.1"
    # -----------------------------
    trend_mins = _parse_csv_floats("GRID_TREND_MINS", [0.20, 0.30, 0.40, 0.55])
    vol_maxes = _parse_csv_floats("GRID_VOL_MAXES", [0.012, 0.016, 0.020, 0.030])
    spike_maxes = _parse_csv_floats("GRID_SPIKE_MAXES", [1.2, 1.4, 1.6, 1.8])
    adx_rel_mins = _parse_csv_floats("GRID_ADX_REL_MINS", [0.9, 1.0, 1.1])

    # Other knobs pulled from env (single values, not gridded here)
    horizon_h = int(float(os.getenv("SGRT_HORIZON_H", "48")))
    vol_min = float(os.getenv("SGRT_VOL_MIN", "0.003"))
    vol_exit_max = float(os.getenv("SGRT_VOL_EXIT_MAX", "0.030"))
    adx_min = float(os.getenv("SGRT_ADX_MIN", "0.0"))
    adx_rel_win = int(float(os.getenv("SGRT_ADX_REL_WIN", "168")))
    adx_soft_penalty = float(os.getenv("SGRT_ADX_SOFT_PENALTY", "0.10"))
    vol_win = int(float(os.getenv("SGRT_VOL_WIN", "24")))
    trend_win = int(float(os.getenv("SGRT_TREND_WIN", "24")))

    # ADX mode: off/soft/hard. "hard" blocks entry on ADX fail; "soft" applies penalty; "off" disables.
    adx_mode = str(os.getenv("SGRT_ADX_MODE", "soft")).strip().lower() or "soft"
    if adx_mode not in ("off", "soft", "hard"):
        print(f"[GRID] WARNING: SGRT_ADX_MODE={adx_mode} invalid, defaulting to 'soft'")
        adx_mode = "soft"
    os.environ["SGRT_ADX_MODE"] = adx_mode
    print(f"[GRID] ADX mode: {adx_mode}")

    rows: List[Dict[str, Any]] = []
    total = len(trend_mins) * len(vol_maxes) * len(spike_maxes) * len(adx_rel_mins)
    k = 0

    for tmin in trend_mins:
        for vmax in vol_maxes:
            for smax in spike_maxes:
                for adx_rel_min in adx_rel_mins:
                    k += 1

                    # Make sure the external strategy module sees these values
                    # at _get_env_cfg() time.
                    with _temp_env(
                        {
                            "SGRT_TREND_MIN": str(tmin),
                            "SGRT_VOL_MAX": str(vmax),
                            "SGRT_SPIKE_RATIO_MAX": str(smax),
                            "SGRT_ADX_REL_MIN": str(adx_rel_min),
                            "SGRT_ADX_MODE": adx_mode,
                        }
                    ):
                        params = BTParams(
                            trend_min=float(tmin),
                            vol_min=float(vol_min),
                            vol_max=float(vmax),
                            horizon_h=int(horizon_h),
                            vol_exit_max=float(vol_exit_max),
                            reentry_cooldown_h=float(cd_fixed),
                            exit_min_hold_h=float(mh_fixed),
                            adx_min=float(adx_min),
                            adx_rel_win=int(adx_rel_win),
                            adx_rel_min=float(adx_rel_min),
                            adx_mode=adx_mode,
                            adx_soft_penalty=float(adx_soft_penalty),
                            vol_win=int(vol_win),
                            trend_win=int(trend_win),
                            spike_ratio_max=float(smax),
                        )

                        row = _run_one(df_is, df_oos, params)
                        rows.append(row)

                        print(
                            f"[GRID] {k:>3d}/{total} | "
                            f"tmin={tmin:>4.2f} vmax={vmax:>5.3f} smax={smax:>3.1f} adxRel>={adx_rel_min:>3.1f} | "
                            f"OOS CAGR={row['oos_cagr']*100:6.2f}% MDD={row['oos_mdd']*100:6.2f}% TR={row['oos_trades']:4d} | "
                            f"IS  CAGR={row['is_cagr']*100:6.2f}% MDD={row['is_mdd']*100:6.2f}% TR={row['is_trades']:4d}"
                        )

    out = pd.DataFrame(rows).sort_values("score_oos", ascending=False).reset_index(drop=True)
    out_path = os.path.join(OUT_DIR, "bt_sg_regime_trend_v1_param_grid.csv")
    out.to_csv(out_path, index=False)

    print(f"\n[GRID] wrote: {out_path}\n")
    print("Top 20 configs by holdout score:\n")
    print(out.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
