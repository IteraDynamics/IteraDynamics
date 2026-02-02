# sniper/__init__.py
# 🔫 BTC SNIPER PACKAGE INIT – REV 2
#
# - Exposes Rev-2 core API:
#     SniperConfig, add_features, run_sniper_backtest
# - Provides backward-compat shims:
#     prepare_features(df, cfg) -> add_features(...)
#     should_enter_long(row, cfg) -> simple volatility+trend rule

from __future__ import annotations

import numpy as np
import pandas as pd

from .sniper_core import (
    SniperConfig,
    add_features,
    run_sniper_backtest,
)

__all__ = [
    "SniperConfig",
    "add_features",
    "run_sniper_backtest",
    "prepare_features",
    "should_enter_long",
]


def prepare_features(df: pd.DataFrame, cfg: SniperConfig | None = None) -> pd.DataFrame:
    """
    Backwards-compatible wrapper for older code that imported
    `prepare_features` from sniper.

    Simply calls add_features(df, cfg).
    """
    if cfg is None:
        cfg = SniperConfig()
    return add_features(df, cfg)


def should_enter_long(row: pd.Series, cfg: SniperConfig | None = None) -> bool:
    """
    Backwards-compatible helper for older code.

    Implements the same basic idea as the Rev-2 entry condition:
      - fast SMA above slow SMA (uptrend)
      - return z-score >= breakout_zscore (volatility spike)
    """
    if cfg is None:
        cfg = SniperConfig()

    try:
        ret_z = float(row.get("ret_z"))
        sma_fast = float(row.get("sma_fast"))
        sma_slow = float(row.get("sma_slow"))
    except Exception:
        return False

    if any(np.isnan(x) for x in (ret_z, sma_fast, sma_slow)):
        return False

    return (sma_fast > sma_slow) and (ret_z >= cfg.breakout_zscore)
