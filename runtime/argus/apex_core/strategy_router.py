from __future__ import annotations

import importlib
import os
import pandas as pd
from typing import Optional

from .strategy_intent import Intent, Action


def default_intent(reason: str = "") -> Intent:
    return Intent(action=Action.HOLD, reason=reason)


def load_strategy_callable() -> Optional[callable]:
    """
    Env:
      ARGUS_STRATEGY_MODULE="research.strategies.regime_trend_v1"
      ARGUS_STRATEGY_FUNC="generate_intent"
    Strategy must expose: generate_intent(df: pd.DataFrame, context: dict) -> Intent
    """
    mod_name = os.getenv("ARGUS_STRATEGY_MODULE", "").strip()
    fn_name = os.getenv("ARGUS_STRATEGY_FUNC", "generate_intent").strip()

    if not mod_name:
        return None

    mod = importlib.import_module(mod_name)
    fn = getattr(mod, fn_name, None)
    if fn is None:
        raise RuntimeError(f"Strategy func '{fn_name}' not found in module '{mod_name}'")
    return fn


def get_intent(df: pd.DataFrame, context: dict) -> Intent:
    fn = load_strategy_callable()
    if fn is None:
        # If no strategy configured, do nothing (hold).
        return default_intent("no_strategy_configured")

    intent = fn(df, context)
    if not isinstance(intent, Intent):
        raise RuntimeError("Strategy did not return Intent")
    if intent.action not in set(Action):
        raise RuntimeError(f"Invalid intent.action: {intent.action}")
    return intent
