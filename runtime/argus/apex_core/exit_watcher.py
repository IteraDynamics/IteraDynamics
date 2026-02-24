# apex_core/exit_watcher.py
# 🕔 ARGUS 5-MIN EXIT WATCHER - V1.3 (EARLY-EXIT & MIN-HOLD AWARE)

from __future__ import annotations

import os
from datetime import datetime, timezone, timedelta
from typing import Optional

import requests

from config import PRODUCT_ID
from apex_core.signal_generator import (
    _load_trade_state_legacy,
    MIN_HOLD_HOURS,
    MIN_NOTIONAL_USD,
    PROFIT_HURDLE_PCT,
)

try:
    from src.real_broker import RealBroker
except ImportError as e:
    print(f"❌ EXIT WATCHER BROKER IMPORT ERROR: {e}")
    RealBroker = None  # type: ignore

_exit_broker: Optional[RealBroker] = None
if RealBroker is not None:
    try:
        _exit_broker = RealBroker()
    except Exception as e:
        print(f"❌ EXIT WATCHER BROKER INIT ERROR: {e}")
        _exit_broker = None

# --- Early-exit tuning knobs (env-overridable) ---

# Master switch: allow early exits that can bypass MIN_HOLD_HOURS
EARLY_EXIT_ENABLED: bool = os.getenv("ARGUS_EARLY_EXIT_ENABLED", "1") not in (
    "0",
    "false",
    "False",
)

# Minimum time in position before early-exit is even considered (in hours)
EARLY_EXIT_MIN_HOURS: float = float(os.getenv("ARGUS_EARLY_EXIT_MIN_HOURS", "0.5"))

# Profit threshold for early exit (e.g. 0.01 = +1%)
EARLY_EXIT_PROFIT_PCT: float = float(os.getenv("ARGUS_EARLY_EXIT_PROFIT_PCT", "0.01"))


def _utc_now_str() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def _get_live_price() -> Optional[float]:
    """
    Test-friendly:
      - If ARGUS_TEST_PRICE is set, use that.
      - Else hit Coinbase public API.
    """
    test_price = os.getenv("ARGUS_TEST_PRICE")
    if test_price is not None:
        try:
            price = float(test_price)
            if price > 0:
                return price
        except ValueError:
            pass  # fall through to real API

    url = f"https://api.coinbase.com/v2/prices/{PRODUCT_ID}/spot"
    try:
        resp = requests.get(url, timeout=3)
        resp.raise_for_status()
        data = resp.json()
        amt = data.get("data", {}).get("amount")
        price = float(amt)
        if price <= 0:
            raise ValueError(f"Non-positive price: {price}")
        return price
    except Exception as e:
        print(f"[{_utc_now_str()}] ⚠️ EXIT WATCHER PRICE FAIL: {e}")
        return None


def _get_wallet_btc() -> Optional[float]:
    """
    Test-friendly:
      - If ARGUS_TEST_BTC is set, use that.
      - Else use RealBroker wallet snapshot.
    """
    test_btc = os.getenv("ARGUS_TEST_BTC")
    if test_btc is not None:
        try:
            btc = float(test_btc)
            if btc >= 0:
                return btc
        except ValueError:
            pass  # fall through

    global _exit_broker
    if _exit_broker is None:
        print(f"[{_utc_now_str()}] ⚠️ EXIT WATCHER: RealBroker unavailable.")
        return None

    try:
        cash, btc = _exit_broker.get_wallet_snapshot()
        return float(btc)
    except Exception as e:
        print(f"[{_utc_now_str()}] ⚠️ EXIT WATCHER WALLET FAIL: {e}")
        return None


def check_exit_window() -> None:
    """
    Main 5-minute exit check.

    SAFE BEHAVIOR:
      - Requires valid trade_state + BTC position above MIN_NOTIONAL_USD.
      - Uses live or test price/BTC.
      - Supports EARLY EXIT that can bypass MIN_HOLD_HOURS when:
          * EARLY_EXIT_ENABLED, and
          * hold_hours >= EARLY_EXIT_MIN_HOURS, and
          * profit_pct >= EARLY_EXIT_PROFIT_PCT.
      - Otherwise enforces MIN_HOLD_HOURS and normal PROFIT_HURDLE_PCT.
      - In DRY-RUN mode, SELL becomes a logged "Would SELL" via RealBroker.
    """
    ts_now = _utc_now_str()
    print(f"[{ts_now}] 🕔 EXIT WATCHER CHECK...")

    # 1) Load trade_state
    state, status = _load_trade_state_legacy()

    if status != "OK" or state is None:
        print(
            f"[{_utc_now_str()}] [EXIT] HOLD | Reason: "
            f"NO_VALID_STATE | Status: {status}"
        )
        return

    entry_time = state["_entry_time"]
    entry_price = float(state["entry_price"])

    # 2) Time in position
    now_utc = datetime.now(timezone.utc)
    hold_delta: timedelta = now_utc - entry_time
    hold_hours = hold_delta.total_seconds() / 3600.0

    # 3) Live price
    price = _get_live_price()
    if price is None or price <= 0:
        print(f"[{_utc_now_str()}] [EXIT] HOLD | Reason: PRICE_UNAVAILABLE_5M")
        return

    # 4) BTC balance
    btc = _get_wallet_btc()
    if btc is None:
        print(f"[{_utc_now_str()}] [EXIT] HOLD | Reason: WALLET_UNVERIFIED_5M")
        return

    btc_notional = btc * price
    if btc_notional < MIN_NOTIONAL_USD:
        print(
            f"[{_utc_now_str()}] [EXIT] HOLD | Reason: NO_POSITION_OR_BELOW_MIN_NOTIONAL_5M | "
            f"BTC_Notional: ${btc_notional:.2f}"
        )
        return

    # 5) Profit
    profit_pct = (price - entry_price) / entry_price

    # 5a) EARLY-EXIT OVERRIDE (can bypass MIN_HOLD_HOURS)
    if (
        EARLY_EXIT_ENABLED
        and hold_hours >= EARLY_EXIT_MIN_HOURS
        and profit_pct >= EARLY_EXIT_PROFIT_PCT
    ):
        if _exit_broker is None:
            print(
                f"[{_utc_now_str()}] [EXIT] HOLD | "
                f"Reason: BROKER_UNAVAILABLE_EARLY_EXIT_5M"
            )
            return

        print(
            f"[{_utc_now_str()}] [EXIT] EARLY_EXECUTION_5M | "
            f"Profit: {profit_pct:.3%} | Held: {hold_hours:.2f}h | "
            f"Entry: ${entry_price:.2f} | Now: ${price:.2f} | "
            f"HurdleEarly: {EARLY_EXIT_PROFIT_PCT:.3%}"
        )
        try:
            ok = _exit_broker.execute_trade("SELL", btc, price)
            if ok:
                print(f"[{_utc_now_str()}] [EXIT] ✅ EARLY 5M SELL ORDER ACCEPTED.")
            else:
                print(
                    f"[{_utc_now_str()}] [EXIT] ❌ EARLY 5M SELL ORDER NOT CONFIRMED."
                )
        except Exception as e:
            print(f"[{_utc_now_str()}] [EXIT] ❌ EARLY 5M SELL ERROR: {e}")
        return

    # 5b) Normal min-hold guard for standard exits
    if hold_hours < MIN_HOLD_HOURS:
        print(
            f"[{_utc_now_str()}] [EXIT] HOLD | Reason: MIN_HOLD_NOT_MET_5M | "
            f"Held: {hold_hours:.2f}h | Min: {MIN_HOLD_HOURS:.2f}h"
        )
        return

    # 5c) Standard profit hurdle (post min-hold)
    if profit_pct < PROFIT_HURDLE_PCT:
        print(
            f"[{_utc_now_str()}] [EXIT] HOLD | Reason: PROFIT_HURDLE_NOT_MET_5M | "
            f"Profit: {profit_pct:.3%} | Hurdle: {PROFIT_HURDLE_PCT:.3%}"
        )
        return

    # 6) Route SELL (standard path)
    if _exit_broker is None:
        print(f"[{_utc_now_str()}] [EXIT] HOLD | Reason: BROKER_UNAVAILABLE_5M")
        return

    print(
        f"[{_utc_now_str()}] [EXIT] EXECUTION_5M | "
        f"Profit: {profit_pct:.3%} | Held: {hold_hours:.2f}h | "
        f"Entry: ${entry_price:.2f} | Now: ${price:.2f}"
    )
    try:
        ok = _exit_broker.execute_trade("SELL", btc, price)
        if ok:
            print(f"[{_utc_now_str()}] [EXIT] ✅ 5M SELL ORDER ACCEPTED.")
        else:
            print(f"[{_utc_now_str()}] [EXIT] ❌ 5M SELL ORDER NOT CONFIRMED.")
    except Exception as e:
        print(f"[{_utc_now_str()}] [EXIT] ❌ 5M SELL ERROR: {e}")


if __name__ == "__main__":
    # Convenience: allow `python -m apex_core.exit_watcher` to run a single check.
    check_exit_window()
