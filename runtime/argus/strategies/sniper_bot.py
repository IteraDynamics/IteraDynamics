# strategies/sniper_bot.py
# BTC Volatility Sniper (Fast trading strategy) — Phase 1 shell

from __future__ import annotations

import sys
import json
import time
import datetime
import os
from decimal import Decimal
from pathlib import Path

# ---------------------------
# Path resolution (mirrors signal_generator.py pattern)
# ---------------------------
_CURRENT_FILE = Path(__file__).resolve()
_PROJECT_ROOT = _CURRENT_FILE.parent.parent  # runtime/argus/

if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

try:
    from src.real_broker import RealBroker
except ImportError as e:
    print(f"❌ SNIPER BROKER IMPORT ERROR: {e}")
    RealBroker = None  # type: ignore

import requests

STATE = _PROJECT_ROOT / "sniper_state.json"


def get_price() -> float:
    """Fetch live BTC-USD price from Coinbase."""
    try:
        url = "https://api.coinbase.com/v2/prices/BTC-USD/spot"
        resp = requests.get(url, timeout=5)
        resp.raise_for_status()
        return float(resp.json()["data"]["amount"])
    except Exception as e:
        print(f"[SNIPER] ⚠️ Price fetch failed: {e}")
        return 0.0


def log(msg: str) -> None:
    """Simple logger for sniper bot."""
    ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}")


def load_state():
    if STATE.exists():
        return json.loads(STATE.read_text())
    return {"active": False, "entry_price": None, "timestamp": None}


def save_state(s):
    STATE.write_text(json.dumps(s))


def sniper_cycle(dry=False):
    price = get_price()

    # Placeholder entry condition
    # Later: volatility burst detection — RSI spike, ATR expansion, MAs, etc
    long_entry = price % 2 < 1  # fake trigger just for testing

    state = load_state()

    if not state["active"]:
        if long_entry:
            qty = 20 / price  # allocating $20 micro entry for testing
            print(f"[SNIPER] 🚀 would BUY {qty:.6f} BTC @ {price}")
            if not dry:
                if RealBroker is None:
                    print("[SNIPER] ❌ RealBroker unavailable, skipping trade")
                else:
                    RealBroker().execute_trade("BUY", qty, price)
            save_state({"active": True, "entry_price": price, "timestamp": time.time()})
            return

    else:
        # Placeholder exit logic — later real logic based on volatility feedback
        profit = (price - state["entry_price"]) / state["entry_price"] * 100
        if profit > 0.5:  # exit condition sample
            print(f"[SNIPER] 🔥 would SELL — Profit {profit:.2f}%")
            if not dry:
                if RealBroker is None:
                    print("[SNIPER] ❌ RealBroker unavailable, skipping trade")
                else:
                    RealBroker().execute_trade("SELL", 0.00025)  # replace with dynamic
            save_state({"active": False, "entry_price": None, "timestamp": None})
            return

    print(f"[SNIPER] 💤 No action. Price {price}")

    

if __name__ == "__main__":
    dry = bool(int(os.getenv("SNIPER_DRY", "1"))) # 1 = safe default
    sniper_cycle(dry=dry)
