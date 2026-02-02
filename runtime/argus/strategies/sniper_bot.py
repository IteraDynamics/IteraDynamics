# strategies/sniper_bot.py
# BTC Volatility Sniper (Fast trading strategy) — Phase 1 shell

import json, time, datetime, os
from decimal import Decimal
from pathlib import Path

from src.real_broker import RealBroker
from apex_core.market_feed import get_price  # we already have market fetches via core
from apex_core.utils import log  # same logger formatting for consistency

STATE = Path("sniper_state.json")


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
                RealBroker().execute_trade("BUY", qty, price)
            save_state({"active": True, "entry_price": price, "timestamp": time.time()})
            return

    else:
        # Placeholder exit logic — later real logic based on volatility feedback
        profit = (price - state["entry_price"]) / state["entry_price"] * 100
        if profit > 0.5:  # exit condition sample
            print(f"[SNIPER] 🔥 would SELL — Profit {profit:.2f}%")
            if not dry:
                RealBroker().execute_trade("SELL", 0.00025)  # replace with dynamic
            save_state({"active": False, "entry_price": None, "timestamp": None})
            return

    print(f"[SNIPER] 💤 No action. Price {price}")

    

if __name__ == "__main__":
    dry = bool(int(os.getenv("SNIPER_DRY", "1"))) # 1 = safe default
    sniper_cycle(dry=dry)
