import time
import logging
import signal
import sys
import random
from pathlib import Path
from datetime import datetime, timezone
import requests

# Import the Core Architecture
from apex_core.paper_broker import PaperBroker

# Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("MoonWire_Live")

# ---------------------------------------------------------------------
# MOCKS (Placeholders for your real code)
# ---------------------------------------------------------------------
class CoinbasePublicFeed:
    """Fetches REAL live prices from Coinbase Public API (US-Friendly)."""
    def fetch_latest_price(self, symbol: str) -> float:
        # Coinbase format: BTC-USD
        pair = f"{symbol.upper()}-USD"
        url = f"https://api.coinbase.com/v2/prices/{pair}/spot"
        
        try:
            resp = requests.get(url, timeout=5)
            data = resp.json()
            # Structure: {'data': {'base': 'BTC', 'currency': 'USD', 'amount': '95000.00'}}
            price = float(data['data']['amount'])
            
            logger.info(f"🔌 Feed: {symbol} @ ${price:,.2f}")
            return price
        except Exception as e:
            logger.error(f"Data Feed Error: {e}")
            # Fallback to keep loop alive
            return 95000.0

class MockStrategy:
    """Simulates your ML Model + Feature Engineering."""
    def analyze(self, price: float) -> str:
        # Randomly generate signals for testing the loop
        # 10% chance to BUY, 10% chance to SELL, 80% HOLD
        roll = random.random()
        if roll < 0.10:
            return "BUY"
        elif roll > 0.90:
            return "SELL"
        return "HOLD"

# ---------------------------------------------------------------------
# THE LIVE ENGINE
# ---------------------------------------------------------------------
class LiveEngine:
    def __init__(self, symbol: str, broker: PaperBroker):
        self.symbol = symbol.upper()
        self.broker = broker
        self.feed = CoinbasePublicFeed()      # <--- We will swap this later
        self.strategy = MockStrategy()  # <--- We will swap this later
        self.running = True

        # Handle Ctrl+C gracefully
        signal.signal(signal.SIGINT, self._handle_exit)
        signal.signal(signal.SIGTERM, self._handle_exit)

    def _handle_exit(self, signum, frame):
        logger.info("\n🛑 Shutdown signal received. Saving state...")
        self.running = False

    def run_loop(self, interval_seconds: int = 5):
        logger.info(f"🚀 Starting Engine on {self.symbol}")
        logger.info(f"💰 Balance: ${self.broker.get_balance():,.2f}")
        logger.info(f"Press Ctrl+C to stop.\n")

        while self.running:
            try:
                # 1. FETCH DATA
                price = self.feed.fetch_latest_price(self.symbol)
                
                # 2. STRATEGY INFERENCE
                signal = self.strategy.analyze(price)
                
                # 3. EXECUTION LOGIC
                # (Simple logic: If Buy signal and we have no position -> Buy)
                # (If Sell signal and we have position -> Sell)
                
                current_pos = self.broker.get_position(self.symbol)
                
                if signal == "BUY" and current_pos == 0:
                    logger.info(f"🟢 SIGNAL: BUY {self.symbol} @ {price:.2f}")
                    # Size: Buy 0.1 BTC (Hardcoded for test)
                    self.broker.submit_order(self.symbol, "BUY", 0.1, price)
                    
                elif signal == "SELL" and current_pos > 0:
                    logger.info(f"🔴 SIGNAL: SELL {self.symbol} @ {price:.2f}")
                    # Size: Sell everything
                    self.broker.submit_order(self.symbol, "SELL", current_pos, price)
                
                else:
                    # Heartbeat log (optional, keeps you sane)
                    # logger.debug(f"Holding... {price:.2f}")
                    pass

                # 4. SLEEP
                time.sleep(interval_seconds)

            except Exception as e:
                logger.error(f"⚠️ Crash in loop: {e}")
                time.sleep(interval_seconds) # Safety backoff

        logger.info("✅ Engine stopped safely.")

# ---------------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------------
if __name__ == "__main__":
    # Initialize the Paper Broker
    # Note: We save the state in the root folder so it's easy to find
    state_path = Path("moonwire_paper_state.json")
    
    broker = PaperBroker(
        state_file=state_path,
        initial_capital=10000.0,
        slippage_bps=5
    )

    # Launch the Engine
    engine = LiveEngine(symbol="BTC", broker=broker)
    engine.run_loop(interval_seconds=2) # Fast loop for testing