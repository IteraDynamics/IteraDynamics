import time
import uuid
import logging
from decimal import Decimal

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - [EXECUTION] - %(message)s')

class SmartExecutor:
    def __init__(self, api_client, symbol):
        """
        Initialize with a trading client (Coinbase) and a target symbol (BTC-USD).
        """
        self.client = api_client
        self.symbol = symbol
        self.max_retries = 5
        self.wait_time = 15  # Seconds to wait before checking fill

    def get_best_bid_ask(self):
        """
        Fetches the current order book to find the 'Maker' price.
        Returns: (best_bid, best_ask) as Floats
        """
        try:
            # Fetch level 1 order book (just the top)
            book = self.client.get_product_book(product_id=self.symbol, limit=1)
            
            # Parse the response (SDK returns specific object structure)
            # We look at 'pricebook' -> 'bids'/'asks' -> index 0 -> 'price'
            best_bid = float(book.pricebook.bids[0].price)
            best_ask = float(book.pricebook.asks[0].price)
            
            return best_bid, best_ask
        except Exception as e:
            logging.error(f"Failed to fetch order book: {e}")
            raise e

    def place_limit_order(self, side, quantity, price):
        """
        Posts a limit order to the exchange.
        """
        client_oid = str(uuid.uuid4()) # Unique ID for this specific attempt
        price_str = f"{price:.2f}"     # Format to 2 decimal places (USD)
        size_str = f"{quantity:.8f}"   # Format to 8 decimal places (Crypto)

        print(f">> 📝 Posting LIMIT {side} {size_str} {self.symbol} @ ${price_str}")
        
        try:
            if side == 'BUY':
                response = self.client.limit_order_gtc_buy(
                    client_order_id=client_oid,
                    product_id=self.symbol,
                    base_size=size_str,
                    limit_price=price_str
                )
            else:
                response = self.client.limit_order_gtc_sell(
                    client_order_id=client_oid,
                    product_id=self.symbol,
                    base_size=size_str,
                    limit_price=price_str
                )
            
            # Extract Order ID from response
            if response.success:
                return response.success_response.order_id
            else:
                logging.error(f"Order Placement Failed: {response.error_response}")
                return None

        except Exception as e:
            logging.error(f"Critical Error placing order: {e}")
            return None

    def check_order_status(self, order_id):
        """
        Checks if the order is FILLED, OPEN, or CANCELLED.
        """
        try:
            order = self.client.get_order(order_id=order_id)
            return order.order.status # Returns 'OPEN', 'FILLED', 'CANCELLED', etc.
        except Exception as e:
            logging.error(f"Failed to check order status: {e}")
            return "UNKNOWN"

    def cancel_order(self, order_id):
        """
        Cancels an open order so we can move it.
        """
        try:
            print(f">> ❌ Cancelling {order_id}")
            self.client.cancel_orders(order_ids=[order_id])
            return True
        except Exception as e:
            logging.error(f"Failed to cancel order: {e}")
            return False

    def execute_trade(self, side, quantity_usd):
        """
        The Main Loop: Tries to get filled as a Maker.
        """
        print(f"🚀 STARTING SMART EXECUTION: {side} ${quantity_usd}")
        
        attempt = 0
        filled = False
        
        while attempt < self.max_retries and not filled:
            attempt += 1
            print(f"--- Attempt {attempt}/{self.max_retries} ---")
            
            # 1. Get Price
            try:
                best_bid, best_ask = self.get_best_bid_ask()
            except:
                print(">> ⚠️ Data fetch failed. Retrying...")
                time.sleep(2)
                continue
            
            # 2. Calculate Maker Price (+ $0.01 for Buy, - $0.01 for Sell)
            if side == 'BUY':
                target_price = best_bid + 0.01
            else:
                target_price = best_ask - 0.01
            
            # Calculate quantity in crypto terms (Base Size)
            quantity_crypto = quantity_usd / target_price
            
            # 3. Post Order
            order_id = self.place_limit_order(side, quantity_crypto, target_price)
            
            if not order_id:
                print(">> ❌ Critical: Failed to get Order ID. Retrying...")
                continue

            # 4. Wait
            print(f"⏳ Waiting {self.wait_time}s for fill...")
            time.sleep(self.wait_time)
            
            # 5. Check
            status = self.check_order_status(order_id)
            
            if status == 'FILLED':
                print("✅ FILL CONFIRMED!")
                filled = True
            else:
                print(f"⚠️ Order status: {status}. Market moved?")
                self.cancel_order(order_id)
        
        if not filled:
            print("🚨 FAILED to fill after max retries. Taking liquidity (Market Order)?")
            # In a real bot, we would return False here and let the logic decide 
            # if it wants to pay the Taker fee or just abort.
            return False
            
        return True

# No __main__ block needed here because we can't test without keys.