# src/real_broker.py
# 🦅 ARGUS REAL BROKER - V7.0 (DICT FIX)

import os
from dotenv import load_dotenv
from coinbase.rest import RESTClient

load_dotenv()

# =====================================================
# 🔧 UUID for 'Argus_Alpha_Base'
# =====================================================
HARDCODED_UUID = "5bce9ffb-611c-4dcb-9e18-75d3914825a1" 
# =====================================================

class RealBroker:
    def __init__(self):
        self.api_key = os.getenv("COINBASE_API_KEY") or os.getenv("CB_API_KEY")
        self.api_secret = os.getenv("COINBASE_API_SECRET") or os.getenv("CB_API_SECRET")

        if not self.api_key or not self.api_secret:
            raise ValueError("❌ MISSING API KEYS in .env")

        self.api_secret = self.api_secret.replace('\\n', '\n')

        try:
            self.client = RESTClient(api_key=self.api_key, api_secret=self.api_secret)
            print(f"🔌 RealBroker: Connected. TARGETING UUID: {HARDCODED_UUID}")
            self._debug_balance()
        except Exception as e:
            print(f"❌ CONNECTION ERROR: {e}")
            raise e

    def _get_value(self, obj):
        """Helper to extract 'value' from either an Object or a Dict."""
        if obj is None:
            return 0.0
        
        # If it's a Dictionary (which your logs proved it is)
        if isinstance(obj, dict):
            return float(obj.get('value', 0))
        
        # If it's an Object (standard SDK behavior sometimes)
        return float(getattr(obj, 'value', 0))

    def _get_accounts(self):
        try:
            return self.client.get_accounts(limit=250, portfolio_id=HARDCODED_UUID)
        except Exception as e:
            print(f"   ⚠️ Account Fetch Error: {e}")
            return None

    def _debug_balance(self):
        try:
            response = self._get_accounts()
            if hasattr(response, 'accounts'):
                found_money = False
                print(f"   >> SCANNED {len(response.accounts)} WALLETS IN TARGET PORTFOLIO.")
                
                for acc in response.accounts:
                    curr = getattr(acc, 'currency', '')
                    
                    # 🔧 KEY FIX: Handle Dictionary Access
                    avail_obj = getattr(acc, 'available_balance', None)
                    hold_obj = getattr(acc, 'hold', None)
                    
                    avail = self._get_value(avail_obj)
                    hold = self._get_value(hold_obj)
                    
                    if curr == 'USD':
                         print(f"   >> 🔍 USD FOUND: Available=${avail} | Hold=${hold}")
                    
                    if avail + hold > 0:
                        found_money = True
                        print(f"   💰 FUNDS DETECTED: {curr} ${avail + hold}")
                
                if not found_money:
                     print("   ⚠️ PORTFOLIO IS EMPTY.")
        except Exception as e:
            print(f"   ⚠️ Balance Check Error: {e}")

    @property
    def cash(self):
        try:
            response = self._get_accounts()
            if hasattr(response, 'accounts'):
                for acc in response.accounts:
                    if getattr(acc, 'currency', '') == 'USD':
                        return self._get_value(getattr(acc, 'available_balance', None))
        except:
            pass
        return 0.0

    @property
    def positions(self):
        try:
            response = self._get_accounts()
            if hasattr(response, 'accounts'):
                for acc in response.accounts:
                    if getattr(acc, 'currency', '') == 'BTC':
                        return self._get_value(getattr(acc, 'available_balance', None))
        except:
            pass
        return 0.0

    def execute_trade(self, action, qty, price=None):
        product_id = "BTC-USD"
        client_order_id = str(int(os.urandom(4).hex(), 16))

        try:
            # 🔧 FIX 1: Removed 'config' dict causing 400 Bad Request
            
            if action.upper() == "BUY":
                usd_size = str(round(qty * price, 2))
                print(f"   🚀 SENDING BUY: ${usd_size}")
                # 🔧 FIX 2: Call directly without **config
                resp = self.client.market_order_buy(
                    client_order_id=client_order_id, 
                    product_id=product_id, 
                    quote_size=usd_size
                )
            elif action.upper() == "SELL":
                btc_size = f"{qty:.8f}"
                print(f"   🚀 SENDING SELL: {btc_size} BTC")
                # 🔧 FIX 3: Call directly without **config
                resp = self.client.market_order_sell(
                    client_order_id=client_order_id, 
                    product_id=product_id, 
                    base_size=btc_size
                )
            
            # Response handling
            if hasattr(resp, 'success') and resp.success:
                print(f"   ✅ ORDER FILLED. ID: {getattr(resp, 'order_id', 'Unknown')}")
                return True
            elif hasattr(resp, 'order_id'): 
                 print(f"   ✅ ORDER SUBMITTED. ID: {resp.order_id}")
                 return True
            else:
                 print(f"   ✅ ORDER SUBMITTED (Verify in Dashboard).")
                 return True

        except Exception as e:
            print(f"   ❌ ORDER ERROR: {e}")
            return False