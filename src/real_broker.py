import os
import time
import hmac
import hashlib
import json
import requests
from dotenv import load_dotenv

# Load keys from .env
load_dotenv()

class RealBroker:
    def __init__(self):
        self.api_key = os.getenv("COINBASE_API_KEY")
        self.api_secret = os.getenv("COINBASE_API_SECRET")
        self.base_url = "https://api.coinbase.com"
        
        if not self.api_key or not self.api_secret:
            print("❌ CRITICAL: Coinbase API Keys missing from .env")
            raise ValueError("Missing API Keys")
            
        print("🔌 RealBroker: Connected to Coinbase Advanced Trade.")

    def _generate_signature(self, method, request_path, body=""):
        timestamp = str(int(time.time()))
        message = timestamp + method + request_path + body
        signature = hmac.new(
            self.api_secret.encode('utf-8'),
            message.encode('utf-8'),
            digestmod=hashlib.sha256
        ).hexdigest()
        return timestamp, signature

    def _request(self, method, endpoint, payload=None):
        request_path = f"/api/v3/brokerage{endpoint}"
        url = self.base_url + request_path
        body = json.dumps(payload) if payload else ""
        
        timestamp, signature = self._generate_signature(method, request_path, body)
        
        headers = {
            "CB-ACCESS-KEY": self.api_key,
            "CB-ACCESS-SIGN": signature,
            "CB-ACCESS-TIMESTAMP": timestamp,
            "Content-Type": "application/json"
        }
        
        try:
            if method == "GET":
                response = requests.get(url, headers=headers)
            else:
                response = requests.post(url, headers=headers, data=body)
            
            if response.status_code != 200:
                print(f"   ⚠️ Coinbase Error ({response.status_code}): {response.text}")
                return None
                
            return response.json()
        except Exception as e:
            print(f"   ⚠️ Network Error: {e}")
            return None

    @property
    def cash(self):
        """Get USD Balance"""
        data = self._request("GET", "/accounts")
        if data:
            for account in data.get('accounts', []):
                if account['currency'] == 'USD':
                    return float(account['available_balance']['value'])
        return 0.0

    @property
    def positions(self):
        """Get BTC Balance"""
        data = self._request("GET", "/accounts")
        if data:
            for account in data.get('accounts', []):
                if account['currency'] == 'BTC':
                    return float(account['available_balance']['value'])
        return 0.0

    def execute_trade(self, action, qty, price=None):
        """
        Executes a MARKET Order.
        Argus V1.0 uses Market Orders to ensure execution.
        """
        product_id = "BTC-USD"
        client_order_id = str(int(time.time() * 1000)) # Unique ID
        
        # Coinbase requires String for quantities
        # BUY is in Quote Currency (USD) -> "Spend $50"
        # SELL is in Base Currency (BTC) -> "Sell 0.001 BTC"
        
        payload = {
            "client_order_id": client_order_id,
            "product_id": product_id,
            "side": action.upper(),
            "order_configuration": {}
        }

        if action.upper() == "BUY":
            # Market Buy: Spend specific dollar amount
            # We calculate roughly the BTC qty to USD, but Coinbase API wants "quote_size" (USD) for buys
            usd_size = str(round(qty * price, 2)) 
            payload["order_configuration"] = {
                "market_market_ioc": {
                    "quote_size": usd_size 
                }
            }
            print(f"   🚀 SENDING LIVE BUY ORDER: ${usd_size} of BTC")

        elif action.upper() == "SELL":
            # Market Sell: Sell specific BTC amount
            btc_size = f"{qty:.8f}"
            payload["order_configuration"] = {
                "market_market_ioc": {
                    "base_size": btc_size
                }
            }
            print(f"   🚀 SENDING LIVE SELL ORDER: {btc_size} BTC")

        # Execute
        resp = self._request("POST", "/orders", payload)
        
        if resp and resp.get('success'):
            order_id = resp.get('order_id')
            print(f"   ✅ ORDER FILLED. ID: {order_id}")
            return True
        else:
            print(f"   ❌ ORDER FAILED: {resp}")
            return False