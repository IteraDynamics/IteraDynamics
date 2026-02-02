# inspect_wallet.py
# 🩻 DATA X-RAY
# Prints the raw dictionary of your wallet to find the hidden balance.

import os
from dotenv import load_dotenv
from coinbase.rest import RESTClient

load_dotenv()

# The UUID we know exists
TARGET_UUID = "5bce9ffb-611c-4dcb-9e18-75d3914825a1"

def x_ray_wallet():
    api_key = os.getenv("COINBASE_API_KEY") or os.getenv("CB_API_KEY")
    api_secret = os.getenv("COINBASE_API_SECRET") or os.getenv("CB_API_SECRET")
    
    if not api_key:
        print("❌ KEYS MISSING")
        return

    api_secret = api_secret.replace('\\n', '\n')
    client = RESTClient(api_key=api_key, api_secret=api_secret)

    print(f"🩻 INSPECTING PORTFOLIO: {TARGET_UUID}")
    
    try:
        # Fetch accounts for this specific portfolio
        response = client.get_accounts(limit=10, portfolio_id=TARGET_UUID)
        
        if hasattr(response, 'accounts'):
            for acc in response.accounts:
                # We are looking for USD
                if getattr(acc, 'currency', '') == 'USD':
                    print("\n" + "="*40)
                    print("🎯 RAW USD WALLET DATA:")
                    print("="*40)
                    
                    # Print the object as a dictionary to see ALL fields
                    print(acc) 
                    
                    # Also try to print specific attributes manually just in case
                    print("-" * 20)
                    try: print(f"UUID: {acc.uuid}") 
                    except: pass
                    try: print(f"Available Balance Object: {acc.available_balance}") 
                    except: pass
                    try: print(f"Retail ID: {acc.retail_portfolio_id}") 
                    except: pass
                    
                    print("="*40 + "\n")
                    return
        
        print("❌ Could not find a USD wallet in this portfolio.")

    except Exception as e:
        print(f"❌ ERROR: {e}")

if __name__ == "__main__":
    x_ray_wallet()