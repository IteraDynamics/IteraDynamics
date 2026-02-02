# verify_reality.py
# 🕵️‍♀️ THE FINAL REALITY CHECK
# No filters. No targets. Just raw truth.

import os
from dotenv import load_dotenv
from coinbase.rest import RESTClient

# 1. Force reload of .env to ensure we aren't using cached stale keys
load_dotenv(override=True)

def check_reality():
    api_key = os.getenv("COINBASE_API_KEY") or os.getenv("CB_API_KEY")
    api_secret = os.getenv("COINBASE_API_SECRET") or os.getenv("CB_API_SECRET")
    
    if not api_key:
        print("❌ CRITICAL: No API Key found in .env")
        return

    # Print Key ID to ensure it matches what you just created
    # (The Key ID is usually the first part of the string or visible in the portal)
    print(f"🔑 USING KEY STARTING WITH: {api_key[:5]}...")

    api_secret = api_secret.replace('\\n', '\n')

    try:
        client = RESTClient(api_key=api_key, api_secret=api_secret)
        print("📡 SCANNING EVERYTHING...\n")

        # A. LIST ALL PORTFOLIOS THE KEY CAN SEE
        print("--- 📂 PORTFOLIO VISIBILITY CHECK ---")
        try:
            p_resp = client.get_portfolios()
            portfolios = p_resp.portfolios if hasattr(p_resp, 'portfolios') else []
            
            if not portfolios:
                print("⚠️  Use Warning: This Key sees 0 Portfolios. (Is it restricted to just one?)")
            
            for p in portfolios:
                print(f"   📁 NAME: {getattr(p, 'name', 'Unknown')} | UUID: {getattr(p, 'uuid', 'Unknown')}")
        except Exception as e:
            print(f"   ❌ Portfolio Permission Error: {e}")

        # B. LIST ALL NON-ZERO BALANCES (ACROSS ALL VISIBLE ACCOUNTS)
        print("\n--- 💰 CASH & ASSET HUNT ---")
        found_anything = False
        
        # We fetch accounts with a high limit to catch everything
        try:
            acc_resp = client.get_accounts(limit=250)
            accounts = acc_resp.accounts if hasattr(acc_resp, 'accounts') else []
            
            print(f"   (Scanned {len(accounts)} total wallets...)")

            for acc in accounts:
                # Get Values
                avail = float(getattr(acc.available_balance, 'value', 0))
                hold = float(getattr(acc.hold, 'value', 0))
                total = avail + hold
                
                # IF IT EXISTS, PRINT IT. NO FILTERS.
                if total > 0:
                    curr = getattr(acc, 'currency', 'UNKNOWN')
                    p_uuid = getattr(acc, 'portfolio_uuid', getattr(acc, 'retail_portfolio_id', 'Unknown'))
                    print(f"   🟢 FOUND: {curr} | Balance: {total} | Portfolio UUID: {p_uuid}")
                    found_anything = True
            
            if not found_anything:
                print("   🔴 RESULT: ABSOLUTELY NO FUNDS FOUND.")
                print("      This implies the Key is looking at an empty portfolio.")

        except Exception as e:
            print(f"   ❌ Account Read Error: {e}")

    except Exception as e:
        print(f"❌ CONNECTION ERROR: {e}")

if __name__ == "__main__":
    check_reality()