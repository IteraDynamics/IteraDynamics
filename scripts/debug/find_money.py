# find_money.py
# 🕵️‍♀️ PORTFOLIO DETECTIVE
# This script finds which Portfolio UUID actually holds your cash.

import os
from dotenv import load_dotenv
from coinbase.rest import RESTClient

load_dotenv()

def find_the_money():
    api_key = os.getenv("COINBASE_API_KEY") or os.getenv("CB_API_KEY")
    api_secret = os.getenv("COINBASE_API_SECRET") or os.getenv("CB_API_SECRET")
    
    if not api_key:
        print("❌ KEYS MISSING in .env")
        return

    # Sanitize secret
    api_secret = api_secret.replace('\\n', '\n')

    client = RESTClient(api_key=api_key, api_secret=api_secret)
    print("🕵️‍♀️ SCANNING ALL PORTFOLIOS...\n")

    try:
        # 1. Get List of Portfolios
        portfolios = client.get_portfolios()
        
        if not hasattr(portfolios, 'portfolios'):
            print("❌ Could not fetch portfolios.")
            return

        for p in portfolios.portfolios:
            p_name = getattr(p, 'name', 'Unknown')
            p_uuid = getattr(p, 'uuid', 'Unknown')
            print(f"📁 CHECKING PORTFOLIO: {p_name} (ID: {p_uuid})")
            
            # 2. Check Balance of this specific Portfolio
            try:
                accounts = client.get_accounts(portfolio_id=p_uuid)
                found_cash = False
                
                if hasattr(accounts, 'accounts'):
                    for acc in accounts.accounts:
                        currency = getattr(acc, 'currency', '')
                        val = float(getattr(acc.available_balance, 'value', 0))
                        
                        if val > 0:
                            print(f"   💰 FOUND FUNDS: {currency} {val}!")
                            found_cash = True
                            
                if not found_cash:
                    print("   (Empty)")
                    
            except Exception as e:
                print(f"   ⚠️ Access Denied to this portfolio: {e}")
            
            print("-" * 30)

    except Exception as e:
        print(f"❌ CRITICAL ERROR: {e}")

if __name__ == "__main__":
    find_the_money()