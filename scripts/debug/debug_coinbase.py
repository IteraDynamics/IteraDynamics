# debug_coinbase.py
# Diagnostic script to find your money in Coinbase

import sys
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')

import os
from dotenv import load_dotenv
from coinbase.rest import RESTClient

load_dotenv()

api_key = os.getenv("COINBASE_API_KEY") or os.getenv("CB_API_KEY")
api_secret = os.getenv("COINBASE_API_SECRET") or os.getenv("CB_API_SECRET")

if not api_key or not api_secret:
    print("❌ Missing API keys!")
    exit(1)

api_secret = api_secret.replace('\\n', '\n')
client = RESTClient(api_key=api_key, api_secret=api_secret)

print("=" * 60)
print("🔍 COINBASE DIAGNOSTIC - FINDING YOUR MONEY")
print("=" * 60)

# 1. List ALL portfolios
print("\n📁 STEP 1: Listing ALL Portfolios...")
try:
    portfolios = client.get_portfolios()
    if hasattr(portfolios, 'portfolios'):
        for p in portfolios.portfolios:
            name = getattr(p, 'name', 'Unknown')
            uuid = getattr(p, 'uuid', 'Unknown')
            ptype = getattr(p, 'type', 'Unknown')
            print(f"   📂 {name}")
            print(f"      UUID: {uuid}")
            print(f"      Type: {ptype}")
            print()
except Exception as e:
    print(f"   ⚠️ Could not list portfolios: {e}")

# 2. Get accounts WITHOUT portfolio filter (see everything)
print("\n💰 STEP 2: ALL Accounts (No Filter)...")
try:
    response = client.get_accounts(limit=250)
    if hasattr(response, 'accounts'):
        print(f"   Found {len(response.accounts)} total accounts")
        for acc in response.accounts:
            curr = getattr(acc, 'currency', '')
            avail = float(getattr(acc.available_balance, 'value', 0))
            hold = float(getattr(acc.hold, 'value', 0))
            acc_uuid = getattr(acc, 'uuid', 'Unknown')
            
            if avail > 0 or hold > 0 or curr in ['USD', 'BTC', 'USDC']:
                print(f"   💵 {curr}: Available={avail:.6f} | Hold={hold:.6f}")
                print(f"      Account UUID: {acc_uuid}")
except Exception as e:
    print(f"   ⚠️ Error: {e}")

# 3. Try with your hardcoded UUID
HARDCODED_UUID = "5bce9ffb-611c-4dcb-9e18-75d3914825a1"
print(f"\n🎯 STEP 3: Accounts with portfolio_id={HARDCODED_UUID}...")
try:
    response = client.get_accounts(limit=250, portfolio_id=HARDCODED_UUID)
    if hasattr(response, 'accounts'):
        print(f"   Found {len(response.accounts)} accounts in this portfolio")
        for acc in response.accounts:
            curr = getattr(acc, 'currency', '')
            avail = float(getattr(acc.available_balance, 'value', 0))
            hold = float(getattr(acc.hold, 'value', 0))
            
            if avail > 0 or hold > 0 or curr in ['USD', 'BTC', 'USDC']:
                print(f"   💵 {curr}: Available={avail:.6f} | Hold={hold:.6f}")
except Exception as e:
    print(f"   ⚠️ Error: {e}")

# 4. Try alternative parameter name
print(f"\n🎯 STEP 4: Trying 'portfolio' parameter instead...")
try:
    # Some API versions use 'portfolio' instead of 'portfolio_id'
    response = client.get_accounts(limit=250, portfolio=HARDCODED_UUID)
    if hasattr(response, 'accounts'):
        print(f"   Found {len(response.accounts)} accounts")
        for acc in response.accounts:
            curr = getattr(acc, 'currency', '')
            avail = float(getattr(acc.available_balance, 'value', 0))
            hold = float(getattr(acc.hold, 'value', 0))
            
            if avail > 0 or hold > 0 or curr in ['USD', 'BTC', 'USDC']:
                print(f"   💵 {curr}: Available={avail:.6f} | Hold={hold:.6f}")
except Exception as e:
    print(f"   ⚠️ Error with 'portfolio' param: {e}")

# 5. Check ALL portfolios for funds
print("\n🔑 STEP 5: Scanning ALL portfolios for funds...")
try:
    portfolios = client.get_portfolios()
    if hasattr(portfolios, 'portfolios'):
        for p in portfolios.portfolios:
            p_uuid = getattr(p, 'uuid', '')
            p_name = getattr(p, 'name', 'Unknown')
            print(f"\n   Checking portfolio: {p_name} ({p_uuid})")
            try:
                resp = client.get_accounts(limit=250, portfolio_id=p_uuid)
                if hasattr(resp, 'accounts'):
                    total_value = 0
                    for acc in resp.accounts:
                        curr = getattr(acc, 'currency', '')
                        avail = float(getattr(acc.available_balance, 'value', 0))
                        hold = float(getattr(acc.hold, 'value', 0))
                        if avail > 0 or hold > 0:
                            print(f"      💵 {curr}: Available={avail:.6f} | Hold={hold:.6f}")
                            total_value += avail + hold
                    if total_value == 0:
                        print(f"      (empty)")
            except Exception as e:
                print(f"      Error: {e}")
except Exception as e:
    print(f"   Error: {e}")

# 6. Raw API response check
print("\n📋 STEP 6: Raw Account Response Structure...")
try:
    response = client.get_accounts(limit=5)
    if hasattr(response, 'accounts') and len(response.accounts) > 0:
        acc = response.accounts[0]
        print(f"   Sample account attributes:")
        for attr in dir(acc):
            if not attr.startswith('_'):
                try:
                    val = getattr(acc, attr)
                    if not callable(val):
                        print(f"      {attr}: {val}")
                except:
                    pass
except Exception as e:
    print(f"   Error: {e}")

print("\n" + "=" * 60)
print("DIAGNOSTIC COMPLETE")
print("=" * 60)
print("\nIf all portfolios show $0, your API key likely doesn't have")
print("permission to view account balances. Create a new API key with")
print("'View' permission on the Argus_Alpha_Base portfolio.")

