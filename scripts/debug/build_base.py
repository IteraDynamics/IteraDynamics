# build_base.py
# 🏗️ THE ARCHITECT
# This script creates a fresh Portfolio and gives you the Key (UUID) immediately.

import os
from dotenv import load_dotenv
from coinbase.rest import RESTClient

load_dotenv()

def build_new_base():
    api_key = os.getenv("COINBASE_API_KEY") or os.getenv("CB_API_KEY")
    api_secret = os.getenv("COINBASE_API_SECRET") or os.getenv("CB_API_SECRET")
    
    if not api_key:
        print("❌ KEYS MISSING in .env")
        return

    api_secret = api_secret.replace('\\n', '\n')

    try:
        client = RESTClient(api_key=api_key, api_secret=api_secret)
        print("🏗️  CONNECTING TO COINBASE...")
        
        # 1. Create the Portfolio
        # We use a unique name so you can find it easily
        new_name = "Argus_Alpha_Base"
        print(f"🔨 CREATING NEW PORTFOLIO: '{new_name}'...")
        
        portfolio = client.create_portfolio(name=new_name)
        
        # 2. Extract Data
        # The SDK returns a response object
        p_uuid = getattr(portfolio, 'uuid', None)
        p_name = getattr(portfolio, 'name', None)

        if p_uuid:
            print("\n" + "="*40)
            print(f"✅ SUCCESS! NEW BASE ESTABLISHED.")
            print(f"📛 NAME: {p_name}")
            print(f"🔑 UUID: {p_uuid}")
            print("="*40)
            print("\n👉 NEXT STEPS:")
            print("1. Go to Coinbase.com -> Portfolios.")
            print(f"2. Find '{p_name}'.")
            print("3. TRANSFER your $100 USD into it.")
            print("4. Copy the UUID above and paste it into src/real_broker.py")
        else:
            print(f"❌ CREATION FAILED. Response: {portfolio}")

    except Exception as e:
        print(f"❌ ERROR: {e}")

if __name__ == "__main__":
    build_new_base()