import time
import pandas as pd
import joblib
from datetime import datetime
import yfinance as yf
from coinbase.rest import RESTClient

# --- NEW IMPORTS ---
import vault as secrets              # The Secure Key Vault
from apex_core.execution_engine import SmartExecutor # The Smart Hand

# --- CONFIGURATION ---
SYMBOL = "BTC-USD"
CAPITAL = 1000  # Example: Allocate $1,000 for this bot to trade
TIMEFRAME = "1h"

print("🦅 Itera Dynamics | Argus V2.0 (Smart Execution) INITIALIZING...")

# 1. SETUP CONNECTIONS
try:
    # Connect to Coinbase (The Bank)
    key_clean = secrets.API_KEY.strip().replace('"', '')
    secret_clean = secrets.API_SECRET.strip()
    client = RESTClient(api_key=key_clean, api_secret=secret_clean)
    
    # Initialize The Executor (The Hand)
    executor = SmartExecutor(client, SYMBOL)
    print(">> ✅ Coinbase Connection Established.")
    
except Exception as e:
    print(f">> ❌ CRITICAL: Failed to connect to Coinbase. {e}")
    exit()

# 2. LOAD BRAIN
try:
    model = joblib.load("random_forest.pkl")
    print(">> 🧠 Cortex Loaded (Random Forest)")
except:
    print(">> ❌ Brain missing! Run train_model.py first.")
    exit()

def get_market_data():
    """Fetches data for the Brain (Yahoo)"""
    df = yf.download(SYMBOL, period="7d", interval=TIMEFRAME, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df

def feature_engineering(df):
    """Prepares data for the Brain"""
    df['Returns'] = df['Close'].pct_change()
    
    # RSI
    window_length = 14
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window_length).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window_length).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['SMA_200'] = df['Close'].rolling(window=200).mean()
    df['Volatility'] = df['Close'].rolling(window=20).std()
    df['Momentum'] = df['Close'].pct_change(periods=10)
    
    df.dropna(inplace=True)
    return df

# --- MAIN LOOP ---
print(f"🚀 ARGUS V2 EXECUTION ENGINE ACTIVE [{datetime.now()}]")
last_processed = None

while True:
    now = datetime.now()
    
    # Run only at the top of the hour (XX:00)
    if now.minute == 0 and now.strftime("%Y-%m-%d %H") != last_processed:
        print(f"\n⏰ WAKING UP: {now}")
        
        # A. GET DATA
        raw_df = get_market_data()
        df = feature_engineering(raw_df)
        
        # B. THINK (Predict)
        latest_features = df[['RSI', 'SMA_50', 'SMA_200', 'Volatility', 'Momentum']].iloc[-1:]
        current_price = df['Close'].iloc[-1]
        
        prediction = model.predict(latest_features)[0]
        confidence = model.predict_proba(latest_features)[0][1]
        
        print(f"   [BRAIN] Price: ${current_price:,.2f} | Conf: {confidence:.2f}")
        
        # C. ACT (Smart Execution)
        # Note: We need a way to track if we are already in a position.
        # For V2 prototype, we will check our Last Action from a file or simple logic.
        # This is a simplified toggle for demonstration.
        
        if prediction == 1 and confidence > 0.60:
            print(f"   [SIGNAL] STRONG BUY DETECTED.")
            
            # THE UPGRADE: REAL TRADING
            # We attempt to buy $1000 worth of BTC using Limit Orders
            success = executor.execute_trade("BUY", CAPITAL)
            
            if success:
                print("   🎉 TRADE COMPLETE. We are Long.")
            else:
                print("   ⚠️ TRADE FAILED. Execution engine could not fill.")
                
        elif prediction == 0:
            print(f"   [SIGNAL] SELL / CASH DETECTED.")
            
            # THE UPGRADE: REAL SELLING
            # We attempt to sell everything (logic to calculate balance needed here)
            # Placeholder: Sell same amount we bought
            success = executor.execute_trade("SELL", CAPITAL) 
            
            if success:
                print("   📉 POSITION CLOSED. We are Liquid.")

        last_processed = now.strftime("%Y-%m-%d %H")
        print("   💤 Cycle Complete. Sleeping...")
        
    time.sleep(30)