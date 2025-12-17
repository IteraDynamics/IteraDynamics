# apex_core/signal_generator.py
# 🦅 ARGUS LIVE PILOT - V2.0 (REGIME GATED + RISK SIZING)

from __future__ import annotations
import sys
import os
import joblib
import pandas as pd
import pandas_ta as ta
import requests
import json
import time
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

# --- 🔧 CRITICAL PATH FIX ---
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent # apex_core -> IteraDynamics_Mono
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

load_dotenv(project_root / ".env")

# --- CONFIGURATION ---
MODELS_DIR = project_root / "moonwire/models"
MODEL_FILE = "random_forest.pkl"
DATA_FILE = project_root / "flight_recorder.csv"

# --- ⚠️ LIVE BROKER IMPORT ⚠️ ---
try:
    from src.real_broker import RealBroker
except ImportError as e:
    print(f"❌ CRITICAL IMPORT ERROR: {e}")
    sys.exit(1)

# Connect to API
try:
    print("   >> 🦅 CONNECTING TO LIVE COINBASE API...")
    _broker = RealBroker() 
except Exception as e:
    print(f"❌ CRITICAL: Broker Connection Failed: {e}")
    sys.exit(1)


def update_market_data(csv_path: Path = DATA_FILE):
    """ Fetches latest candle from Coinbase to keep the CSV alive. """
    try:
        url = "https://api.exchange.coinbase.com/products/BTC-USD/candles"
        params = {'granularity': 3600} # 1 Hour
        resp = requests.get(url, params=params, timeout=10)
        data = resp.json()
        data.sort(key=lambda x: x[0]) 
        
        if not csv_path.exists():
            print("   >> ⚠️ Flight Recorder missing. Creating new...")
            pd.DataFrame(columns=["Timestamp","Open","High","Low","Close","Volume"]).to_csv(csv_path, index=False)

        df = pd.read_csv(csv_path)
        if not df.empty:
            last_ts = pd.to_datetime(df['Timestamp']).max()
        else:
            last_ts = datetime.min
        
        new_rows = []
        for candle in data:
            ts = datetime.fromtimestamp(candle[0])
            if ts > last_ts:
                new_rows.append({
                    'Timestamp': ts.strftime('%Y-%m-%d %H:%M:%S'),
                    'Open': candle[3], 'High': candle[2], 'Low': candle[1], 'Close': candle[4], 'Volume': candle[5]
                })
        
        if new_rows:
            pd.DataFrame(new_rows).to_csv(csv_path, mode='a', header=False, index=False)
            print(f"   >> ✅ Market Data Updated. Newest: {new_rows[-1]['Timestamp']}")
    except Exception as e:
        print(f"   >> ⚠️ Data Update Glitch: {e}")

def detect_regime(df):
    """
    🛡️ REGIME DETECTION ENGINE (The Full 6-Regime Matrix)
    1. 🐂 BULL QUIET (Strong Uptrend, Low Vol) -> Aggressive Buy (0.90)
    2. 🐎 BULL VOLATILE (Strong Uptrend, High Vol) -> Moderate Buy (0.50)
    3. 🐻 BEAR QUIET (Downtrend, Low Vol) -> Cash (0.00)
    4. 🌪️ BEAR VOLATILE (Downtrend, High Vol) -> Cash (0.00)
    5. 🐯 RECOVERY (Below 200, Above 50) -> Speculative Buy (0.25)
    6. ⚠️ PULLBACK (Above 200, Below 50) -> Caution/Cash (0.00)
    """
    try:
        # Calculate Trend (SMA 50 vs SMA 200)
        sma_50 = ta.sma(df['Close'], length=50)
        sma_200 = ta.sma(df['Close'], length=200)
        
        # Calculate Volatility (ATR / Price)
        atr = ta.atr(df['High'], df['Low'], df['Close'], length=14)
        volatility = atr / df['Close']
        vol_threshold = volatility.rolling(100).mean().iloc[-1] 
        
        current_price = df['Close'].iloc[-1]
        current_sma50 = sma_50.iloc[-1]
        current_sma200 = sma_200.iloc[-1] if not sma_200.isnull().all() else current_sma50 
        current_vol = volatility.iloc[-1]

        regime = "Unknown"
        risk_mult = 0.0

        # --- PRIMARY TREND: BULLISH (Price > SMA 200) ---
        if current_price > current_sma200:
            if current_price > current_sma50:
                # Full Bull (Above Both)
                if current_vol < vol_threshold:
                    regime = "🐂 BULL QUIET"
                    risk_mult = 0.90
                else:
                    regime = "🐎 BULL VOLATILE"
                    risk_mult = 0.50
            else:
                # Weak Bull (Above 200, Below 50) - THE 6TH REGIME
                regime = "⚠️ PULLBACK (Warning)"
                risk_mult = 0.0 # Step aside, trend is weakening

        # --- PRIMARY TREND: BEARISH (Price < SMA 200) ---
        else:
            if current_price < current_sma50:
                # Full Bear (Below Both)
                if current_vol < vol_threshold:
                    regime = "🐻 BEAR QUIET"
                    risk_mult = 0.0
                else:
                    regime = "🌪️ BEAR VOLATILE"
                    risk_mult = 0.0
            else:
                # Recovery (Below 200, Above 50)
                regime = "🐯 RECOVERY"
                risk_mult = 0.25

        return regime, risk_mult

    except Exception as e:
        print(f"   >> ⚠️ Regime Detection Failed: {e}")
        return "Unknown", 0.0
        
def get_latest_features(csv_path: Path = DATA_FILE):
    try:
        df = pd.read_csv(csv_path)
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
        df.sort_values('Timestamp', inplace=True)
        
        # Feature Engineering
        df['RSI'] = ta.rsi(df['Close'], length=14)
        bband = ta.bbands(df['Close'], length=20, std=2)
        lower_col = next(c for c in bband.columns if c.startswith("BBL"))
        upper_col = next(c for c in bband.columns if c.startswith("BBU"))
        df['BB_Pos'] = (df['Close'] - bband[lower_col]) / (bband[upper_col] - bband[lower_col])
        df['Vol_Z'] = (df['Volume'] - df['Volume'].rolling(20).mean()) / df['Volume'].rolling(20).std()
        
        last = df.iloc[-1]
        
        # RUN REGIME DETECTION
        regime_name, risk_mult = detect_regime(df)
        
        return pd.DataFrame([[last['RSI'], last['BB_Pos'], last['Vol_Z']]], columns=['RSI', 'BB_Pos', 'Vol_Z']), last['Close'], regime_name, risk_mult
    except:
        return None, None, "Error", 0.0

def generate_signals():
    print(f"[{datetime.now().time()}] 🦅 ARGUS LIVE EXECUTION CYCLE...")
    
    update_market_data()
    try:
        model = joblib.load(MODELS_DIR / MODEL_FILE)
    except FileNotFoundError:
        print(f"❌ Model not found at {MODELS_DIR / MODEL_FILE}")
        return

    features, price, regime, risk_mult = get_latest_features()
    
    if features is None:
        print("   >> ❌ No Data. Skipping.")
        return

    prediction = model.predict(features)[0]
    raw_signal = "BUY" if prediction == 1 else "SELL"
    
    print(f"   >> [BRAIN] Raw Signal: {raw_signal}")
    print(f"   >> [REGIME] {regime} | Risk Multiplier: {risk_mult:.2f}")

    # --- 🔧 NEW: SAVE CORTEX STATE FOR DASHBOARD ---
    cortex_data = {
        "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "regime": regime,
        "risk_mult": risk_mult,
        "raw_signal": raw_signal,
        "conviction_score": int(risk_mult * 100) # Convert 0.50 -> 50
    }
    
    # Save to a tiny JSON file that Dashboard watches
    try:
        with open(project_root / "cortex.json", "w") as f:
            json.dump(cortex_data, f)
    except Exception as e:
        print(f"   >> ⚠️ Cortex Save Error: {e}")
    # -----------------------------------------------

    # 3. LIVE EXECUTION
    try:
        cash = _broker.cash
        position = _broker.positions
        
        print(f"   >> [WALLET] Cash: ${cash:.2f} | BTC: {position:.6f}")
        
        est_value = cash + (position * price)
        if est_value < 85.00:
            print("   >> 🛑 CRITICAL: Account Value < $85. Trading Halted.")
            return

        # --- GATED LOGIC ---
        if raw_signal == "BUY":
            # ONLY BUY IF REGIME PERMITS
            if risk_mult > 0.0:
                # 🛡️ RISK SIZING: Use risk_mult % of available cash
                target_spend = cash * risk_mult
                
                # Minimum viable trade check ($5 min on Coinbase usually)
                if target_spend > 5.0:
                    qty = target_spend / price
                    print(f"   >> 🦅 ROUTING ORDER: BUY {qty:.6f} BTC (~${target_spend:.2f})")
                    _broker.execute_trade("BUY", qty, price)
                else:
                    print(f"   >> [HOLD] Valid Signal, but Alloc (${target_spend:.2f}) too small.")
            else:
                print(f"   >> [HOLD] BUY Signal Rejected by Regime ({regime})")

        elif raw_signal == "SELL":
            if (position * price) > 5.0:
                print(f"   >> 🦅 ROUTING ORDER: SELL {position:.6f} BTC")
                _broker.execute_trade("SELL", position, price)
            else:
                print("   >> [HOLD] SELL Signal, but no BTC to sell.")
                
    except Exception as e:
        print(f"   >> ❌ EXECUTION ERROR: {e}")

if __name__ == "__main__":
    generate_signals()