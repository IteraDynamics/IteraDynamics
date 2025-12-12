# src/signal_generator.py
# V1.0 PRODUCTION - DIRECT MODEL LOADING

from __future__ import annotations
import sys
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')

import os
import json
import logging
import pandas as pd
import pandas_ta as ta
import numpy as np
import joblib
import requests
import time
from pathlib import Path
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List

# --- CONFIGURATION ---
MODELS_DIR = Path("moonwire/models")
MODEL_FILE = "random_forest.pkl" # The V1.0 Survivor
GOVERNANCE_PARAMS_PATH = Path("governance_params.json")

# --- BROKER IMPORT ---
try:
    from src.paper_broker import PaperBroker
except ImportError:
    try:
        from paper_broker import PaperBroker
    except:
        print("⚠️ PAPER BROKER NOT FOUND. Trading disabled.")
        PaperBroker = None

# --- SINGLETON BROKER ---
_broker = PaperBroker() if PaperBroker else None 

def update_market_data(csv_path: Path = Path("flight_recorder.csv")):
    """
    Fetches the latest completed 1H candle from Coinbase and appends it to the CSV.
    """
    try:
        print("   >> Updating Market Data (Coinbase)...")
        url = "https://api.exchange.coinbase.com/products/BTC-USD/candles"
        params = {'granularity': 3600} # 1 Hour
        resp = requests.get(url, params=params, timeout=10)
        data = resp.json()
        
        # Coinbase returns [time, low, high, open, close, volume]
        # Reverse to get oldest first
        data.sort(key=lambda x: x[0]) 
        
        if not csv_path.exists():
            print("   >> CSV missing. Cannot append.")
            return

        df = pd.read_csv(csv_path)
        last_timestamp = pd.to_datetime(df['Timestamp']).max()
        
        new_rows = []
        for candle in data:
            ts = datetime.fromtimestamp(candle[0])
            if ts > last_timestamp:
                new_rows.append({
                    'Timestamp': ts,
                    'Open': candle[3],
                    'High': candle[2],
                    'Low': candle[1],
                    'Close': candle[4],
                    'Volume': candle[5]
                })
        
        if new_rows:
            new_df = pd.DataFrame(new_rows)
            new_df.to_csv(csv_path, mode='a', header=False, index=False)
            print(f"   >> ✅ Added {len(new_rows)} new hourly candle(s). Latest: {new_rows[-1]['Timestamp']}")
        else:
            print("   >> Data is up to date.")

    except Exception as e:
        print(f"   >> ⚠️ Data Update Failed: {e}")

def get_latest_features(csv_path: Path = Path("flight_recorder.csv")):
    """
    Recreates the exact V1.0 Feature Engineering from the backtest.
    """
    try:
        df = pd.read_csv(csv_path)
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
        df.sort_values('Timestamp', inplace=True)
        
        # --- FEATURE ENGINEERING (Must match train_real_models.py) ---
        df['RSI'] = ta.rsi(df['Price'], length=14)
        bband = ta.bbands(df['Price'], length=20, std=2)
        
        # Handle BB names which can vary
        lower_col = next(c for c in bband.columns if c.startswith("BBL"))
        upper_col = next(c for c in bband.columns if c.startswith("BBU"))
        
        df['BB_Pos'] = (df['Price'] - bband[lower_col]) / (bband[upper_col] - bband[lower_col])
        df['Vol_Z'] = (df['Volume'] - df['Volume'].rolling(20).mean()) / df['Volume'].rolling(20).std()
        
        # Get the very last row (the most recent closed candle)
        last_row = df.iloc[-1]
        
        features = pd.DataFrame([[last_row['RSI'], last_row['BB_Pos'], last_row['Vol_Z']]], 
                                columns=['RSI', 'BB_Pos', 'Vol_Z'])
        
        market_data = {
            "price": last_row['Price'],
            "volume": last_row['Volume'],
            "timestamp": last_row['Timestamp']
        }
        
        return features, market_data
        
    except Exception as e:
        print(f"   >> Feature Engineering Failed: {e}")
        return None, None

def generate_signals():
    print(f"[{datetime.now().time()}] Starting Signal Generator (V1.0 PRODUCTION)...")
    
    # 1. Update Data
    update_market_data()
    
    # 2. Load Model
    model_path = MODELS_DIR / MODEL_FILE
    if not model_path.exists():
        print(f"   >> ❌ CRITICAL: Model file not found at {model_path}")
        return
        
    try:
        model = joblib.load(model_path)
        print(f"   >> Loaded {MODEL_FILE}")
    except Exception as e:
        print(f"   >> ❌ Model Load Failed: {e}")
        return

    # 3. Get Features (The Brain Input)
    features, market_data = get_latest_features()
    if features is None:
        print("   >> ❌ No features available. Aborting.")
        return

    # 4. Predict (The Brain Output)
    try:
        prediction = model.predict(features)[0]
        # In random_forest.pkl, 1 = UP (Buy), 0 = DOWN (Sell/Cash)
        signal_str = "BUY" if prediction == 1 else "SELL"
        print(f"   >> [BRAIN] Prediction: {signal_str} (Price: ${market_data['price']:,.2f})")
    except Exception as e:
        print(f"   >> ❌ Prediction Failed: {e}")
        return

    # 5. Execute (The Hands)
    if _broker:
        asset = "BTC"
        cash = _broker.cash
        price = market_data['price']
        
        # CONFIG: V1.0 Sizing (50% Risk)
        RISK_PER_TRADE = 0.50 
        MIN_TRADE_DOLLARS = 20.0
        
        # Check Cooldown (10 mins to prevent double-fire in same hour)
        # In a real hourly loop, this acts as a debounce
        last_trade_ts = _broker.trade_log[-1].get('ts') if _broker.trade_log else None
        cooldown_active = False
        if last_trade_ts:
            last_dt = datetime.fromisoformat(str(last_trade_ts))
            if (datetime.now() - last_dt).total_seconds() < 600: # 10 mins
                cooldown_active = True
        
        if cooldown_active:
             print(f"   >> [SKIPPED] Cooldown Active (Already traded this hour)")
             return

        # EXECUTION LOGIC
        if signal_str == "BUY":
            # Only buy if we have cash and no position (or adding to position)
            # For V1.0 simplicity: If we have cash, deploy 50% of it.
            if cash > MIN_TRADE_DOLLARS:
                trade_amt = cash * RISK_PER_TRADE
                qty = trade_amt / price
                print(f"   >> [EXECUTING] BUY {qty:.6f} BTC (${trade_amt:.2f})")
                _broker.execute_trade("BUY", qty, price)
            else:
                print("   >> [HOLD] Signal is BUY, but insufficient cash (Fully Invested?)")

        elif signal_str == "SELL":
            # Sell everything
            if _broker.positions > 0:
                print(f"   >> [EXECUTING] SELL {_broker.positions:.6f} BTC (Closing Position)")
                _broker.execute_trade("SELL", _broker.positions, price)
            else:
                print("   >> [HOLD] Signal is SELL, but no position to close (Already in Cash)")
        
        # Report
        val = _broker.get_portfolio_value(price)
        print(f"   >> [PORTFOLIO] Cash: ${_broker.cash:,.2f} | Equity: ${val:,.2f}")

if __name__ == "__main__":
    generate_signals()