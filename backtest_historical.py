import pandas as pd
import pandas_ta as ta
import numpy as np
import joblib
import sys
import os
import warnings
from pathlib import Path

# --- SILENCE WARNINGS ---
warnings.filterwarnings("ignore", category=UserWarning, module='sklearn')

# --- CONFIGURATION ---
INITIAL_CAPITAL = 10000.0
TRADE_FEE = 0.001
SLIPPAGE = 0.0005
COOLDOWN_CANDLES = 24  
MODELS_DIR = Path("moonwire/models")
DATA_PATH = Path("flight_recorder.csv")
RISK_PER_TRADE = 0.50 # <-- NEW: Only bet 50% of cash on each trade

# --- SETUP PATHS ---
current_dir = Path(os.getcwd())
if str(current_dir) not in sys.path:
    sys.path.append(str(current_dir))

def prepare_data(filepath):
    print(f"⏳ Loading data from {filepath}...")
    df = pd.read_csv(filepath)
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df.sort_values('Timestamp', inplace=True)
    df.reset_index(drop=True, inplace=True)
    
    print("   Calculating technical indicators...")
    df['RSI'] = ta.rsi(df['Price'], length=14)
    bband = ta.bbands(df['Price'], length=20, std=2)
    try:
        lower_col = next(c for c in bband.columns if c.startswith("BBL"))
        upper_col = next(c for c in bband.columns if c.startswith("BBU"))
        df['BB_Pos'] = (df['Price'] - bband[lower_col]) / (bband[upper_col] - bband[lower_col])
    except:
        df['BB_Pos'] = 0.5 
    df['Vol_Z'] = (df['Volume'] - df['Volume'].rolling(20).mean()) / df['Volume'].rolling(20).std()
    
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    print(f"✅ Data Ready: {len(df)} rows.")
    return df

def load_models():
    models = {}
    try:
        models['gradient_boost'] = joblib.load(MODELS_DIR / "gradient_boost.pkl")
        models['random_forest'] = joblib.load(MODELS_DIR / "random_forest.pkl")
        print(f"✅ Loaded {list(models.keys())}")
    except Exception as e:
        print(f"❌ Error loading models: {e}")
        exit()
    return models

def run_backtest():
    df = prepare_data(DATA_PATH)
    models = load_models()
    
    cash = INITIAL_CAPITAL
    btc_balance = 0.0
    equity_curve = []
    
    trades_executed = 0
    last_trade_index = -COOLDOWN_CANDLES 
    
    print(f"\n🚀 STARTING HISTORICAL BACKTEST (Run 3 Logic + {RISK_PER_TRADE*100}% Risk)...")
    print(f"   Range: {df['Timestamp'].iloc[0]} to {df['Timestamp'].iloc[-1]}")
    
    feature_cols = ['RSI', 'BB_Pos', 'Vol_Z'] # SIMPLE FEATURES
    active_model = models['random_forest']

    for i, row in df.iterrows():
        current_price = row['Price']
        timestamp = row['Timestamp']
        
        equity = cash + (btc_balance * current_price)
        equity_curve.append({'timestamp': timestamp, 'equity': equity})
        
        # Check Cooldown
        if i - last_trade_index < COOLDOWN_CANDLES:
            continue

        # GET UNLEASHED SIGNAL
        features_df = pd.DataFrame([[row['RSI'], row['BB_Pos'], row['Vol_Z']]], columns=feature_cols)
        pred = active_model.predict(features_df)[0]
        
        if pred == 1:
            signal = "BUY"
        else:
            signal = "SELL"

        # EXECUTE TRADES (WITH SIZING)
        trade_happened = False
        
        if signal == "BUY" and cash > 10:
            # SIZING LOGIC: Only invest RISK_PER_TRADE % of available cash
            investable_cash = cash * RISK_PER_TRADE 
            
            trade_value = investable_cash
            fee = trade_value * TRADE_FEE
            net_trade_value = trade_value - fee
            
            btc_bought = net_trade_value / (current_price * (1 + SLIPPAGE))
            
            btc_balance += btc_bought
            cash -= trade_value # Deduct the invested amount
            trade_happened = True
            
        elif signal == "SELL" and btc_balance > 0.0001:
            # Sell 100% of holdings on a Sell Signal
            trade_value = btc_balance * current_price * (1 - SLIPPAGE)
            fee = trade_value * TRADE_FEE
            cash_received = trade_value - fee
            
            cash += cash_received
            btc_balance = 0.0
            trade_happened = True

        if trade_happened:
            trades_executed += 1
            last_trade_index = i 

        if i % 50000 == 0:
            print(f"   Processed {i} candles... Equity: ${equity:.2f}")

    final_equity = cash + (btc_balance * df['Price'].iloc[-1])
    total_return = (final_equity - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100
    
    equity_series = pd.Series([e['equity'] for e in equity_curve])
    rolling_max = equity_series.cummax()
    drawdown = (equity_series - rolling_max) / rolling_max
    max_drawdown = drawdown.min() * 100
    
    start_price = df['Price'].iloc[0]
    end_price = df['Price'].iloc[-1]
    bnh_return = (end_price - start_price) / start_price * 100

    print("\n" + "="*40)
    print(f"🏁 BACKTEST COMPLETE (The Survivor)")
    print("="*40)
    print(f"Start Date:     {df['Timestamp'].iloc[0]}")
    print(f"End Date:       {df['Timestamp'].iloc[-1]}")
    print(f"Trades:         {trades_executed}")
    print("-" * 20)
    print(f"Initial Cash:   ${INITIAL_CAPITAL:,.2f}")
    print(f"Final Equity:   ${final_equity:,.2f}")
    print(f"Total Return:   {total_return:.2f}%")
    print(f"Max Drawdown:   {max_drawdown:.2f}%")
    print("-" * 20)
    print(f"Bitcoin B&H:    {bnh_return:.2f}%")
    print("="*40)

if __name__ == "__main__":
    run_backtest()