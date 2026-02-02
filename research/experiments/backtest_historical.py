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
RISK_PER_TRADE = 0.50 # 50% Sizing
MODELS_DIR = Path("moonwire/models")
DATA_PATH = Path("flight_recorder.csv")

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
    # 1. Standard Features for ML
    df['RSI'] = ta.rsi(df['Price'], length=14)
    bband = ta.bbands(df['Price'], length=20, std=2)
    try:
        lower_col = next(c for c in bband.columns if c.startswith("BBL"))
        upper_col = next(c for c in bband.columns if c.startswith("BBU"))
        df['BB_Pos'] = (df['Price'] - bband[lower_col]) / (bband[upper_col] - bband[lower_col])
    except:
        df['BB_Pos'] = 0.5 
    df['Vol_Z'] = (df['Volume'] - df['Volume'].rolling(20).mean()) / df['Volume'].rolling(20).std()
    
    # 2. Features for HMM (Must match train_hmm.py)
    df['Returns'] = df['Price'].pct_change()
    df['Vol_20'] = df['Returns'].rolling(window=20).std()
    
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    print(f"✅ Data Ready: {len(df)} rows.")
    return df

def load_models():
    models = {}
    try:
        models['bull'] = joblib.load(MODELS_DIR / "random_forest.pkl")
        models['bear'] = joblib.load(MODELS_DIR / "short_forest.pkl")
        models['hmm']  = joblib.load(MODELS_DIR / "hmm_model.pkl")
        print(f"✅ Loaded Tri-Brain System: {list(models.keys())}")
    except Exception as e:
        print(f"❌ Error loading models: {e}")
        exit()
    return models

def run_backtest():
    df = prepare_data(DATA_PATH)
    models = load_models()
    bull_model = models['bull']
    bear_model = models['bear']
    hmm_model  = models['hmm']
    
    # --- PRE-CALCULATE REGIMES (The Manager) ---
    print("🧠 The HMM Manager is categorizing the market...")
    X_hmm = df[['Returns', 'Vol_20']].values
    df['Regime'] = hmm_model.predict(X_hmm)
    
    # Identify the CRASH state (State 2 in your previous run)
    # We define it dynamically as the state with lowest average return
    state_avgs = df.groupby('Regime')['Returns'].mean()
    CRASH_STATE = state_avgs.idxmin()
    print(f"🐻 CRASH STATE IDENTIFIED: State {CRASH_STATE}")
    
    cash = INITIAL_CAPITAL
    btc_balance = 0.0
    equity_curve = []
    
    trades_executed = 0
    last_trade_index = -COOLDOWN_CANDLES 
    
    print(f"\n🚀 STARTING BACKTEST V3.0 (HMM Swarm + {RISK_PER_TRADE*100}% Risk)...")
    
    feature_cols = ['RSI', 'BB_Pos', 'Vol_Z'] 

    for i, row in df.iterrows():
        current_price = row['Price']
        
        # EQUITY CALC
        equity = cash + (btc_balance * current_price)
        equity_curve.append({'timestamp': row['Timestamp'], 'equity': equity})
        
        if i - last_trade_index < COOLDOWN_CANDLES:
            continue

        # --- THE HMM GATING LOGIC ---
        regime = row['Regime']
        features_df = pd.DataFrame([[row['RSI'], row['BB_Pos'], row['Vol_Z']]], columns=feature_cols)
        
        signal = "NEUTRAL"
        
        # LOGIC TREE:
        if regime == CRASH_STATE:
            # 🚨 IN CRASH STATE: Only listen to the BEAR
            is_bear = bear_model.predict(features_df)[0]
            if is_bear == 1:
                signal = "SHORT"
            else:
                signal = "CASH" # Stay out if Bear isn't sure
        else:
            # 🌤️ IN NORMAL STATE: Only listen to the BULL
            is_bull = bull_model.predict(features_df)[0]
            if is_bull == 1:
                signal = "BUY"
            else:
                signal = "CASH" # Stay out if Bull isn't sure

        # --- EXECUTION ENGINE ---
        trade_happened = False
        
        # 1. BUY
        if signal == "BUY":
            if btc_balance < 0: # Cover Short
                cost = abs(btc_balance) * current_price * (1 + SLIPPAGE)
                cash -= cost
                btc_balance = 0.0
                trade_happened = True
            if cash > 10 and btc_balance == 0: # Open Long
                invest_amount = cash * RISK_PER_TRADE
                fee = invest_amount * TRADE_FEE
                btc_bought = (invest_amount - fee) / (current_price * (1 + SLIPPAGE))
                btc_balance += btc_bought
                cash -= invest_amount
                trade_happened = True

        # 2. SHORT
        elif signal == "SHORT":
            if btc_balance > 0: # Sell Long
                revenue = btc_balance * current_price * (1 - SLIPPAGE)
                cash += revenue - (revenue * TRADE_FEE)
                btc_balance = 0.0
                trade_happened = True
            if btc_balance == 0: # Open Short
                collateral = cash * RISK_PER_TRADE
                fee = collateral * TRADE_FEE
                short_value = collateral - fee
                btc_sold = short_value / (current_price * (1 - SLIPPAGE))
                btc_balance -= btc_sold 
                cash += short_value 
                trade_happened = True

        # 3. CASH
        elif signal == "CASH":
            if btc_balance != 0: # Close All
                if btc_balance > 0:
                    revenue = btc_balance * current_price * (1 - SLIPPAGE)
                    cash += revenue - (revenue * TRADE_FEE)
                else:
                    cost = abs(btc_balance) * current_price * (1 + SLIPPAGE)
                    cash -= cost
                btc_balance = 0.0
                trade_happened = True

        if trade_happened:
            trades_executed += 1
            last_trade_index = i 

        if i % 50000 == 0:
            print(f"   Processed {i} candles... Equity: ${equity:.2f}")

    # FINAL REPORT
    final_equity = cash + (btc_balance * df['Price'].iloc[-1])
    total_return = (final_equity - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100
    
    eq_df = pd.DataFrame(equity_curve)
    eq_df['returns'] = eq_df['equity'].pct_change()
    ann_return = eq_df['returns'].mean() * 8760 
    ann_vol = eq_df['returns'].std() * np.sqrt(8760)
    sharpe = (ann_return - 0.04) / ann_vol if ann_vol != 0 else 0
    
    roll_max = eq_df['equity'].cummax()
    drawdown = (eq_df['equity'] - roll_max) / roll_max
    max_drawdown = drawdown.min() * 100
    
    print("\n" + "="*40)
    print(f"🏁 BACKTEST COMPLETE (V3.0 HMM Swarm)")
    print("="*40)
    print(f"Trades:         {trades_executed}")
    print(f"Final Equity:   ${final_equity:,.2f}")
    print(f"Total Return:   {total_return:.2f}%")
    print(f"Max Drawdown:   {max_drawdown:.2f}%")
    print(f"Sharpe Ratio:   {sharpe:.2f}")
    print("="*40)

if __name__ == "__main__":
    run_backtest()