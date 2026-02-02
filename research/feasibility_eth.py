import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

print("🧪 STARTING Itera Dynamics RESEARCH LAB: ETH-USD Feasibility Study (v3)")
print("---------------------------------------------------------------")

# 1. DATA
print(">> 📡 Downloading ETH-USD Hourly Data...")
df = yf.download("ETH-USD", period="730d", interval="1h", progress=False)

if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)

# 2. INDICATORS
print(">> ⚙️  Calculating Indicators...")
df['Returns'] = df['Close'].pct_change()
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
df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
df.dropna(inplace=True)

# 3. TRAIN
features = ['RSI', 'SMA_50', 'SMA_200', 'Volatility', 'Momentum']
X = df[features]
y = df['Target']
split = int(len(df) * 0.8)
X_train, X_test = X.iloc[:split], X.iloc[split:]
y_train, y_test = y.iloc[:split], y.iloc[split:]
prices_test = df['Close'].iloc[split:]

print(f">> 🧠 Training Random Forest on {len(X_train)} hours...")
model = RandomForestClassifier(n_estimators=100, min_samples_split=10, random_state=42)
model.fit(X_train, y_train)

# 4. SIMULATION (MAKER FEES)
print(">> 🔮 Running Backtest (Maker Fee Model)...")
probs = model.predict_proba(X_test)[:, 1]

# --- SETTINGS ---
START_CAPITAL = 10000
COMMISSION = 0.0020  # 0.20% (Standard Maker Fee on Coinbase Advanced)
BUY_THRESHOLD = 0.70 
SELL_THRESHOLD = 0.50
# ----------------

capital = START_CAPITAL
holdings = 0
in_position = False
trades = 0
total_fees_paid = 0

equity_curve = []

for i in range(len(X_test)):
    current_price = prices_test.iloc[i]
    confidence = probs[i]
    
    # BUY LOGIC
    if not in_position and confidence > BUY_THRESHOLD:
        fee = capital * COMMISSION
        total_fees_paid += fee
        capital -= fee
        holdings = capital / current_price
        capital = 0
        in_position = True
        trades += 1
    
    # SELL LOGIC
    elif in_position and confidence < SELL_THRESHOLD:
        gross_value = holdings * current_price
        fee = gross_value * COMMISSION
        total_fees_paid += fee
        capital = gross_value - fee
        holdings = 0
        in_position = False
    
    current_val = capital if not in_position else (holdings * current_price)
    equity_curve.append(current_val)

final_value = equity_curve[-1]
bh_value = (prices_test.iloc[-1] / prices_test.iloc[0]) * START_CAPITAL
bh_return_pct = (bh_value - START_CAPITAL) / START_CAPITAL * 100
algo_return_pct = (final_value - START_CAPITAL) / START_CAPITAL * 100

print("\n💰 --- REALITY REPORT ($10k Start) ---")
print(f"   Settings: Threshold={BUY_THRESHOLD*100}% | Fee={COMMISSION*100}% (Maker)")
print(f"   -------------------------------------")
print(f"   Trades Executed:  {trades}")
print(f"   Total Fees Paid: ${total_fees_paid:.2f}")
print(f"   -------------------------------------")
print(f"   Buy & Hold End Value: ${bh_value:.2f} ({bh_return_pct:.2f}%)")
print(f"   Argus End Value:      ${final_value:.2f} ({algo_return_pct:.2f}%)")

diff = final_value - bh_value
print(f"\n📊 Performance vs Market: ${diff:.2f}")

if final_value > bh_value:
    print("✅ VERDICT: PASSED. With Limit Orders, Argus beats the market.")
else:
    print("❌ VERDICT: FAILED. Even with Maker fees, it's not enough.")