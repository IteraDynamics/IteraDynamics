import pandas as pd
import numpy as np
import joblib
import os
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score
import pandas_ta as ta 

# 1. Setup Paths
models_dir = Path("moonwire/models")
models_dir.mkdir(parents=True, exist_ok=True)
data_path = Path("flight_recorder.csv")

print(f"🚀 Starting Training Run (V1.0 Survivor - Long Only)...")

# 2. Load & Prepare Data
if not data_path.exists():
    print("❌ Error: flight_recorder.csv not found!")
    exit()

df = pd.read_csv(data_path)
print(f"   Loaded {len(df)} rows.")

# --- FEATURE ENGINEERING (SIMPLE) ---
print("   Calculating indicators...")
df['RSI'] = ta.rsi(df['Price'], length=14)
bband = ta.bbands(df['Price'], length=20, std=2)
lower_col = next(c for c in bband.columns if c.startswith("BBL"))
upper_col = next(c for c in bband.columns if c.startswith("BBU"))
df['BB_Pos'] = (df['Price'] - bband[lower_col]) / (bband[upper_col] - bband[lower_col])
df['Vol_Z'] = (df['Volume'] - df['Volume'].rolling(20).mean()) / df['Volume'].rolling(20).std()

feature_cols = ['RSI', 'BB_Pos', 'Vol_Z']
df = df.dropna(subset=feature_cols)

# --- TARGET (24H Swing) ---
LOOKAHEAD = 24 
df['Future_Price'] = df['Price'].shift(-LOOKAHEAD)
df['Target'] = (df['Future_Price'] > df['Price']).astype(int)
df = df.dropna(subset=['Future_Price'])

# 3. Train
X = df[feature_cols]
y = df['Target']
split = int(len(df) * 0.8)
X_train, X_test = X.iloc[:split], X.iloc[split:]
y_train, y_test = y.iloc[:split], y.iloc[split:]

models = {
    "random_forest.pkl": RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42),
    "gradient_boost.pkl": GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
}

print("\n🧠 Training Phase:")
for name, model in models.items():
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))
    joblib.dump(model, models_dir / name)
    print(f"   ✅ {name:<20} | Accuracy: {acc:.2%} | Saved.")

print("\n🎯 V1.0 Models Deployed.")