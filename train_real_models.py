import pandas as pd
import numpy as np
import joblib
import os
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import pandas_ta as ta 

# 1. Setup Paths
# We need to save the models where Moonwire looks for them
models_dir = Path("moonwire/models")
models_dir.mkdir(parents=True, exist_ok=True)
data_path = Path("flight_recorder.csv")

print(f"🚀 Starting Training Run...")
print(f"   Input Data: {data_path}")
print(f"   Output Dir: {models_dir}")

# 2. Load & Prepare Data
if not data_path.exists():
    print("❌ Error: flight_recorder.csv not found!")
    exit()

df = pd.read_csv(data_path)
print(f"   Loaded {len(df)} rows of history.")

# --- NEW FEATURE CALCULATION SECTION (ROBUST) ---
# This calculates the required technical indicators based on raw OHLCV data.
df['RSI'] = ta.rsi(df['Price'], length=14)

# Calculate Bollinger Bands (BB)
bband = ta.bbands(df['Price'], length=20, std=2)

# DYNAMIC COLUMN FINDER: Finds the correct column names automatically
# This prevents KeyErrors whether the lib returns 'BBL_20_2.0' or 'BBL_20_2'
lower_col = next(c for c in bband.columns if c.startswith("BBL"))
upper_col = next(c for c in bband.columns if c.startswith("BBU"))

print(f"   DEBUG: Using BB Cols -> Lower: {lower_col}, Upper: {upper_col}")

# BB_Pos: Position relative to the bands
df['BB_Pos'] = (df['Price'] - bband[lower_col]) / (bband[upper_col] - bband[lower_col])

# Vol_Z (Volatility Z-Score): A simple Z-score of volume
df['Vol_Z'] = (df['Volume'] - df['Volume'].rolling(20).mean()) / df['Volume'].rolling(20).std()
# --- END NEW FEATURE CALCULATION SECTION ---


# --- Feature Engineering ---
# We use the exact features your system is already calculating:
# RSI, BB_Pos (Bollinger Band Position), Vol_Z (Volatility Z-Score)
feature_cols = ['RSI', 'BB_Pos', 'Vol_Z']

# Drop any rows where these features are missing (mainly the initial 20 rows needed for calculation)
df = df.dropna(subset=feature_cols)

# --- Define the Target (The "Truth") ---
# We want to predict if price will be HIGHER in 5 minutes (periods)
LOOKAHEAD = 5 
df['Future_Price'] = df['Price'].shift(-LOOKAHEAD)
df['Target'] = (df['Future_Price'] > df['Price']).astype(int)

# Drop the last N rows because they don't have a future yet
df = df.dropna(subset=['Future_Price'])

print(f"   Training on {len(df)} valid samples (Lookahead: {LOOKAHEAD}m).")

# 3. Split Data
X = df[feature_cols]
y = df['Target']

# Use last 20% of data for testing (Time-series split, no shuffling)
split = int(len(df) * 0.8)
X_train, X_test = X.iloc[:split], X.iloc[split:]
y_train, y_test = y.iloc[:split], y.iloc[split:]

# 4. Train the "Strategy Map" Models
# We train different model types for different strategies
models_to_train = {
    "random_forest.pkl": RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42),
    "gradient_boost.pkl": GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42),
    "logistic_regression.pkl": LogisticRegression(random_state=42),
    "svm.pkl": SVC(probability=True, kernel='rbf', random_state=42)
}

print("\n🧠 Training Phase:")
for filename, model in models_to_train.items():
    # Train
    model.fit(X_train, y_train)
    
    # Test
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    
    # Save
    save_path = models_dir / filename
    joblib.dump(model, save_path)
    
    print(f"   ✅ {filename:<25} | Accuracy: {acc:.2%} | Saved.")

print("\n🎯 Mission Complete. Real models deployed to Moonwire.")
print("   Argus will now use these brains for the next prediction.")