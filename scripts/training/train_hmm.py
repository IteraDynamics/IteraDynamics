import pandas as pd
import numpy as np
import joblib
from hmmlearn.hmm import GaussianHMM
import matplotlib.pyplot as plt
from pathlib import Path

# --- CONFIG ---
DATA_PATH = Path("flight_recorder.csv")
MODELS_DIR = Path("moonwire/models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# 1. LOAD DATA
print("⏳ Loading data...")
df = pd.read_csv(DATA_PATH)
df['Timestamp'] = pd.to_datetime(df['Timestamp'])
df.sort_values('Timestamp', inplace=True)

# 2. CALCULATE "BEHAVIOR" FEATURES
# HMMs work best with Returns (Growth) and Volatility (Fear)
df['Returns'] = df['Price'].pct_change()
df['Vol_20'] = df['Returns'].rolling(window=20).std()

# Drop NaN (First 20 rows)
df.dropna(inplace=True)

# Prepare Matrix for HMM (Reshape for sklearn)
# We feed it [[Return, Volatility], [Return, Volatility]...]
X = df[['Returns', 'Vol_20']].values

# 3. TRAIN HMM (Unsupervised)
print("🧠 Training Hidden Markov Model (Discovering Regimes)...")
# n_components=3 means we assume 3 market states (Bull, Bear, Sideways)
hmm_model = GaussianHMM(n_components=3, covariance_type="full", n_iter=100, random_state=42)
hmm_model.fit(X)

# 4. PREDICT STATES
hidden_states = hmm_model.predict(X)
df['Regime'] = hidden_states

# 5. ANALYZE THE STATES (Who is who?)
print("\n📊 REGIME DECODER RING:")
print("-" * 60)
print(f"{'State':<6} | {'Avg Return (Hourly)':<20} | {'Volatility (Risk)':<18} | {'Count':<8}")
print("-" * 60)

state_summary = []

for i in range(hmm_model.n_components):
    mask = (hidden_states == i)
    avg_ret = df.loc[mask, 'Returns'].mean() * 100 # In Percent
    avg_vol = df.loc[mask, 'Vol_20'].mean() * 100  # In Percent
    count = np.sum(mask)
    
    state_summary.append((i, avg_ret, avg_vol, count))
    
    print(f"{i:<6} | {avg_ret:>19.4f}% | {avg_vol:>17.4f}% | {count:>8}")

print("-" * 60)

# 6. SAVE THE MODEL
joblib.dump(hmm_model, MODELS_DIR / "hmm_model.pkl")
print(f"✅ HMM Model saved to {MODELS_DIR / 'hmm_model.pkl'}")

# 7. IDENTIFY THE "CRASH" STATE
# The Crash state usually has NEGATIVE Returns and HIGH Volatility
crash_state = sorted(state_summary, key=lambda x: x[1])[0][0] # Sort by Return (Lowest first)
print(f"\n🐻 DETECTED CRASH STATE: State {crash_state}")
print("(This is the state where we will UNLEASH the Short Seller)")