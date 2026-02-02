import os
import joblib
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.datasets import make_classification

# 1. Setup Directories
models_dir = Path("moonwire/models")
models_dir.mkdir(parents=True, exist_ok=True)

print(f"📍 Generating dummy models in: {models_dir}")

# 2. Create Dummy Training Data (just to make the models valid)
# We need 20 features to match what FeatureBuilder likely outputs
X, y = make_classification(n_samples=100, n_features=20, random_state=42)

# 3. Define the Model List (Matching your Strategy Map)
models = {
    "random_forest.pkl": RandomForestClassifier(n_estimators=10, random_state=42),
    "gradient_boost.pkl": GradientBoostingClassifier(n_estimators=10, random_state=42),
    "logistic_regression.pkl": LogisticRegression(random_state=42),
    "svm.pkl": SVC(probability=True, random_state=42), # Probability=True is needed for confidence scores
    "btc_model_v1.pkl": RandomForestClassifier(n_estimators=10, random_state=99) # Default fallback
}

# 4. Train & Save
for filename, model in models.items():
    print(f"  ... Training & Saving {filename}")
    model.fit(X, y)
    
    save_path = models_dir / filename
    joblib.dump(model, save_path)

print("\n✅ All dummy models created successfully!")
print("   You can now run Moonwire, and the Switchboard will find these files.")