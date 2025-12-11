# apex_core\signal_engine\ml\infer.py
from __future__ import annotations
import sys
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
import os # <-- ADDED: Need os for path resolution later

# --- Bridge to Moonwire ---
try:
    current = Path(__file__).resolve()
    # Assuming standard monorepo path: apex_core/signal_engine/ml/infer.py
    root_dir = current.parents[3] 
    if not (root_dir / "moonwire").exists():
        root_dir = Path(os.getcwd())

    if str(root_dir) not in sys.path:
        sys.path.append(str(root_dir))
    
    from moonwire.strategies.ml_adapter import MLStrategyAdapter
except ImportError:
    MLStrategyAdapter = None

try: from src.cache_instance import cache
except ImportError: cache = None

logger = logging.getLogger(__name__)

def vectorize_features(features, feature_order):
    return np.array([[float(features.get(k, 0.0) or 0.0) for k in feature_order]], dtype=float)

class InferenceEngine:
    def __init__(self, model, feature_order, metadata=None):
        self.model = model
        self.feature_order = feature_order
        self.metadata = metadata or {}
    def predict_proba(self, features, explain=False, top_n=5):
        return {"probability": 0.5}

def infer_asset_signal(symbol: str, strategy: str = None) -> Dict[str, Any]:
    if MLStrategyAdapter is None:
        return {"ok": False, "reason": "Moonwire Adapter not found"}

    try:
        # 1. Get Data
        current_price = 0.0
        df = None
        
        if cache:
            s_data = cache.get_signal(symbol)
            if isinstance(s_data, dict):
                current_price = float(s_data.get('price') or s_data.get('close') or s_data.get('last') or 0.0)

        # Fallback to file if price is 0
        if current_price == 0.0:
            fr_path = root_dir / "flight_recorder.csv"
            if fr_path.exists():
                import pandas as pd
                df = pd.read_csv(fr_path)
                current_price = float(df['Price'].iloc[-1])

        if current_price == 0.0:
            return {"ok": False, "reason": "No price data available"}

        # 2. Execute Moonwire
        models_path = root_dir / "moonwire" / "models"
        adapter = MLStrategyAdapter(symbol, models_dir=models_path)
        
        # --- AMNESIA FIX: PRELOAD HISTORY ---
        # The 'amnesia fix' is a temporary way to pass history to the adapter
        if df is not None and not df.empty:
            history = df['Price'].tail(100).tolist()
            if len(history) > 1:
                adapter.prices.extend(history[:-1])
        # ------------------------------------

        result = adapter.analyze(current_price, strategy=strategy)

        # === START CRITICAL LOGGING CHANGE (Task 2) ===
        # We assume the model filename is returned in result.metadata.model_filename
        
        model_name = "Model Not Logged"
        
        # Heuristic: Extract model name from filename (e.g., random_forest.pkl -> Random Forest)
        if result.get("metadata") and result.get("metadata").get("model_filename"):
            filename = result.get("metadata").get("model_filename").lower()
            if "random_forest" in filename:
                model_name = "Random Forest"
            elif "gradient_boost" in filename:
                model_name = "Gradient Boost"
            elif "logistic_regression" in filename:
                model_name = "Logistic Regression"
            elif "svm" in filename:
                model_name = "SVM"
            # Fallback for unexpected filenames
            else:
                model_name = filename.replace(".pkl", "").replace("_", " ").title()
        # === END CRITICAL LOGGING CHANGE ===

        # 3. Translate
        direction_map = {"BUY": "long", "SELL": "short", "HOLD": None}
        direction = direction_map.get(result.get("signal"), None)
        conf_percent = result.get("confidence", 0.0)
        confidence = conf_percent / 100.0

        return {
            "ok": True,
            "direction": direction,
            "confidence": confidence,
            "reason": result.get("reason", "ok"),
            "model_name": model_name, # CRITICAL ADDITION for the log
            "metadata": result
        }

    except Exception as e:
        logger.error("Inference Engine Exception: %s", e)
        return {"ok": False, "reason": f"Exception: {e}"}

__all__ = ['InferenceEngine', 'infer_asset_signal', 'vectorize_features']