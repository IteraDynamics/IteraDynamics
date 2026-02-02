"""
ApexCortex - The Brain

Asset-agnostic signal generation engine for quantitative trading systems.
Used by MoonWire (crypto) and AlphaEngine (securities).

Quick Start:
    # Regime detection
    from apex_core import MarketRegimeDetector
    detector = MarketRegimeDetector()
    regime = detector.detect_regime(df)

    # Signal engine ML tools
    from apex_core.signal_engine import ml
    result = ml.run_backtest(signals, prices)
    
    # Inference (if models are loaded)
    from apex_core import infer_score, infer_score_ensemble
    score = infer_score(features)

Submodules:
    - signal_engine.ml: Backtesting, metrics, cross-validation, tuning
    - signal_engine.analytics: Origin analysis, burst detection, volatility regimes
    - signal_engine.validation: Calibration, reliability metrics
    - signal_engine.governance: Model lifecycle, drift detection, retraining
    - signal_engine.threshold: Threshold optimization and simulation
"""

__version__ = "1.0.0"
__author__ = "Itera Dynamics"

# ============================================================================
# Core Classes
# ============================================================================

from apex_core.regime_detector import (
    MarketRegimeDetector,
    MetaStrategySelector,
    PositionSizer,
)

# ============================================================================
# Inference Functions
# ============================================================================

try:
    from apex_core.infer import (
        infer_score,
        infer_score_ensemble,
        infer_asset_signal,
        compute_volatility_adjusted_threshold,
        model_metadata,
    )
except ImportError:
    # Inference requires additional dependencies (joblib, etc.)
    infer_score = None
    infer_score_ensemble = None
    infer_asset_signal = None
    compute_volatility_adjusted_threshold = None
    model_metadata = None

# ============================================================================
# Signal Engine (organized subpackage)
# ============================================================================

from apex_core import signal_engine

# ============================================================================
# Utilities
# ============================================================================

from apex_core.paths import MODELS_DIR, LOGS_DIR, BASE_DIR

# ============================================================================
# Public API
# ============================================================================

__all__ = [
    # Version
    "__version__",
    "__author__",
    # Core Classes
    "MarketRegimeDetector",
    "MetaStrategySelector", 
    "PositionSizer",
    # Inference
    "infer_score",
    "infer_score_ensemble",
    "infer_asset_signal",
    "compute_volatility_adjusted_threshold",
    "model_metadata",
    # Submodules
    "signal_engine",
    # Paths
    "MODELS_DIR",
    "LOGS_DIR",
    "BASE_DIR",
]

