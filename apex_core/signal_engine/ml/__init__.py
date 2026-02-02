# ml module
"""
Signal Engine ML module.

Core machine learning functionality for signal generation systems including:
- Performance metrics
- Cross-validation
- Hyperparameter tuning
- Backtesting

NOTE: Additional functionality available elsewhere:
  - Inference: apex_core.infer (infer_score, infer_score_ensemble, etc.)
  - Regime Detection: apex_core.regime_detector (MarketRegimeDetector class)
"""

from .metrics import (
    compute_accuracy_by_version,
    rolling_precision_recall_snapshot,
)
from .backtest import (
    Trade,
    run_backtest,
)
from .cv_eval import (
    FoldStats,
    compute_fold_stats,
    compute_future_return,
    walk_forward_cv,
    time_series_split,
)
from .tuner import (
    extract_backtest_metrics,
    aggregate_metrics,
    objective_score,
    grid_search_thresholds,
)

__all__ = [
    # Metrics
    'compute_accuracy_by_version',
    'rolling_precision_recall_snapshot',
    # Backtesting
    'Trade',
    'run_backtest',
    # Cross-Validation
    'FoldStats',
    'compute_fold_stats',
    'compute_future_return',
    'walk_forward_cv',
    'time_series_split',
    # Hyperparameter Tuning
    'extract_backtest_metrics',
    'aggregate_metrics',
    'objective_score',
    'grid_search_thresholds',
]
