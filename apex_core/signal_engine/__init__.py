"""
Signal Engine Core

Shared ML/Governance/Validation engine for multi-asset signal generation platforms.

Modules:
- ml: Machine learning (inference, metrics, backtest, tuning, regime detection, cross-validation)
- validation: Model validation and feedback reliability
- threshold: Threshold optimization and simulation
- analytics: Data processing and origin utilities
- interfaces: Abstract interfaces for data providers
- governance: Model governance and lifecycle management
- performance: Performance monitoring
- core: Core utilities
"""

__version__ = "1.0.0"
__author__ = "Signal Engine Team"

# Use relative imports within the package
from .interfaces import (
    PriceProvider,
    SentimentProvider,
    MarketDataProvider,
)

# Import submodules for easier access
from . import ml
from . import validation
from . import threshold
from . import analytics
from . import interfaces

__all__ = [
    # Version info
    "__version__",
    "__author__",
    # Interfaces
    "PriceProvider",
    "SentimentProvider",
    "MarketDataProvider",
    # Submodules
    "ml",
    "validation",
    "threshold",
    "analytics",
    "interfaces",
]
