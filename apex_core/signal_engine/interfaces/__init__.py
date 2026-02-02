"""
Abstract interfaces for product-specific implementations.

Products (moonwire-backend, alphaengine) must implement these interfaces
to integrate with the core signal engine.
"""

from .data_providers import (
    PriceProvider,
    SentimentProvider,
    MarketDataProvider,
)

__all__ = [
    "PriceProvider",
    "SentimentProvider",
    "MarketDataProvider",
]
