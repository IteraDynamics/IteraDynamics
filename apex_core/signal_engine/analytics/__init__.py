# analytics module
"""
Signal Engine Analytics Module.

Provides comprehensive analytics for signal generation systems including:
- Origin normalization and data processing
- Burst detection and trend analysis
- Correlation and lead-lag analysis
- Volatility regime classification
- Source quality metrics and yield scoring
- Attention scoring and nowcasting

Example:
    ```python
    from . import (
        compute_bursts,
        compute_origin_correlations,
        compute_lead_lag,
        compute_volatility_regimes,
        compute_nowcast_attention
    )

    events = [
        {"timestamp": "2025-01-01T10:00:00Z", "origin": "twitter", "type": "flag"},
        {"timestamp": "2025-01-01T10:00:00Z", "origin": "twitter", "type": "trigger"},
        ...
    ]

    # Detect bursts
    bursts = compute_bursts(events, days=7, interval="hour", z_thresh=2.0)

    # Compute correlations
    corr = compute_origin_correlations(events, days=7, interval="day")

    # Attention ranking
    attention = compute_nowcast_attention(events, days=7, interval="hour", top=10)
    ```
"""

# Origin utilities
from .origin_utils import (
    DEFAULT_ALIAS_MAP,
    normalize_origin,
    extract_origin,
    parse_timestamp,
    parse_ts,
    is_within_window,
    stream_jsonl,
    tolerant_jsonl_stream,
    compute_origin_breakdown,
)

# Threshold policy
from .threshold_policy import (
    threshold_for_regime,
)

# Burst detection
from .burst_detection import (
    compute_bursts,
)

# Correlation analysis
from .origin_correlations import (
    compute_origin_correlations,
)

# Trend analysis
from .origin_trends import (
    compute_origin_trends,
)

# Volatility regimes
from .volatility_regimes import (
    compute_volatility_regimes,
)

# Lead-lag analysis
from .lead_lag import (
    compute_lead_lag,
)

# Source metrics
from .source_metrics import (
    compute_source_metrics,
)

# Source yield
from .source_yield import (
    compute_source_yield,
)

# Nowcast attention
from .nowcast_attention import (
    compute_nowcast_attention,
)

__all__ = [
    # Origin utilities
    'DEFAULT_ALIAS_MAP',
    'normalize_origin',
    'extract_origin',
    'parse_timestamp',
    'parse_ts',
    'is_within_window',
    'stream_jsonl',
    'tolerant_jsonl_stream',
    'compute_origin_breakdown',
    # Threshold policy
    'threshold_for_regime',
    # Burst detection
    'compute_bursts',
    # Correlation analysis
    'compute_origin_correlations',
    # Trend analysis
    'compute_origin_trends',
    # Volatility regimes
    'compute_volatility_regimes',
    # Lead-lag analysis
    'compute_lead_lag',
    # Source metrics
    'compute_source_metrics',
    # Source yield
    'compute_source_yield',
    # Nowcast attention
    'compute_nowcast_attention',
]
