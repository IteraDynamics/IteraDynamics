"""
Threshold Policy Module

Maps volatility regimes to suggested consensus/trigger thresholds.
Used for adjusting signal filtering based on market conditions.

Example:
    ```python
    from . import threshold_for_regime

    regime = "turbulent"
    threshold = threshold_for_regime(regime)
    print(f"Threshold for {regime}: {threshold}")  # 3.0
    ```
"""

from __future__ import annotations


def threshold_for_regime(regime: str) -> float:
    """
    Map volatility regime to a suggested consensus/trigger threshold.

    Args:
        regime: Volatility regime ("calm", "normal", or "turbulent")

    Returns:
        Suggested threshold value:
        - calm: 2.2 (lowers bar - easier to trigger)
        - normal: 2.5 (baseline)
        - turbulent: 3.0 (raises bar - harder to trigger)

    Example:
        >>> threshold_for_regime("calm")
        2.2
        >>> threshold_for_regime("normal")
        2.5
        >>> threshold_for_regime("turbulent")
        3.0
        >>> threshold_for_regime("unknown")  # Falls back to normal
        2.5
    """
    mapping = {
        "calm": 2.2,
        "normal": 2.5,
        "turbulent": 3.0
    }
    return float(mapping.get(regime, 2.5))


__all__ = ["threshold_for_regime"]
