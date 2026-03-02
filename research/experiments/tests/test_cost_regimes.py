# research/experiments/tests/test_cost_regimes.py
"""
Unit tests for cost regime resolution. Asserts that:
- cost_regime=retail_launch sets fee_bps=120, slippage_bps=10 in resolved config.
- CLI overrides win: cost_regime=retail_launch + fee_bps=50 => fee_bps=50, slippage_bps=10.
"""

import sys
import unittest
from pathlib import Path

# Add experiments to path so we can import cost_regimes
_EXPERIMENTS = Path(__file__).resolve().parents[1]
if str(_EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(_EXPERIMENTS))

from cost_regimes import (
    COST_REGIME_CUSTOM,
    COST_REGIME_RETAIL_LAUNCH,
    resolve_cost_params,
)


class TestCostRegimes(unittest.TestCase):
    def test_retail_launch_sets_fee_and_slippage_defaults(self):
        fee_bps, slippage_bps = resolve_cost_params(
            COST_REGIME_RETAIL_LAUNCH,
            fee_bps_cli=None,
            slippage_bps_cli=None,
        )
        self.assertEqual(fee_bps, 120.0)
        self.assertEqual(slippage_bps, 10.0)

    def test_cli_fee_override_wins_over_regime(self):
        fee_bps, slippage_bps = resolve_cost_params(
            COST_REGIME_RETAIL_LAUNCH,
            fee_bps_cli=50,
            slippage_bps_cli=None,
        )
        self.assertEqual(fee_bps, 50.0)
        self.assertEqual(slippage_bps, 10.0)

    def test_cli_slippage_override_wins_over_regime(self):
        fee_bps, slippage_bps = resolve_cost_params(
            COST_REGIME_RETAIL_LAUNCH,
            fee_bps_cli=None,
            slippage_bps_cli=3,
        )
        self.assertEqual(fee_bps, 120.0)
        self.assertEqual(slippage_bps, 3.0)

    def test_custom_uses_custom_defaults_when_no_cli(self):
        fee_bps, slippage_bps = resolve_cost_params(
            COST_REGIME_CUSTOM,
            fee_bps_cli=None,
            slippage_bps_cli=None,
            custom_fee_bps=10.0,
            custom_slippage_bps=5.0,
        )
        self.assertEqual(fee_bps, 10.0)
        self.assertEqual(slippage_bps, 5.0)


if __name__ == "__main__":
    unittest.main()
