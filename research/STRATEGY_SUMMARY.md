# BTC Regime Trend Strategy

## Overview

A capital-preservation focused BTC trading strategy that captures major uptrends while protecting against severe drawdowns. The strategy is designed to be:

- **Profitable**: Makes meaningful returns over time
- **Disciplined**: Not a gambling/Vegas approach
- **Risk-Aware**: Professionally respectable drawdown management

## Final Performance (2019-2025, ~7 years)

| Metric | Strategy | Buy & Hold | Comparison |
|--------|----------|------------|------------|
| Total Return | **470%** | 2,280% | Captures 21% of upside |
| Annual Return | **28%** | ~50% | Strong absolute returns |
| Max Drawdown | **-38%** | -77% | **50% less drawdown** |
| Sharpe Ratio | **1.02** | ~0.7 | Better risk-adjusted |
| Profit Factor | **1.84** | n/a | Strong edge |
| Win Rate | 32% | n/a | Low but profitable |
| Trades | 161 | 1 | ~23/year |

## Strategy Logic

### Entry Conditions (All must be true)
1. Price > 200 SMA + 3% buffer (confirmed uptrend)
2. 50 SMA > 200 SMA (trend confirmation)
3. 200 SMA is rising (momentum check)

### Exit Conditions (Any triggers exit)
1. Price < 200 SMA (trend breakdown)
2. 50 SMA < 200 SMA (trend reversal)
3. Price drops 12% from peak (trailing stop)
4. Position down 10% from entry (hard stop)

### Position Sizing
- Deploy 75% of capital per trade
- Maintain 25% cash buffer for safety

## Key Parameters

```python
RegimeTrendParams(
    regime_sma=200,         # Long-term trend
    confirm_sma=50,         # Confirmation
    entry_buffer_pct=3.0,   # Enter 3% above SMA
    exit_buffer_pct=0.0,    # Exit at SMA
    position_size_pct=75.0, # 75% deployment
    use_trailing_stop=True,
    trailing_stop_pct=12.0, # 12% trail from peak
    max_loss_pct=10.0,      # 10% hard stop
)
```

## Trade Characteristics

- **Average Trade:** Moderate wins, small losses
- **Exit Reasons:**
  - 96% regime-based (trend breakdown)
  - 3% trailing stop (profit protection)
  - 1% hard stop (capital protection)
- **Time in Market:** ~35% (defensive)

## Why This Strategy Works

1. **Trend Following**: BTC trends hard in both directions. Being long during bull markets captures the big moves.

2. **Capital Protection**: The combination of:
   - Waiting for confirmation (3% above SMA)
   - Trailing stop (12% from peak)
   - Hard stop (10% from entry)
   ...prevents catastrophic losses

3. **Simplicity**: Simple rules don't overfit. This strategy has few parameters and clear logic.

4. **Cost Efficiency**: ~23 trades/year means low transaction costs.

## Risks & Limitations

1. **Still has -38% drawdown**: While better than B&H (-77%), this is aggressive by traditional standards. Suitable for investors who can stomach volatility.

2. **Lower return capture**: Only captures ~21% of BTC's upside. Trade-off for reduced risk.

3. **Long underwater periods**: Can go 800+ days without new highs during bear markets.

4. **Backtest only**: Past performance doesn't guarantee future results.

## How to Run

```bash
cd research
python run_regime_trend.py
```

## Files

- `strategies/regime_trend.py` - Strategy logic
- `run_regime_trend.py` - Backtest runner
- `backtests/results/regime_trend_trades.csv` - Trade history
- `backtests/results/regime_trend_equity.csv` - Equity curve

## Quality Assessment

| Check | Status |
|-------|--------|
| Profitable | ✅ +470% |
| Sharpe > 0.5 | ✅ 1.02 |
| Profit Factor > 1.5 | ✅ 1.84 |
| Statistical Significance | ✅ 161 trades |
| Drawdown < B&H | ✅ 50% less |
| Max DD < 50% | ⚠️ -38% (close) |

**Verdict**: Solid, tradeable strategy with professional risk characteristics.


