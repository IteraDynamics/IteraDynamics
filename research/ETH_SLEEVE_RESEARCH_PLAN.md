# ETH Sleeve Research Plan

## Objective

Find a long-only ETH strategy sleeve that is genuinely additive to the current BTC Volatility Breakout (VB) sleeve in a combined portfolio, not just profitable in ETH-only isolation.

## Current Decision Baseline

- BTC VB remains the active benchmark sleeve.
- ETH breakout-style candidates tested so far are not additive at portfolio level.
- ETH research continues, but ETH is disabled by default in portfolio construction until acceptance gates are met.

## Acceptance Gates (Must Pass)

A candidate is considered viable only if it passes all of the following when combined with BTC VB:

1. Combined Calmar is not materially worse than BTC-only.
2. Combined Max Drawdown does not exceed BTC-only MaxDD by more than 10 percentage points.
3. Combined CAGR is at least 90% of BTC-only CAGR at 10/5 costs.
4. Combined performance does not collapse at 25/10 costs (no severe negative-CAGR failure mode).
5. Results are directionally consistent across allocator modes (`normalize`, `btc_priority`, `btc_capped_eth`, `eth_when_btc_flat`).

## Research Scope (Next Candidate Families)

Focus on long-only ETH families that are structurally different from breakout-chasing:

1. **Trend Pullback Continuation**
   - Trade in trend direction only; enter on controlled pullbacks.
   - Goal: lower churn and smoother equity than pure breakout entries.

2. **Regime-Gated Mean Reversion**
   - Mean reversion entries only in favorable trend regime.
   - Goal: capture ETH oversold rebounds without constant knife-catching.

3. **Volatility Compression to Expansion**
   - Enter post-compression breakouts with trend/risk filters.
   - Goal: reduce false breakouts and improve return per trade.

4. **State-Machine Hybrid (Trend + MR)**
   - Switch behavior by regime state (trend/mr/flat).
   - Goal: avoid one-style failure across all market conditions.

## 2-Week Execution Plan

### Week 1: Fast Discovery and Pruning

1. Add templates for the four families to `research/experiments/fast_strategy_screener.py`.
2. Run ETH screener in ranking mode (`--mode ranking`) to identify top candidates by composite score.
3. Keep only top 8-12 candidates for harness validation.
4. Reject any candidate with extreme turnover and drawdown profile likely to fail under 25/10 costs.

### Week 2: Harness Validation and Portfolio Additivity

1. Implement top 3-5 candidates as harness strategy modules in `research/strategies/` and runtime mirrors.
2. Run asset-swap harness for ETH-only with both cost sets:
   - 10/5
   - 25/10
3. For survivors, run multi-sleeve with BTC VB across allocator modes.
4. Select only candidates that pass the Acceptance Gates.
5. Produce a short decision memo: **Promote**, **Research-only**, or **Reject**.

## Standard Test Matrix

For every candidate that reaches harness stage:

1. ETH-only harness
   - Cost set A: fee/slippage = 10/5
   - Cost set B: fee/slippage = 25/10
2. BTC+ETH multi-sleeve
   - `normalize`
   - `btc_priority`
   - `btc_capped_eth`
   - `eth_when_btc_flat`

## Required Outputs

For each candidate:

- ETH-only metrics CSV (10/5 and 25/10)
- Multi-sleeve metrics CSV per allocator mode and cost set
- Equity CSVs and audit CSVs
- One consolidated comparison table with:
  - CAGR
  - MaxDD
  - Calmar
  - Profit Factor
  - Exposure
  - Final equity

## Governance Rules

1. Do not modify BTC VB strategy logic during ETH candidate evaluation.
2. Do not modify backtest engine internals during candidate comparisons.
3. Keep costs, lookback, and initial equity fixed for apples-to-apples comparability.
4. Treat any candidate that improves ETH-only but degrades combined BTC+ETH as **non-viable**.

## Immediate Next Task

Implement the first two non-breakout families in the fast screener:

1. Trend Pullback Continuation
2. Regime-Gated Mean Reversion

Then run ETH screener ranking and shortlist the next harness candidates.
