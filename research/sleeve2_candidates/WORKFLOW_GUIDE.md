# GitHub Actions Workflow Guide - Sleeve 2 Backtest

## Overview

Automated backtest and selection system for Sleeve 2 candidates using your existing research infrastructure.

**Workflow:** `.github/workflows/sleeve2_backtest.yml`

---

## What It Does

### 1. **Smoke Tests**
- Imports all 4 strategies (Core v2, Sleeve 2A/B/C)
- Verifies `generate_intent()` callable
- Fails fast if any imports broken

### 2. **Backtests (2023-2025)**

Tests each strategy independently on your hardest period:
- **Core v2** with locked parameters:
  ```bash
  SG_CORE_ENABLE_MACRO_FILTER=1
  SG_CORE_MACRO_EMA_LEN=2000
  SG_CORE_MACRO_EXPO_CAP_BEAR=0.00
  ```
- **Sleeve 2A** (RSI mean reversion)
- **Sleeve 2B** (Bollinger mean reversion)
- **Sleeve 2C** (Hybrid RSI+BB)

**Output metrics:**
- Total Return, CAGR, Max Drawdown
- Calmar, Sortino
- Avg Exposure, Time in Market

### 3. **Crash Window Diagnostics (2021-2022)**

Runs `crash_window_report.py` for each strategy:
- Window: 2021-07-01 to 2022-12-31
- Lookback: 200 bars
- Captures: DD proxy, Avg exposure, Time exposure >50%

### 4. **Guardrail Filtering**

Applies hard filters to Sleeve 2 candidates (NOT Core):

| Guardrail | Threshold | Rationale |
|-----------|-----------|-----------|
| Crash DD | >= -0.15 | Prevents catastrophic drawdown in bear |
| Time Expo >50% | <= 0.02 | Avoids high-exposure during crash |
| Crash Avg Expo | <= 0.15 | Conservative positioning in downturn |
| 2023-25 Time in Market | >= 0.05 | Must participate meaningfully |

**Result:** Only candidates passing ALL guardrails advance.

### 5. **Winner Selection**

Among survivors, ranks by:
1. **Highest Calmar** (primary)
2. **Lower Max DD** (tie-breaker 1)
3. **Higher Time in Market** (tie-breaker 2)

**Output:** `results/summary.json` with winner + full metrics

---

## How to Run

### Option 1: Manual Trigger (Recommended)

1. Go to: https://github.com/IteraDynamics/IteraDynamics/actions
2. Click "Sleeve 2 Backtest & Selection"
3. Click "Run workflow"
4. Select branch: `feature/sleeve2-candidates`
5. Click green "Run workflow" button

### Option 2: Automatic on Push

Pushes to `feature/sleeve2-candidates` trigger automatically.

---

## Required Files

The workflow expects these datasets in your repo:

```
data/
├── btcusd_3600s_2023-01-01_to_2025-12-30.csv  # Main test period
└── btcusd_3600s_2019-01-01_to_2025-12-30.csv  # For crash window slicing
```

**If missing, workflow will fail fast** with clear error.

---

## Output Artifacts

After successful run:

### Files Generated

```
results/
├── core_v2_2023_2025.json          # Core backtest metrics
├── sleeve2a_2023_2025.json         # Candidate A metrics
├── sleeve2b_2023_2025.json         # Candidate B metrics
├── sleeve2c_2023_2025.json         # Candidate C metrics
├── core_v2_crash_window.csv        # Core crash diagnostics
├── sleeve2a_crash_window.csv       # Candidate A crash
├── sleeve2b_crash_window.csv       # Candidate B crash
├── sleeve2c_crash_window.csv       # Candidate C crash
└── summary.json                    # FINAL RESULTS

logs/
├── core_v2_backtest.log
├── sleeve2a_backtest.log
├── sleeve2b_backtest.log
├── sleeve2c_backtest.log
├── core_v2_crash.log
├── sleeve2a_crash.log
├── sleeve2b_crash.log
└── sleeve2c_crash.log
```

### Downloading Artifacts

1. Go to workflow run page
2. Scroll to "Artifacts" section at bottom
3. Click "sleeve2-backtest-results" to download ZIP
4. Extract to review all files

---

## Summary JSON Structure

```json
{
  "timestamp": "2026-02-22T13:10:00Z",
  "test_period": "2023-2025",
  "crash_window": "2021-07-01 to 2022-12-31",
  "strategies_tested": ["core_v2", "sleeve2a", "sleeve2b", "sleeve2c"],
  "survivors": ["sleeve2a", "sleeve2c"],
  "winner": "sleeve2c",
  "winner_reason": "Highest Calmar (2.14) among survivors",
  "results": {
    "core_v2": { /* full metrics */ },
    "sleeve2a": { /* full metrics */ },
    "sleeve2b": { /* full metrics */ },
    "sleeve2c": { /* full metrics */ }
  }
}
```

---

## Interpreting Results

### Scenario 1: Clear Winner

```
🏆 WINNER: SLEEVE2C
   Reason: Highest Calmar (2.14) among survivors
   
   Performance:
   - CAGR: 15.23%
   - Calmar: 2.14
   - Max DD: -7.12%
```

**Action:** Lock Sleeve 2C as your official Sleeve 2.

### Scenario 2: No Survivors

```
❌ NO WINNER
   Reason: No candidates passed guardrails. Recommend redesigning Sleeve 2.
```

**Action:** All candidates too risky in crash window. Redesign Sleeve 2 with:
- Lower base exposure
- Tighter exit triggers
- More conservative regime gates

### Scenario 3: Multiple Survivors, Close Calmars

Review tie-breakers:
- Prefer strategy with **lower Max DD** (smoother ride)
- If still tied, prefer **higher Time in Market** (more participation)

---

## Guardrail Tuning

If guardrails are too strict/loose, edit workflow file:

```yaml
crash_dd_ok = crash.get('dd_proxy', 0) >= -0.15        # Current: -15% max DD
crash_time_expo_ok = crash.get('time_exposure_gt_50pct', 1) <= 0.02  # Current: 2% time
crash_avg_expo_ok = crash.get('avg_exposure', 1) <= 0.15              # Current: 15% avg
time_in_market_ok = bt.get('time_in_market', 0) >= 0.05               # Current: 5% min
```

Adjust thresholds based on your risk tolerance.

---

## Troubleshooting

### Workflow Fails: "Data file not found"

**Fix:** Add required CSV files to `data/` directory:
```bash
# Ensure files exist
ls data/btcusd_3600s_2023-01-01_to_2025-12-30.csv
ls data/btcusd_3600s_2019-01-01_to_2025-12-30.csv
```

### Workflow Fails: "Strategy import error"

**Fix:** Check smoke test logs. Likely:
- Typo in strategy module name
- Missing dependency in requirements
- Syntax error in strategy file

### All Candidates Fail Guardrails

**Diagnosis:** Strategies too aggressive for 2021-2022 crash.

**Solutions:**
1. Lower base exposure in strategy code
2. Add tighter PANIC exit logic
3. Widen guardrail thresholds (if risk-tolerant)

### Winner Has Low Calmar (<1.0)

**Diagnosis:** Even winner isn't great on 2023-2025.

**Solutions:**
1. Accept lower Calmar if it still beats Core
2. Redesign Sleeve 2 entirely
3. Test on different period to confirm

---

## Next Steps After Winner Selected

1. **Review full metrics:**
   - Download artifacts
   - Check trade log (if available)
   - Verify equity curve shape

2. **Lock Sleeve 2 in production:**
   ```bash
   export ARGUS_STRATEGY_MODULE="research.strategies.sg_mean_reversion_c"  # example
   export ARGUS_STRATEGY_FUNC="generate_intent"
   ```

3. **Design Layer 3:**
   - Blend Core + Sleeve 2 (winner)
   - Regime-weighted allocation
   - Capital routing logic

4. **Move to Sleeve 3:**
   - Different mandate (breakout/convex)
   - Orthogonal to Sleeve 2
   - Test full 3-sleeve portfolio

---

## Customization

### Adding More Candidates

Edit workflow to include Sleeve 2D, 2E, etc.:

```yaml
- name: Backtest Sleeve 2D (2023-2025)
  env:
    ARGUS_STRATEGY_MODULE: "research.strategies.sg_mean_reversion_d"
    ARGUS_STRATEGY_FUNC: "generate_intent"
    ARGUS_DATA_FILE: "../../data/btcusd_3600s_2023-01-01_to_2025-12-30.csv"
    ARGUS_OUTPUT_FILE: "../../results/sleeve2d_2023_2025.json"
  run: |
    cd runtime/argus
    python -c "import sys; sys.path.insert(0, r'.'); from research.harness.backtest_runner import main; main()" \
      2>&1 | tee ../../logs/sleeve2d_backtest.log
```

Add to crash window + summary script.

### Changing Test Period

Replace `2023-01-01_to_2025-12-30.csv` references with your desired range.

### Different Crash Window

Edit `--start` and `--end` in crash_window_report steps:

```yaml
--start 2020-03-01 \
--end 2020-06-30 \
```

---

## Performance

**Typical runtime:** 5-10 minutes on GitHub Actions (Ubuntu)

- Smoke tests: <10s
- Each backtest: 1-2 min
- Crash reports: 30s each
- Summary generation: <5s

**Resource usage:** ~2GB RAM, standard GitHub runner handles easily.

---

## Questions?

Check the workflow run logs:
1. Click on failed step
2. Expand log output
3. Look for Python tracebacks or error messages

Most common issues:
- Missing data files
- Import errors (check `sys.path`)
- JSON parsing (check backtest_runner output format)

---

**Workflow is live and ready to run.** Just add your data files and trigger it! 🚀
