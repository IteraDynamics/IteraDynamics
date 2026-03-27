# MoonWire Signal Integration Guide

**Integration Type:** Signal Feed Artifact (Non-Coupled)  
**Status:** ✅ Production Ready  
**Last Updated:** 2026-03-04

---

## Overview

This guide covers integration of MoonWire ML signal intelligence into Itera's execution framework. The integration uses deterministic signal feed artifacts — no code coupling, no direct imports, clean separation of concerns.

**Architecture:**
```
MoonWire (Signal Generation)
    ↓ [export_signal_feed.py]
Signal Feed Files (.jsonl + manifest.json)
    ↓ [Environment Variable]
Itera Strategy Layer-2 (sg_moonwire_intent_v1.py)
    ↓ [StrategyIntent]
Itera Layer-3 Governance
    ↓
Execution
```

---

## Quick Start

### 1. MoonWire owns signal generation; Itera only consumes

**MoonWire (moonwire-backend)** produces the signal feed via its own export script. Itera does not contain any MoonWire inference or feature logic.

**From moonwire-backend** (canonical producer):

```bash
cd /path/to/moonwire-backend

python scripts/export_signal_feed.py \
  --product BTC-USD \
  --bar_seconds 3600 \
  --start 2019-01-01 \
  --end 2025-12-30 \
  --out feeds/btc_signals \
  --format jsonl \
  --horizon_hours 3 \
  --model_dir models/standard

# Output: feeds/btc_signals.jsonl, feeds/btc_signals.manifest.json
```

**From Itera (consumer):** To check that a feed exists, validate it, or request export by calling moonwire-backend as an external process (no inference in Itera):

```bash
cd /path/to/IteraDynamics_Mono

# Set where the feed should live and where moonwire-backend is
$env:MOONWIRE_SIGNAL_FILE = "C:\path\to\feeds\btc_signals.jsonl"
$env:MOONWIRE_BACKEND_ROOT = "C:\path\to\moonwire-backend"

python scripts/ensure_moonwire_signal_feed.py
```

This script only: (a) checks if the file exists, (b) validates freshness/schema, (c) if missing/stale calls `moonwire-backend/scripts/export_signal_feed.py` via subprocess.

See **`docs/MOONWIRE_FEED_ITERA.md`** for how Itera ensures a feed exists and example PowerShell commands.

**Output Format (JSONL):**
```json
{"timestamp": 1546300800, "probability": 0.6542, "symbol": "BTC-USD", "model_version": "standard_v1.0"}
{"timestamp": 1546304400, "probability": 0.5821, "symbol": "BTC-USD", "model_version": "standard_v1.0"}
...
```

### 2. Configure Itera Strategy

Set environment variables (PowerShell example):

```powershell
# Required: Path to signal feed
$env:MOONWIRE_SIGNAL_FILE = "C:\path\to\moonwire-backend\feeds\btc_signals.jsonl"

# Thresholds (optional, defaults shown)
$env:MOONWIRE_LONG_THRESH = "0.65"   # Probability ≥ 0.65 → ENTER_LONG
$env:MOONWIRE_SHORT_THRESH = "0.35"  # Probability ≤ 0.35 → EXIT (short proxy)

# Features
$env:MOONWIRE_ALLOW_SHORT = "0"      # 1 to enable SHORT signals
$env:MOONWIRE_REQUIRE_EXACT_TS = "1" # 0 to allow fallback to nearest prior signal
```

**Bash/Linux:**
```bash
export MOONWIRE_SIGNAL_FILE="/home/user/moonwire-backend/feeds/btc_signals.jsonl"
export MOONWIRE_LONG_THRESH="0.65"
export MOONWIRE_SHORT_THRESH="0.35"
export MOONWIRE_ALLOW_SHORT="0"
export MOONWIRE_REQUIRE_EXACT_TS="1"
```

### 3. Run Itera Backtest

```bash
cd /path/to/iteradynamics

# Run backtest with MoonWire strategy
python research/backtests/run_backtest.py \
  --strategy sg_moonwire_intent_v1 \
  --product BTC-USD \
  --start 2019-01-01 \
  --end 2025-12-30
```

---

## Configuration Reference

### Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `MOONWIRE_SIGNAL_FILE` | ✅ Yes | - | Path to signals.jsonl file (consumed by Itera) |
| `MOONWIRE_BACKEND_ROOT` | When using ensure script | - | Path to moonwire-backend repo (so Itera can call export_signal_feed.py if feed missing/stale) |
| `MOONWIRE_FEED_MAX_AGE_SECONDS` | No | - | If set, feed older than this is considered stale and re-export may be triggered |
| `MOONWIRE_LONG_THRESH` | No | `0.65` | Probability threshold for LONG (0-1) |
| `MOONWIRE_SHORT_THRESH` | No | `0.35` | Probability threshold for SHORT (0-1) |
| `MOONWIRE_ALLOW_SHORT` | No | `0` | Enable SHORT signals (`1` = yes, `0` = no) |
| `MOONWIRE_REQUIRE_EXACT_TS` | No | `1` | Fail if timestamp not found (`1` = fail, `0` = fallback) |

### Signal Mapping Logic

```python
if probability >= LONG_THRESH:
    action = ENTER_LONG
    confidence = probability

elif ALLOW_SHORT and probability <= SHORT_THRESH:
    action = EXIT  # Short proxy (until SHORT validated)
    confidence = 1.0 - probability

else:
    action = FLAT
    confidence = 0.5
```

---

## Validation

### Run Unit Tests

```bash
cd iteradynamics
pytest tests/test_sg_moonwire_intent.py -v
```

**Tests cover:**
- ✅ Signal mapping (LONG/SHORT/FLAT)
- ✅ Timestamp alignment
- ✅ Determinism (same input → same output)
- ✅ Missing timestamp error handling
- ✅ Config validation
- ✅ Edge cases (thresholds, empty feeds)

### Validate Feed Alignment

Before running a backtest, check timestamp alignment:

```python
from research.strategies.sg_moonwire_intent_v1 import validate_feed_alignment
import pandas as pd

# Load your strategy bars
df = pd.read_csv("your_bars.csv", index_col="timestamp", parse_dates=True)

# Validate alignment
result = validate_feed_alignment(df, "feeds/btc_signals.jsonl")

print(f"Coverage: {result['coverage']:.2%}")
print(f"Matched: {result['matched']} / {result['total_bars']}")
print(f"Missing: {result['missing']}")
```

---

## Architecture Details

### Determinism

**Guaranteed:** Same signal feed + same environment variables = identical intents every run.

**Why it matters:**
- Reproducible backtests
- No randomness, no external API calls, no time-based logic
- Audit-friendly (every decision is traceable)

### Closed-Bar Semantics

**Decision Timeline:**
1. Bar `t` closes
2. Strategy receives data through bar `t`
3. Signal feed provides probability for bar `t`
4. Intent applies to next bar (`t → t+1`)

**No lookahead bias:** Decision at `t` uses only data available at `t`.

### Fail-Fast by Default

**`REQUIRE_EXACT_TS=1` (default):**
- Missing timestamp → immediate `KeyError`
- Forces you to fix data alignment issues
- No silent fallbacks that could mask bugs

**`REQUIRE_EXACT_TS=0` (fallback mode):**
- Missing timestamp → uses nearest prior signal
- Logs age of fallback signal in metadata
- Intent action set to `FLAT` (conservative)
- **⚠️ NOT RECOMMENDED for production**

### Module-Level Caching

**Performance optimization:**
- Signal feed loaded once per Python process
- Subsequent calls read from memory
- No I/O overhead on every bar

**Cache invalidation:**
- Restart Python process to reload feed
- Or set `_SIGNAL_CACHE = None` in testing

---

## Troubleshooting

### Error: "MOONWIRE_SIGNAL_FILE not set"

**Solution:** Set environment variable before running strategy:
```bash
export MOONWIRE_SIGNAL_FILE="/path/to/signals.jsonl"
```

### Error: "Signal file not found"

**Check:**
1. Path is absolute (or relative to working directory)
2. File exists: `ls -la /path/to/signals.jsonl`
3. File was generated: Run `export_signal_feed.py`

### Error: "Timestamp X not found in signal feed"

**Causes:**
1. Feed doesn't cover your backtest date range
2. Bar frequency mismatch (e.g., feed is hourly, bars are 15min)
3. Timezone issues (feed is UTC, bars are in local time)

**Solutions:**
1. Re-export feed with correct `--start` and `--end` dates
2. Match `--bar_seconds` to your strategy's bar interval
3. Ensure strategy bars have UTC timezone: `df.index = df.index.tz_localize("UTC")`

### Performance: Slow Feed Loading

**If loading takes >5 seconds:**
1. Check feed size: `wc -l feeds/btc_signals.jsonl`
2. Consider splitting large feeds by date range
3. Use binary formats (future: Parquet, Arrow)

### Debugging: See What Signals Are Used

Enable verbose logging in strategy:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

Or add print statements:
```python
def generate_intent(df, ctx):
    # ... existing code ...
    print(f"[MoonWire] Bar {bar_ts}: probability={probability:.3f}, action={intent.action}")
    return intent
```

---

## Advanced Usage

### Core Gating Wrapper

**Use case:** Only allow Core strategy entries when MoonWire agrees.

```python
from research.strategies.sg_moonwire_intent_v1 import generate_intent as moonwire_intent
from research.strategies.your_core_strategy import generate_intent as core_intent

def generate_intent(df, ctx):
    core = core_intent(df, ctx)
    moonwire = moonwire_intent(df, ctx)
    
    # Gate Core entries with MoonWire
    if core.action == Action.ENTER_LONG:
        if moonwire.action != Action.ENTER_LONG:
            return Intent(
                action=Action.FLAT,
                reason="Core LONG rejected by MoonWire gate",
                meta={"core_action": core.action, "moonwire_action": moonwire.action},
            )
    
    return core
```

### Dynamic Threshold Optimization

**Use case:** Backtest different thresholds to find optimal values.

```python
import os

thresholds = [0.55, 0.60, 0.65, 0.70]

for thresh in thresholds:
    os.environ["MOONWIRE_LONG_THRESH"] = str(thresh)
    
    # Clear cache to reload config
    import research.strategies.sg_moonwire_intent_v1 as mw
    mw._CONFIG_CACHE = None
    
    # Run backtest
    results = run_backtest(...)
    print(f"Threshold {thresh}: Sharpe={results.sharpe:.2f}")
```

### Multi-Asset Feeds

**Use case:** Single feed file with multiple symbols.

```python
def generate_intent(df, ctx):
    # ... load signals ...
    
    # Filter by symbol from context
    symbol_signals = {
        ts: prob for ts, prob in signals.items()
        if signals_metadata[ts]["symbol"] == ctx.product
    }
    
    # ... rest of logic ...
```

---

## Performance Summary

**Standard Tier (270d lookback):**
- Win Rate: 54.74%
- Signals/Month: 15
- Max Drawdown: -13.42%
- Profit Factor: 1.41

**Elite Tier (365d lookback):**
- Win Rate: 59.54%
- Signals/Month: 11
- Max Drawdown: -32.36%
- Profit Factor: 1.44

**Recommendation:** Start with Standard tier for lower drawdown risk.

---

## File Structure

```
iteradynamics/
├── research/
│   └── strategies/
│       └── sg_moonwire_intent_v1.py   # Strategy implementation
├── tests/
│   └── test_sg_moonwire_intent.py      # Unit tests
├── docs/
│   └── MOONWIRE_INTEGRATION.md         # This file
└── feeds/  (optional, gitignored)
    ├── btc_signals.jsonl
    └── btc_signals.manifest.json
```

---

## Next Steps

1. **Generate Feeds:** Export signals from MoonWire for your target date range
2. **Run Tests:** Validate integration with `pytest tests/test_sg_moonwire_intent.py`
3. **Backtest:** Test with historical data using Standard tier thresholds
4. **Optimize:** Grid search thresholds if needed (0.55-0.70 for LONG)
5. **Production:** Deploy with validated config

---

## Support

**MoonWire Repo:** `/home/clawd/clawd/moonwire-backend`  
**MoonWire Docs:** `moonwire-backend/docs/VALIDATED_CONFIGS.md`  
**Integration Summary:** `/home/clawd/clawd/MOONWIRE_ITERA_INTEGRATION_SUMMARY.md`

**Common Issues:**
- Timestamp alignment → Use `validate_feed_alignment()` diagnostic
- Config errors → Check environment variables with `echo $MOONWIRE_*`
- Missing signals → Re-export feed with correct date range

---

**✅ Integration Ready - Happy Trading!**
