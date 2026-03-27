# Mean Reversion Extreme Strategy (v1)

**Status:** Layer 2 sleeve candidate  
**Module:** `runtime.argus.research.strategies.sg_mean_reversion_extreme_v1`  
**Last Updated:** 2026-03-06

---

## Strategy hypothesis

- **Long-only** mean reversion after **extreme downside** (RSI oversold).
- Intended as a **defensive / crash-alpha** sleeve, not a BTC replacement.
- Entries occur when price has sold off sharply (RSI ≤ 25); exits on RSI recovery or after a fixed max hold to limit drawdown and reversion risk.

---

## Baseline rules

| Rule | Value |
|------|--------|
| **Entry** | RSI(21) ≤ 25 (oversold) |
| **Exit** | (a) RSI recovers above 50, **or** (b) holding period reaches 48 bars (48 hours) |
| **Otherwise** | Hold existing state / remain flat |
| **Exposure cap** | 0.985 (research default) |

- **Closed-bar only:** decisions use only data through the current bar; no lookahead.
- **Deterministic:** same OHLCV + env → same intent sequence.

---

## Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `SG_MR_RSI_LEN` | `21` | RSI period (bars) |
| `SG_MR_OVERSOLD` | `25` | RSI ≤ this → eligible for long entry |
| `SG_MR_EXIT_RSI` | `50` | RSI > this → exit (recovery) |
| `SG_MR_MAX_HOLD_BARS` | `48` | Max bars in position (e.g. 48 hours) |
| `SG_MR_MAX_EXPOSURE` | `0.985` | Cap on desired long exposure |

---

## Harness backtest (PowerShell)

Run from **repo root** (`IteraDynamics_Mono`). The runner resolves `ARGUS_DATA_FILE` relative to the current directory; if unset, it uses `runtime\argus\flight_recorder.csv` under the repo.

```powershell
# From repo root
cd C:\Users\admin\OneDrive\Desktop\Desktop\IteraDynamics_Mono

$env:ARGUS_STRATEGY_MODULE = "research.strategies.sg_mean_reversion_extreme_v1"
$env:ARGUS_STRATEGY_FUNC   = "generate_intent"
# Default data: runtime\argus\flight_recorder.csv (create or point to your OHLCV CSV)
$env:ARGUS_DATA_FILE       = ".\runtime\argus\flight_recorder.csv"
# Or use canonical BTC hourly data if present (same as portfolio_geometry):
# $env:ARGUS_DATA_FILE     = ".\data\btcusd_3600s_2019-01-01_to_2025-12-30.csv"

python -c "import sys; sys.path.insert(0, r'./runtime/argus'); from research.harness.backtest_runner import main; main()"
```

To run the runner script directly from the argus directory:

```powershell
cd C:\Users\admin\OneDrive\Desktop\Desktop\IteraDynamics_Mono\runtime\argus

$env:ARGUS_STRATEGY_MODULE = "research.strategies.sg_mean_reversion_extreme_v1"
$env:ARGUS_STRATEGY_FUNC   = "generate_intent"
$env:ARGUS_DATA_FILE       = ".\flight_recorder.csv"

python -c "from research.harness.backtest_runner import main; main()"
```

CSV must have columns: `Timestamp`, `Open`, `High`, `Low`, `Close`, and optionally `Volume`.

---

## Contract

- **API:** `generate_intent(df, ctx, closed_only=True, **kwargs)` → `dict`
- **Dict keys:** `action`, `confidence`, `desired_exposure_frac`, `horizon_hours`, `reason`, `meta`
- **Actions:** `ENTER_LONG` | `EXIT` | `HOLD`

---

## Tests

From repo root:

```powershell
cd C:\Users\admin\OneDrive\Desktop\Desktop\IteraDynamics_Mono
pytest tests\test_sg_mean_reversion_extreme_v1.py -v
```

Covers: oversold entry, exit on max hold, exit on RSI recovery, determinism, no-lookahead (prefix-only dependency), and intent dict shape.
