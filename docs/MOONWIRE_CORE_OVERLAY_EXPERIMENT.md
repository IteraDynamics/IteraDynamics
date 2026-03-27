# MoonWire → Core integration experiment

Minimal, testable, reversible overlay: MoonWire modifies Core BTC sleeve **desired exposure** after Layer 2 intent is generated. Core entry/exit logic is unchanged. No new strategies or sleeves.

## Control flow

1. **Layer 1** (Regime Engine) and **Layer 2** (Core sleeve) run as today → Core produces `desired_exposure_frac`.
2. **Overlay (governance)**  
   If `MOONWIRE_OVERLAY_ENABLED=1`, a timestamp-aligned MoonWire state (bullish/neutral/bearish) is looked up and a **multiplier** is applied:  
   `final_desired_exposure_frac = core_desired_exposure_frac * multiplier`
3. **Downstream** (backtest harness or allocator) uses `final_desired_exposure_frac`; Core logic is untouched.

**Backtest:** Overlay is applied inside the harness after each `generate_intent()` call; trace CSV records `core_desired_exposure_frac`, `moonwire_state`, `moonwire_multiplier`, `desired_exposure_frac` (final).

**Live:** Overlay is applied in `run_portfolio_live` after building Layer 2 intents and before `generate_portfolio_decision()`; the allocator receives already-modified intents.

## Variant mappings

| State   | Variant A | Variant B | Variant C |
|--------|-----------|-----------|-----------|
| bullish | 1.00x     | 1.00x     | 1.00x     |
| neutral | 0.50x     | 0.75x     | 1.00x     |
| bearish | 0.00x     | 0.25x     | 0.00x     |

State is derived from MoonWire probability feed:  
`prob >= MOONWIRE_OVERLAY_BULL_THRESH` → bullish;  
`prob <= MOONWIRE_OVERLAY_BEAR_THRESH` → bearish;  
else neutral.

## Env vars

### Required when overlay is **enabled**

| Env var | Meaning |
|---------|--------|
| `MOONWIRE_OVERLAY_ENABLED` | `1` = apply overlay; unset or `0` = Core only (unchanged). |
| `MOONWIRE_SIGNAL_FILE` | Path to JSONL feed: one object per line with `timestamp` (Unix sec) and `probability` (0–1). MoonWire owns signal generation; Itera only consumes. To ensure a feed exists: `python scripts/ensure_moonwire_signal_feed.py` (checks/validates/calls moonwire-backend via subprocess). See `docs/MOONWIRE_INTEGRATION.md`. |

### Optional (overlay)

| Env var | Default | Meaning |
|---------|--------|
| `MOONWIRE_OVERLAY_VARIANT` | `A` | `A`, `B`, or `C` (multiplier map above). |
| `MOONWIRE_OVERLAY_BULL_THRESH` | `0.55` | prob ≥ this → bullish. |
| `MOONWIRE_OVERLAY_BEAR_THRESH` | `0.45` | prob ≤ this → bearish. |
| `MOONWIRE_OVERLAY_STRICT_TS` | `0` | `1` = raise if bar timestamp missing in feed; `0` = treat as neutral. |
| `MOONWIRE_OVERLAY_PRODUCT_ID` | `BTC-USD` | Comma-separated product IDs to apply overlay (live only). |

### Unchanged (Core / backtest)

Same as today: `ARGUS_STRATEGY_MODULE`, `ARGUS_STRATEGY_FUNC`, `ARGUS_DATA_FILE`, `ARGUS_LOOKBACK`, `ARGUS_INITIAL_EQUITY`, `ARGUS_FEE_BPS`, `ARGUS_SLIPPAGE_BPS`, `ARGUS_ENV_FILE`, etc.

## Backtest commands

Run from **repo root**. Data: `ARGUS_DATA_FILE` (default `runtime/argus/flight_recorder.csv`). Strategy: Core v2 (or v1) via `ARGUS_STRATEGY_MODULE` / `ARGUS_STRATEGY_FUNC`; optional `ARGUS_ENV_FILE` for Core/regime params.

### Baseline (no MoonWire overlay)

```powershell
# PowerShell
$env:ARGUS_STRATEGY_MODULE = "research.strategies.sg_core_exposure_v2"
$env:ARGUS_STRATEGY_FUNC = "generate_intent"
$env:ARGUS_ENV_FILE = "research/configs/core_v2/btc_core_v2_tuned_2026_02_27.env"
# Leave MOONWIRE_OVERLAY_ENABLED unset or 0
python -c "import sys; sys.path.insert(0, r'./runtime/argus'); from research.harness.backtest_runner import main; main()"
```

```bash
# Bash
export ARGUS_STRATEGY_MODULE=research.strategies.sg_core_exposure_v2
export ARGUS_STRATEGY_FUNC=generate_intent
export ARGUS_ENV_FILE=research/configs/core_v2/btc_core_v2_tuned_2026_02_27.env
python -c "import sys; sys.path.insert(0, r'./runtime/argus'); from research.harness.backtest_runner import main; main()"
```

### MoonWire variant A

```powershell
$env:MOONWIRE_OVERLAY_ENABLED = "1"
$env:MOONWIRE_SIGNAL_FILE = "C:\Users\admin\OneDrive\Desktop\Desktop\moonwire-backend\feeds\btc_signals.jsonl"
$env:MOONWIRE_OVERLAY_VARIANT = "A"
$env:ARGUS_STRATEGY_MODULE = "research.strategies.sg_core_exposure_v2"
$env:ARGUS_STRATEGY_FUNC = "generate_intent"
$env:ARGUS_ENV_FILE = "research/configs/core_v2/btc_core_v2_tuned_2026_02_27.env"
python -c "import sys; sys.path.insert(0, r'./runtime/argus'); from research.harness.backtest_runner import main; main()"
```

### MoonWire variant B

Same as variant A, set:

```powershell
$env:MOONWIRE_OVERLAY_VARIANT = "B"
```

### MoonWire variant C

Same as variant A, set:

```powershell
$env:MOONWIRE_OVERLAY_VARIANT = "C"
```

Trace output (when overlay enabled): `debug/harness_btc_trace.csv` includes `core_desired_exposure_frac`, `moonwire_state`, `moonwire_multiplier`, `desired_exposure_frac` (final).

## Code locations

- **Overlay logic (canonical):** `research/portfolio/moonwire_overlay.py` (used by run_portfolio_live when repo root is on path).
- **Overlay logic (backtest):** `runtime/argus/research/governance/moonwire_overlay.py` (mirror used by backtest harness so `research` resolves to argus).
- **Backtest integration:** `runtime/argus/research/harness/backtest_runner.py` (applies overlay after `generate_intent`, adds trace columns).
- **Live integration:** `runtime/argus/run_portfolio_live.py` (applies overlay to Layer 2 intents before `generate_portfolio_decision`).

## Assumptions / placeholders

- **Feed format:** JSONL with `timestamp` (Unix seconds) and `probability` (float 0–1). No embedded ML; input is a pre-generated file or column export.
- **Timestamp alignment:** Bar timestamps (backtest or live) are converted to Unix seconds for feed lookup. Missing bars: if `MOONWIRE_OVERLAY_STRICT_TS=0`, treated as **neutral** (variant neutral multiplier).
- **Product scope:** Backtest is single-asset (BTC); overlay applies to that sleeve. Live: overlay applies only to product IDs listed in `MOONWIRE_OVERLAY_PRODUCT_ID` (default `BTC-USD`).
- **Reversibility:** Set `MOONWIRE_OVERLAY_ENABLED=0` or unset to restore pure Core behavior with no code change.
