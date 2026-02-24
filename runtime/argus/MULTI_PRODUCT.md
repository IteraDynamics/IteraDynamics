# Argus multi-product (BTC-USD / ETH-USD)

Argus can run **two independent instances** (e.g. BTC-USD and ETH-USD) at the same time with no shared state. Each instance is configured via environment variables and uses **namespaced files** so they do not overwrite each other.

## Env var

| Variable | Default | Description |
|----------|---------|-------------|
| `ARGUS_PRODUCT_ID` | `BTC-USD` | Coinbase product (e.g. `BTC-USD`, `ETH-USD`). Drives API URLs and file namespacing. |

## Namespaced files

When `ARGUS_PRODUCT_ID` is **not** set or is `BTC-USD`, legacy filenames are used (unchanged behavior):

- `flight_recorder.csv`, `prime_state.json`, `paper_prime_state.json`, `trade_state.json`, `trade_ledger.jsonl`, `cortex.json`, `argus.log`

When `ARGUS_PRODUCT_ID` is set to another product (e.g. `ETH-USD`), filenames are namespaced with a slug (hyphens replaced by underscores for safe filenames):

- `flight_recorder_eth_usd.csv`
- `prime_state_eth_usd.json`, `paper_prime_state_eth_usd.json`
- `trade_state_eth_usd.json`, `trade_ledger_eth_usd.jsonl`
- `cortex_eth_usd.json`, `argus_eth_usd.log`

So BTC and ETH instances can run in the **same directory** without state collisions.

## Running the ETH instance

1. **Env:** set `ARGUS_PRODUCT_ID=ETH-USD` (e.g. in `/etc/argus/argus-eth.env` or `.env`).
2. **Scheduler:** run the same commands as BTC, with that env loaded:
   - `python run_live.py` (with env pointing to ETH env file).
3. **Dashboard (optional):** if you run Streamlit for ETH, use a different port so it doesn’t conflict with BTC (default 8501):
   - `streamlit run dashboard.py --server.port 8502`
   - Use the same env (`ARGUS_PRODUCT_ID=ETH-USD`) so the dashboard shows ETH state and data.

## Example files

- **`argus-eth.env.example`** — env template for ETH (copy and fill secrets).
- **`argus-eth.service.example`** — systemd unit for the ETH instance (do not overwrite existing BTC service).

## Startup diagnostics

On startup, the live scheduler prints product and paths, e.g.:

```
[config] ARGUS_PRODUCT_ID=ETH-USD (slug=eth_usd)
[config] flight_recorder=.../flight_recorder_eth_usd.csv
[config] prime_state(live)=.../prime_state_eth_usd.json paper=.../paper_prime_state_eth_usd.json
[config] trade_state=.../trade_state_eth_usd.json ledger=.../trade_ledger_eth_usd.jsonl
[config] cortex=.../cortex_eth_usd.json log=.../argus_eth_usd.log
```

Default (BTC-USD) behavior is unchanged: no env set ⇒ legacy filenames and BTC-USD product.

---

## Full functionality sweep

Use these tests and checks to confirm multi-product behavior and that files are written/modified as intended.

### 1. Multi-product config sweep (no broker/API required)

From `runtime/argus`:

```bash
python sweep_multi_product.py
```

This verifies:

- **Config default:** With `ARGUS_PRODUCT_ID` unset, paths use legacy names (e.g. `flight_recorder.csv`, `argus.log`) and product is BTC-USD.
- **Config ETH:** With `ARGUS_PRODUCT_ID=ETH-USD`, paths use namespaced names (e.g. `flight_recorder_eth_usd.csv`, `argus_eth_usd.log`).
- **Path consistency:** `signal_generator` and `real_broker` use the same paths as `config` (skipped if broker import fails, e.g. no API keys).
- **No hardcoded BTC-USD in URLs:** Candles and spot price URLs use `PRODUCT_ID` (no literal `BTC-USD` in API URLs in `signal_generator.py`, `exit_watcher.py`, `dashboard.py`).

Optional: `python sweep_multi_product.py --write` also checks that a write with ETH-USD targets the namespaced file (creates `flight_recorder_eth_usd.csv` in `runtime/argus`).

### 2. Pre-deployment test suite (needs API keys and broker)

From `runtime/argus` (with `.env` or credentials set):

```bash
python test_execution.py
```

Includes:

- **State File Paths:** Resolved paths match config (legacy or namespaced depending on `ARGUS_PRODUCT_ID`).
- **Multi-Product Path Namespacing:** Confirms config uses legacy names for BTC-USD and namespaced names for other products.

To test with ETH paths, run with env set:

```bash
# Linux/macOS
ARGUS_PRODUCT_ID=ETH-USD python test_execution.py

# Windows PowerShell
$env:ARGUS_PRODUCT_ID="ETH-USD"; python test_execution.py
```

### 3. Manual checks (optional)

| Check | How |
|-------|-----|
| Default run uses legacy files | Run `python run_live.py` with no `ARGUS_PRODUCT_ID`; confirm startup log shows `flight_recorder.csv`, `prime_state.json`, etc. |
| ETH run uses namespaced files | Set `ARGUS_PRODUCT_ID=ETH-USD`, run `python run_live.py`; confirm log shows `flight_recorder_eth_usd.csv`, `argus_eth_usd.log`, etc. |
| No cross-write | Run BTC instance, then start ETH instance in same dir; confirm only `*_eth_usd.*` and legacy files exist and ETH did not overwrite legacy state. |
| Dashboard shows product | Open dashboard with ETH env; chart and labels should show ETH-USD and read from namespaced CSV/state. |

### 4. Quick reference

- **Sweep script:** `runtime/argus/sweep_multi_product.py` — config and path checks only.
- **Full tests:** `runtime/argus/test_execution.py` — broker, market data, signals, state paths, multi-product paths.
- **Path consistency** may be skipped in the sweep if `signal_generator`/`real_broker` fail to import (e.g. missing API keys); other sweep checks do not require credentials.
