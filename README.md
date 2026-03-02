# Itera Dynamics

![Python](https://img.shields.io/badge/python-3.11%2B-blue?style=flat-square)
![Architecture](https://img.shields.io/badge/architecture-monorepo-orange?style=flat-square)
![Status](https://img.shields.io/badge/status-active-brightgreen?style=flat-square)
![License](https://img.shields.io/badge/license-MIT-lightgrey?style=flat-square)

> **Quantitative Trading Research & Execution Platform**

---

## Overview

**Itera Dynamics** is a quantitative trading platform built around a modular, asset-agnostic architecture. The system separates signal generation from execution, allowing the same core intelligence to power multiple market deployments.

### Current Focus: Argus (BTC & Multi-Asset)

The platform operates **Argus**, an hourly trading system that supports:

- **Single-product mode**: One asset (default **BTC-USD**) via `run_live.py`, with optional **ETH-USD** via `ARGUS_PRODUCT_ID`.
- **Portfolio mode**: Multi-asset (e.g. BTC-USD + ETH-USD) via `run_portfolio_live.py` and the research portfolio allocator.

Execution is against **Coinbase**. Future expansion to securities (stocks, ETFs) is architected in **Alpha Engine** but dormant.

---

## Architecture

### High-Level System Design

```
┌─────────────────────────────────────────────────────────────────────┐
│                       ITERA DYNAMICS                                │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│    ┌─────────────────────────────────────────┐                      │
│    │     APEX CORTEX (runtime/argus/apex_core/)                     │
│    │     Signal logic, governors, execution                         │
│    │  • signal_generator.py — single-asset (Layer 3)                │
│    │  • portfolio_signal_generator.py — multi-asset                 │
│    │  • exit_watcher.py — 5‑min early exit / min-hold               │
│    │  • strategy_router.py — external strategy loading               │
│    └──────────────┬──────────────────────────┘                      │
│                   │                                                 │
│         ┌─────────┴─────────┐                                       │
│         ▼                   ▼                                       │
│    ┌─────────┐        ┌─────────────┐                               │
│    │  ARGUS  │        │ AlphaEngine │                               │
│    │ BTC/ETH │        │ (Securities)│                               │
│    │ ACTIVE  │        │   DORMANT   │                               │
│    └─────────┘        └─────────────┘                               │
│                                                                     │
│    ┌─────────────────────────────────────────┐                      │
│    │            RESEARCH (repo root)         │                       │
│    │  Backtest engine, strategies, configs,  │                       │
│    │  portfolio allocator, experiments       │                       │
│    └─────────────────────────────────────────┘                       │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### Argus 3-Layer Strategy Architecture

The Argus runtime implements a clean 3-layer architecture for strategy execution:

```
┌─────────────────────────────────────────────────────────────────────┐
│                    LAYER 3: SG Execution                            │
│    (signal_generator.py — wallet, notional, DD governors)          │
│              ↑ Receives intent, applies safety gates, executes     │
│              │ Maps: ENTER_LONG→BUY, EXIT_LONG→SELL, HOLD→HOLD      │
└─────────────────────────────────────────────────────────────────────┘
                                    ↑
                           Strategy Intent (dict → StrategyIntent)
                                    │
┌─────────────────────────────────────────────────────────────────────┐
│                    LAYER 2: Strategies                              │
│                (pluggable, deterministic alpha modules)             │
│     generate_intent(df, ctx, *, closed_only=True) → dict            │
│              ↑ Uses regime classification for entry/exit            │
└─────────────────────────────────────────────────────────────────────┘
                                    ↑
                             RegimeState
                                    │
┌─────────────────────────────────────────────────────────────────────┐
│                    LAYER 1: Regime Engine                            │
│         (runtime/argus/research/regime/regime_engine.py)             │
│      classify_regime(df, closed_only=True) → RegimeState            │
└─────────────────────────────────────────────────────────────────────┘
```

**Key Principles:**

- **Layer 1** is authoritative — all strategies share the same regime classification.
- **Layer 2** strategies are deterministic — same input → same output, no side effects.
- **Layer 3** (SG) is the only component that touches broker, wallet, files, or state.
- **Timeline Safety**: `closed_only=True` ensures decisions use closed candles only.

---

## Project Structure

```
IteraDynamics_Mono/
│
├── runtime/                          # 🦅 LIVE EXECUTION
│   └── argus/                        # BTC/ETH trading service
│       ├── apex_core/                # Signal & execution
│       │   ├── signal_generator.py   # Layer 3: single-asset execution + governors
│       │   ├── portfolio_signal_generator.py  # Multi-asset portfolio execution
│       │   ├── strategy_router.py    # Load strategy from ARGUS_STRATEGY_MODULE/FUNC
│       │   ├── strategy_intent.py    # Intent/Action types (optional path)
│       │   └── exit_watcher.py       # 5‑min early exit & min-hold checks
│       │
│       ├── research/                 # 3-LAYER STRATEGY ARCHITECTURE
│       │   ├── regime/               # LAYER 1: Regime Engine
│       │   │   ├── __init__.py
│       │   │   └── regime_engine.py
│       │   │
│       │   ├── strategies/           # LAYER 2: Strategy Modules
│       │   │   ├── sg_regime_trend_v1.py
│       │   │   ├── sg_core_exposure_v1.py
│       │   │   ├── sg_core_exposure_v2.py
│       │   │   ├── sg_trend_probe_v1.py
│       │   │   ├── sg_vol_probe_v1.py
│       │   │   ├── sg_compression_shot_v1.py
│       │   │   ├── sg_compression_shot_v2.py
│       │   │   ├── sg_compression_shot_v3.py
│       │   │   └── sg_stub_strategy.py
│       │   │
│       │   ├── harness/              # Smoke tests & backtest runner
│       │   │   ├── regime_smoke.py
│       │   │   ├── strategy_smoke.py
│       │   │   ├── backtest_smoke.py
│       │   │   └── backtest_runner.py
│       │   │
│       │   ├── diagnostics/
│       │   │   └── crash_window_report.py
│       │   └── results/             # Strategy run artifacts
│       │
│       ├── src/
│       │   └── real_broker.py        # Coinbase API integration
│       │
│       ├── models/                   # Trained ML models & params
│       │   ├── random_forest.pkl
│       │   ├── governance_params.json
│       │   ├── paper_trading_params.json
│       │   └── ml_hyperparameters.json
│       │
│       ├── config.py                 # Product ID, namespaced paths (ARGUS_PRODUCT_ID)
│       ├── run_live.py               # Hourly scheduler (single product)
│       ├── run_portfolio_live.py     # Portfolio orchestrator (multi-asset)
│       ├── sweep_multi_product.py    # Multi-product config/path checks
│       ├── test_execution.py        # Pre-deployment test suite
│       ├── dashboard.py             # Streamlit dashboard (run from here)
│       ├── cortex.json / cortex_<slug>.json
│       ├── prime_state.json / paper_prime_state.json (or _<slug>)
│       ├── trade_state.json / trade_ledger.jsonl (or _<slug>)
│       ├── flight_recorder*.csv
│       ├── portfolio_state.json     # Portfolio mode state
│       ├── MULTI_PRODUCT.md         # Multi-product (BTC/ETH) guide
│       ├── argus-eth.env.example
│       └── argus-eth.service.example
│
├── research/                         # 🔬 STRATEGY R&D (repo root)
│   ├── configs/                      # Strategy/env configs
│   │   └── core_v2/                  # e.g. btc_core_v2_tuned_*.env
│   ├── portfolio/                    # Cross-asset allocator (portfolio mode)
│   │   └── cross_asset_allocator.py
│   ├── regime/                       # Repo-root regime (portfolio mode)
│   │   └── classify_regime.py
│   ├── strategies/                   # Backtest-focused strategies
│   ├── engine/                       # Backtest engine
│   ├── backtest/                     # Backtest utilities
│   ├── experiments/                  # One-off experiments
│   ├── backtests/                    # Results & artifacts
│   └── STRATEGY_SUMMARY.md
│
├── alpha_engine/                     # 📈 Securities platform (Dormant)
├── moonwire/                         # 🌙 Alternative execution engine
├── scripts/                          # 🛠️ Data, training, debug
├── data/                             # 📊 Historical data
├── output/                           # 📁 Backtest results
│
├── pyproject.toml
├── requirements.txt
└── README.md
```

---

## Argus Strategy System

### Layer 1: Regime Engine

The regime engine provides authoritative market state classification shared by all strategies.

**Location:** `runtime/argus/research/regime/regime_engine.py`

**API:**

```python
from research.regime import classify_regime, RegimeState, RegimeLabel

regime = classify_regime(df, closed_only=True)
# regime.label: TREND_UP | TREND_DOWN | CHOP | VOL_EXPANSION | VOL_COMPRESSION | PANIC
# regime.confidence: 0.0 - 1.0
# regime.features: {price, ema_fast, ema_slow, atr, atr_pct, ...}
# regime.meta: {closed_only, dropped_last_row, ...}
```

**Regime Labels (Priority Order):**

| Priority | Label            | Description                          |
|----------|------------------|--------------------------------------|
| 1        | PANIC            | Extreme volatility + high volume    |
| 2        | VOL_EXPANSION    | ATR% above threshold                 |
| 3        | VOL_COMPRESSION  | ATR% below threshold                 |
| 4        | TREND_UP         | EMA fast > slow + trend strength     |
| 5        | TREND_DOWN       | EMA fast < slow + trend strength     |
| 6        | CHOP             | Default / no clear signal            |

### Layer 2: Strategy Modules

Strategies implement a single entrypoint and return a **dict** that the signal generator converts to `StrategyIntent`:

```python
def generate_intent(df: pd.DataFrame, ctx: dict, *, closed_only: bool = True) -> dict:
    """
    Returns:
        {
            "action": "ENTER_LONG" | "EXIT_LONG" | "HOLD" | "FLAT",
            "confidence": float,           # 0-1
            "desired_exposure_frac": float,# 0-1
            "horizon_hours": int,
            "reason": str,
            "meta": dict
        }
    """
```

**Available Strategies:**

| Strategy              | File                    | Description                              |
|-----------------------|-------------------------|------------------------------------------|
| **Regime Trend v1**   | `sg_regime_trend_v1.py` | Trend-following with ADX gate            |
| **Core Exposure v1** | `sg_core_exposure_v1.py`| Volatility-scaled exposure by regime     |
| **Core Exposure v2**  | `sg_core_exposure_v2.py`| Core v1 + macro filter / tuning          |
| **Trend Probe v1**    | `sg_trend_probe_v1.py` | TREND_UP only, strict trend following    |
| **Vol Probe v1**      | `sg_vol_probe_v1.py`   | VOL_COMPRESSION breakout                 |
| **Compression Shot v1–v3** | `sg_compression_shot_v*.py` | Vol compression breakout variants |
| **Stub**              | `sg_stub_strategy.py`  | Testing stub (always HOLD)               |

### Layer 3: Signal Generator (Execution)

The signal generator (`apex_core/signal_generator.py`) is the execution layer that:

- Loads the external strategy via `ARGUS_STRATEGY_MODULE` / `ARGUS_STRATEGY_FUNC`
- Applies safety gates (wallet, notional, drawdown governors, min-hold, reentry cooldown)
- Maps intents to trades: `ENTER_LONG→BUY`, `EXIT_LONG→SELL`
- Manages position state and horizon exits
- Writes to `cortex.json` (or namespaced `cortex_<slug>.json`) and state files

**External strategy (single-asset):**

```bash
ARGUS_STRATEGY_MODULE="research.strategies.sg_core_exposure_v1"
ARGUS_STRATEGY_FUNC="generate_intent"
```

**Exit watcher** (`apex_core/exit_watcher.py`): Optional 5‑minute check for early exit (profit hurdle) and min-hold enforcement. See env vars `ARGUS_EARLY_EXIT_ENABLED`, `ARGUS_EARLY_EXIT_MIN_HOURS`, `ARGUS_EARLY_EXIT_PROFIT_PCT`.

---

## Multi-Product (BTC-USD / ETH-USD)

Argus can run **two independent instances** (e.g. BTC-USD and ETH-USD) with no shared state. Each instance uses **namespaced files** so they do not overwrite each other.

| Variable           | Default   | Description                                      |
|--------------------|-----------|--------------------------------------------------|
| `ARGUS_PRODUCT_ID` | `BTC-USD` | Coinbase product (e.g. `BTC-USD`, `ETH-USD`).   |

When `ARGUS_PRODUCT_ID` is not set or is `BTC-USD`, legacy filenames are used. For other products (e.g. `ETH-USD`), files are namespaced: `flight_recorder_eth_usd.csv`, `prime_state_eth_usd.json`, `trade_ledger_eth_usd.jsonl`, `cortex_eth_usd.json`, etc.

See **`runtime/argus/MULTI_PRODUCT.md`** for full details, and use **`sweep_multi_product.py`** (no broker) or **`test_execution.py`** (with API keys) to verify config and paths.

---

## Portfolio Mode (Multi-Asset)

For a single process trading multiple products with a shared allocation policy:

- **Entrypoint:** `runtime/argus/run_portfolio_live.py`
- **Allocator:** `research/portfolio/cross_asset_allocator.py` (deterministic, closed-bar only)
- **Env:** `ARGUS_PORTFOLIO_PRODUCTS=BTC-USD,ETH-USD` (required). Optional: `PORTFOLIO_MAX_GROSS_EXPOSURE`, `PORTFOLIO_MAX_WEIGHT_PER_ASSET`, `PORTFOLIO_MIN_WEIGHT_PER_ASSET`, `PORTFOLIO_ALLOW_CASH`.

State is stored in `portfolio_state.json`; per-product state/ledgers remain namespaced as in multi-product.

---

## Timeline Safety

**All decisions are based on CLOSED candles only.**

When `closed_only=True` (default):

1. The **last row is dropped** before computing indicators.
2. `meta.closed_only` and `meta.dropped_last_row` are set.
3. `asof_ts` is derived from the last **included** (closed) row.

This prevents lookahead bias in backtesting and ensures live decisions do not rely on the current forming candle.

---

## Quick Start

### 1. Installation

```bash
git clone https://github.com/IteraDynamics/IteraDynamics_Mono.git
cd IteraDynamics_Mono
pip install -e .
pip install -r requirements.txt
```

### 2. Run Smoke Tests (Safe, No Broker)

From repo root:

```powershell
cd "C:\Users\admin\OneDrive\Desktop\Desktop\IteraDynamics_Mono"

python -c "
import sys
sys.path.insert(0, r'./runtime/argus')
from research.harness.regime_smoke import main as regime_test
from research.harness.strategy_smoke import main as strategy_test
regime_test()
strategy_test()
"
```

### 3. Run Live Trading (Argus) — Dry Run

Configure Coinbase in `runtime/argus/.env` (or env):

```env
COINBASE_API_KEY=your_key
COINBASE_API_SECRET=your_secret
COINBASE_PORTFOLIO_UUID=your_portfolio_uuid
```

Single-asset, **dry-run** (paper):

```powershell
cd runtime/argus

$env:PRIME_DRY_RUN = "1"
$env:ARGUS_MODE = "prime"
$env:ARGUS_STRATEGY_MODULE = "research.strategies.sg_core_exposure_v1"
$env:ARGUS_STRATEGY_FUNC = "generate_intent"

python -c "from apex_core.signal_generator import generate_signals; generate_signals()"
```

Optional: set `ARGUS_PRODUCT_ID=ETH-USD` to run the ETH instance with namespaced files.

### 4. Launch Dashboard

```bash
cd runtime/argus
python -m streamlit run dashboard.py
```

For a second product (e.g. ETH), use a different port: `streamlit run dashboard.py --server.port 8502` with `ARGUS_PRODUCT_ID=ETH-USD` set.

---

## Environment Variables

### Safety / Mode

| Variable        | Default   | Description                    |
|-----------------|-----------|--------------------------------|
| `PRIME_DRY_RUN` | `0`       | Set to `1` for paper trading   |
| `ARGUS_DRY_RUN` | `0`       | Alternate dry-run flag         |
| `ARGUS_MODE`    | `legacy`  | Set to `prime` for Prime engine |

### Product & Paths

| Variable           | Default   | Description                          |
|--------------------|-----------|--------------------------------------|
| `ARGUS_PRODUCT_ID` | `BTC-USD` | Product and file namespacing         |

### External Strategy (Single-Asset)

| Variable               | Example                                  | Description   |
|------------------------|------------------------------------------|---------------|
| `ARGUS_STRATEGY_MODULE`| `research.strategies.sg_core_exposure_v1` | Module path   |
| `ARGUS_STRATEGY_FUNC`  | `generate_intent`                        | Function name |

### Prime Governors

| Variable                  | Default | Description                          |
|---------------------------|---------|--------------------------------------|
| `PRIME_CONF_MIN`          | `0.64`  | Min confidence to act                |
| `PRIME_HORIZON`            | `48`    | Default horizon hours                |
| `PRIME_MAX_EXPOSURE`       | `0.25`  | Max exposure fraction                |
| `PRIME_EXIT_MIN_HOLD_H`    | `0`     | Min hours before non-panic exit      |
| `PRIME_REENTRY_COOLDOWN_H` | `0`     | Hours before re-entry after exit     |
| `PRIME_MIN_NOTIONAL_USD`   | `5`     | Min notional per trade               |
| Drawdown bands            | —       | `PRIME_DD_SOFT`, `PRIME_DD_HARD`, `PRIME_DD_KILL`, `PRIME_DD_SOFT_MULT` |

### Regime Engine

| Variable             | Default | Description                |
|----------------------|---------|----------------------------|
| `REGIME_EMA_FAST`    | `20`    | Fast EMA period            |
| `REGIME_EMA_SLOW`    | `50`    | Slow EMA period            |
| `REGIME_ATR_LEN`     | `14`    | ATR period                 |
| `REGIME_TREND_THRESH`| `0.25`  | Trend strength threshold   |
| `REGIME_VOL_LO`      | `0.003` | Vol compression threshold |
| `REGIME_VOL_HI`      | `0.025` | Vol expansion threshold    |
| `REGIME_PANIC_HI`    | `0.040` | Panic threshold           |

### Portfolio Mode

| Variable                      | Default | Description              |
|------------------------------|---------|--------------------------|
| `ARGUS_PORTFOLIO_PRODUCTS`    | —       | Required, e.g. `BTC-USD,ETH-USD` |
| `PORTFOLIO_MAX_GROSS_EXPOSURE`| `1.0`   | Max gross exposure       |
| `PORTFOLIO_MAX_WEIGHT_PER_ASSET` | `0.85` | Max weight per asset |
| `PORTFOLIO_MIN_WEIGHT_PER_ASSET` | `0.0`  | Min weight per asset |
| `PORTFOLIO_ALLOW_CASH`       | `True`  | Allow cash in allocation |

### Exit Watcher (5‑min)

| Variable                     | Default | Description                |
|-----------------------------|---------|----------------------------|
| `ARGUS_EARLY_EXIT_ENABLED`  | `1`     | Enable early-exit checks   |
| `ARGUS_EARLY_EXIT_MIN_HOURS`| `0.5`   | Min hold before early exit |
| `ARGUS_EARLY_EXIT_PROFIT_PCT` | `0.01` | Profit % hurdle for early exit |

Strategy-specific variables are documented in each strategy’s docstring and in **research/configs/** (e.g. `core_v2`).

---

## cortex.json Output

The signal generator writes decision state to `cortex.json` (or `cortex_<slug>.json`):

```json
{
  "timestamp_utc": "2026-02-17 14:00:00",
  "mode": "prime",
  "intent_action": "ENTER_LONG",
  "intent_reason": "trend_up_vol_scaled_exposure(bucket=moderate)",
  "intent_source": "external",
  "external_strategy_module": "research.strategies.sg_core_exposure_v1",
  "external_strategy_func": "generate_intent",
  "p_long": 0.72,
  "conf_min": 0.64,
  "equity_usd": 1000.00,
  "drawdown_frac": -0.02,
  "dd_band": "ok",
  "dry_run": true,
  "wallet_verified": true
}
```

---

## Development

### Adding New Strategies

1. Create a module under `runtime/argus/research/strategies/` (e.g. `sg_my_strategy_v1.py`).
2. Implement `generate_intent(df, ctx, *, closed_only=True) -> dict` with keys: `action`, `confidence`, `desired_exposure_frac`, `horizon_hours`, `reason`, `meta`.
3. Call Layer 1: `from research.regime import classify_regime` (with `sys.path` including `runtime/argus`).
4. Run regime and strategy smoke tests before live use.

### Running Tests

```powershell
# From repo root, with runtime/argus on path
python -c "import sys; sys.path.insert(0, r'./runtime/argus'); from research.harness.regime_smoke import main; main()"
python -c "import sys; sys.path.insert(0, r'./runtime/argus'); from research.harness.strategy_smoke import main; main()"

# Multi-product path/config check (no broker)
cd runtime/argus && python sweep_multi_product.py

# Full execution test (needs API keys)
cd runtime/argus && python test_execution.py
```

### Strategy Configs

Use **research/configs/** (e.g. **core_v2/btc_core_v2_tuned_*.env**) to hold strategy and regime env vars for backtests and live; source or copy into `.env` as needed.

---

## Key Features

### Signal Generation

- **Regime detection**: Volatility + trend-based market state (Layer 1).
- **Pluggable strategies**: Hot-swap via `ARGUS_STRATEGY_MODULE` / `ARGUS_STRATEGY_FUNC` (Layer 2).
- **ML ensemble**: Optional `p_long` from Random Forest in Prime mode.

### Risk Management

- **Drawdown governors**: Soft/Hard/Kill bands and optional soft mult.
- **Execution gates**: Wallet verification, min notional, reentry cooldown, min-hold before exit.
- **Timeline safety**: Closed-candle-only decisions.
- **Panic exit**: Automatic exit on PANIC regime; early-exit via exit watcher when enabled.

### Research & Ops

- **Smoke tests**: Regime and strategy validation without broker.
- **Backtest runner**: Same strategy interface for backtests (see `research/harness/backtest_runner.py`).
- **Multi-product**: BTC/ETH (or other products) via `ARGUS_PRODUCT_ID` and namespaced files.
- **Portfolio mode**: Multi-asset allocation via `run_portfolio_live.py` and research portfolio allocator.

---

## License

MIT License — see `LICENSE` for details.

> **Disclaimer**: This software is for educational and research purposes only. Trading involves substantial risk of loss. Past performance does not guarantee future results.

---

## Acknowledgments

Built with Python, pandas, pandas-ta, scikit-learn, Streamlit, and Coinbase Advanced Trade (`coinbase-advanced-py`).
