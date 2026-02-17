# Itera Dynamics

![Python](https://img.shields.io/badge/python-3.11%2B-blue?style=flat-square)
![Architecture](https://img.shields.io/badge/architecture-monorepo-orange?style=flat-square)
![Status](https://img.shields.io/badge/status-active-brightgreen?style=flat-square)
![License](https://img.shields.io/badge/license-MIT-lightgrey?style=flat-square)

> **Quantitative Trading Research & Execution Platform**

---

## Overview

**Itera Dynamics** is a quantitative trading platform built around a modular, asset-agnostic architecture. The system separates signal generation from execution, allowing the same core intelligence to power multiple market deployments.

### Current Focus: BTC Trading via Argus

The platform currently operates **Argus**, an hourly BTC trading system running against Coinbase. Future expansion to securities (stocks, ETFs) is architected but dormant.

---

## Architecture

### High-Level System Design

```
┌─────────────────────────────────────────────────────────────────────┐
│                       ITERA DYNAMICS                                │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│    ┌─────────────────────────────────────────┐                      │
│    │       APEX CORTEX (apex_core/)          │                      │
│    │       The Brain - Signal Logic          │                      │
│    │  • ML inference & backtesting           │                      │
│    │  • Regime detection                     │                      │
│    │  • Governance & drift monitoring        │                      │
│    └──────────────┬──────────────────────────┘                      │
│                   │                                                 │
│         ┌─────────┴─────────┐                                       │
│         ▼                   ▼                                       │
│    ┌─────────┐        ┌─────────────┐                               │
│    │  ARGUS  │        │ AlphaEngine │                               │
│    │  (BTC)  │        │ (Securities)│                               │
│    │ ACTIVE  │        │   DORMANT   │                               │
│    └─────────┘        └─────────────┘                               │
│                                                                     │
│    ┌─────────────────────────────────────────┐                      │
│    │            RESEARCH LAB                 │                      │
│    │    Strategy development & backtesting   │                      │
│    └─────────────────────────────────────────┘                      │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### Argus 3-Layer Strategy Architecture

The Argus runtime implements a clean 3-layer architecture for strategy execution:

```
┌─────────────────────────────────────────────────────────────────────┐
│                    LAYER 3: SG Execution                            │
│         (signal_generator.py — wallet, notional, DD governors)      │
│              ↑ Receives intent, applies safety gates, executes      │
│              │ Maps: ENTER_LONG→BUY, EXIT_LONG→SELL, HOLD→HOLD      │
└─────────────────────────────────────────────────────────────────────┘
                                    ↑
                           Strategy Intent
                                    │
┌─────────────────────────────────────────────────────────────────────┐
│                    LAYER 2: Strategies                              │
│                (pluggable, deterministic alpha modules)             │
│     generate_intent(df, ctx) → {action, confidence, meta, ...}      │
│              ↑ Uses regime classification for entry/exit            │
└─────────────────────────────────────────────────────────────────────┘
                                    ↑
                             RegimeState
                                    │
┌─────────────────────────────────────────────────────────────────────┐
│                    LAYER 1: Regime Engine                           │
│            (authoritative, stable, shared classification)           │
│      classify_regime(df) → RegimeState{label, confidence, ...}      │
└─────────────────────────────────────────────────────────────────────┘
```

**Key Principles:**
- **Layer 1** is authoritative — all strategies share the same regime classification
- **Layer 2** strategies are deterministic — same input → same output, no side effects
- **Layer 3** (SG) is the only component that touches broker, wallet, files, or state
- **Timeline Safety**: `closed_only=True` ensures decisions use closed candles only

---

## Project Structure

```
IteraDynamics_Mono/
│
├── runtime/                      # 🦅 LIVE EXECUTION
│   └── argus/                    # BTC trading service
│       ├── apex_core/            # Signal generation
│       │   └── signal_generator.py  # Layer 3: execution + governors
│       │
│       ├── research/             # 3-LAYER STRATEGY ARCHITECTURE
│       │   ├── __init__.py
│       │   │
│       │   ├── regime/           # LAYER 1: Regime Engine
│       │   │   ├── __init__.py
│       │   │   └── regime_engine.py
│       │   │
│       │   ├── strategies/       # LAYER 2: Strategy Modules
│       │   │   ├── __init__.py
│       │   │   ├── sg_regime_trend_v1.py    # Original trend strategy
│       │   │   ├── sg_core_exposure_v1.py   # Volatility-scaled core
│       │   │   ├── sg_trend_probe_v1.py     # Trend probe (TREND_UP only)
│       │   │   ├── sg_vol_probe_v1.py       # Vol compression breakout
│       │   │   └── sg_stub_strategy.py      # Testing stub
│       │   │
│       │   └── harness/          # Smoke tests & validation
│       │       ├── __init__.py
│       │       ├── regime_smoke.py
│       │       └── strategy_smoke.py
│       │
│       ├── src/
│       │   └── real_broker.py    # Coinbase API integration
│       │
│       ├── models/               # Trained ML models
│       │   └── random_forest.pkl
│       │
│       ├── run_live.py           # Hourly scheduler (main entry)
│       ├── dashboard.py          # Streamlit dashboard
│       ├── cortex.json           # Runtime state/decision log
│       ├── flight_recorder.csv   # OHLCV data
│       ├── prime_state.json      # Live position state
│       └── paper_prime_state.json # Paper trading state
│
├── research/                     # 🔬 STRATEGY R&D (repo root)
│   ├── strategies/               # Backtest-focused strategies
│   ├── engine/                   # Backtest engine
│   ├── backtest/                 # Backtest utilities
│   ├── experiments/              # One-off experiments
│   └── backtests/                # Results & artifacts
│
├── apex_core/                    # 🧠 Asset-agnostic signal engine
├── alpha_engine/                 # 📈 Securities platform (Dormant)
├── moonwire/                     # 🌙 Alternative execution engine
├── scripts/                      # 🛠️ Utilities
├── data/                         # 📊 Historical data
├── output/                       # 📁 Backtest results
│
├── dashboard.py                  # Mission Control (Streamlit)
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
| Priority | Label | Description |
|----------|-------|-------------|
| 1 | PANIC | Extreme volatility + high volume |
| 2 | VOL_EXPANSION | ATR% above threshold |
| 3 | VOL_COMPRESSION | ATR% below threshold |
| 4 | TREND_UP | EMA fast > slow + trend strength |
| 5 | TREND_DOWN | EMA fast < slow + trend strength |
| 6 | CHOP | Default / no clear signal |

### Layer 2: Strategy Modules

All strategies implement the same interface:

```python
def generate_intent(df: pd.DataFrame, ctx: Any, *, closed_only: bool = True) -> dict:
    """
    Returns:
        {
            "action": "ENTER_LONG" | "EXIT_LONG" | "HOLD" | "FLAT",
            "confidence": float,  # 0-1
            "desired_exposure_frac": float,  # 0-1
            "horizon_hours": int,
            "reason": str,
            "meta": dict
        }
    """
```

**Available Strategies:**

| Strategy | File | Description |
|----------|------|-------------|
| **Regime Trend v1** | `sg_regime_trend_v1.py` | Original trend-following with ADX gate |
| **Core Exposure v1** | `sg_core_exposure_v1.py` | Volatility-scaled exposure by regime |
| **Trend Probe v1** | `sg_trend_probe_v1.py` | TREND_UP only, strict trend following |
| **Vol Probe v1** | `sg_vol_probe_v1.py` | VOL_COMPRESSION breakout hunting |
| **Stub** | `sg_stub_strategy.py` | Testing stub (always HOLD) |

### Layer 3: Signal Generator (Execution)

The signal generator (`signal_generator.py`) is the execution layer that:
- Loads external strategies via environment variables
- Applies safety gates (wallet, notional, drawdown governors)
- Maps intents to trades: `ENTER_LONG→BUY`, `EXIT_LONG→SELL`
- Manages position state and horizon exits
- Writes to `cortex.json` and state files

**External Strategy Loading:**
```bash
ARGUS_STRATEGY_MODULE="research.strategies.sg_core_exposure_v1"
ARGUS_STRATEGY_FUNC="generate_intent"
```

---

## Timeline Safety

**All decisions are based on CLOSED candles only.**

When `closed_only=True` (default):
1. The **last row is ALWAYS dropped** before computing indicators
2. `meta.closed_only` and `meta.dropped_last_row` flags are set
3. `asof_ts` is derived from the last **included** (closed) row

This prevents lookahead bias in backtesting and ensures live trading decisions don't rely on potentially-forming candles.

---

## Quick Start

### 1. Installation

```bash
git clone https://github.com/IteraDynamics/IteraDynamics.git
cd IteraDynamics_Mono
pip install -e .
```

### 2. Run Smoke Tests (Safe, No Broker)

```powershell
cd "C:\Users\admin\OneDrive\Desktop\Desktop\IteraDynamics_Mono"

# Run both tests in one Python session
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

Configure your Coinbase API credentials in `.env`:

```env
COINBASE_API_KEY=your_key
COINBASE_API_SECRET=your_secret
COINBASE_PORTFOLIO_UUID=your_portfolio_uuid
```

Start in **dry-run mode** (paper trading):

```powershell
cd "C:\Users\admin\OneDrive\Desktop\Desktop\IteraDynamics_Mono"

# Set safety flags
$env:PRIME_DRY_RUN = "1"
$env:ARGUS_MODE = "prime"

# Use external strategy
$env:ARGUS_STRATEGY_MODULE = "research.strategies.sg_core_exposure_v1"
$env:ARGUS_STRATEGY_FUNC = "generate_intent"

# Run once
python -c "import sys; sys.path.insert(0, r'./runtime/argus'); from apex_core.signal_generator import generate_signals; generate_signals()"
```

### 4. Launch Dashboard

```bash
cd runtime/argus
python -m streamlit run dashboard.py
```

---

## Environment Variables

### Safety / Mode
| Variable | Default | Description |
|----------|---------|-------------|
| `PRIME_DRY_RUN` | `0` | Set to `1` for paper trading |
| `ARGUS_DRY_RUN` | `0` | Alternate dry-run flag |
| `ARGUS_MODE` | `legacy` | Set to `prime` for Prime engine |

### External Strategy
| Variable | Example | Description |
|----------|---------|-------------|
| `ARGUS_STRATEGY_MODULE` | `research.strategies.sg_core_exposure_v1` | Module path |
| `ARGUS_STRATEGY_FUNC` | `generate_intent` | Function name |

### Regime Engine
| Variable | Default | Description |
|----------|---------|-------------|
| `REGIME_EMA_FAST` | `20` | Fast EMA period |
| `REGIME_EMA_SLOW` | `50` | Slow EMA period |
| `REGIME_ATR_LEN` | `14` | ATR period |
| `REGIME_TREND_THRESH` | `0.25` | Trend strength threshold |
| `REGIME_VOL_LO` | `0.003` | Vol compression threshold |
| `REGIME_VOL_HI` | `0.025` | Vol expansion threshold |
| `REGIME_PANIC_HI` | `0.040` | Panic threshold |

### Strategy-Specific

See individual strategy docstrings for full environment variable documentation.

---

## cortex.json Output

The signal generator writes decision state to `cortex.json`:

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

1. Create strategy in `runtime/argus/research/strategies/sg_your_strategy.py`
2. Implement `generate_intent(df, ctx, *, closed_only=True) -> dict`
3. Call Layer 1: `from research.regime import classify_regime`
4. Return the standard intent dict (action, confidence, desired_exposure_frac, horizon_hours, reason, meta)
5. Test with smoke tests before live use

### Running Tests

```powershell
# Regime Engine test
python -c "import sys; sys.path.insert(0, r'./runtime/argus'); from research.harness.regime_smoke import main; main()"

# Strategy test
python -c "import sys; sys.path.insert(0, r'./runtime/argus'); from research.harness.strategy_smoke import main; main()"

# Verify strategy import
python -c "import sys; sys.path.insert(0, r'./runtime/argus'); from research.strategies.sg_core_exposure_v1 import generate_intent; print('OK')"
```

---

## Key Features

### Signal Generation
- **Regime Detection**: Volatility + trend-based market state classification (Layer 1)
- **Pluggable Strategies**: Hot-swappable via environment variables (Layer 2)
- **ML Ensemble**: Optional p_long confirmation from Random Forest model

### Risk Management
- **Drawdown Governors**: Soft/Hard/Kill bands based on peak equity
- **Execution Gates**: Wallet verification, min notional, cooldown periods
- **Timeline Safety**: Closed-candle-only decisions prevent lookahead
- **Panic Exit**: Automatic exit on PANIC regime

### Research Tools
- **Smoke Tests**: Validate strategy logic without broker connection
- **Deterministic Execution**: Same input → same output for reproducibility
- **Backtest Integration**: Strategies work in both live and backtest contexts

---

## License

MIT License - See `LICENSE` for details.

> **Disclaimer**: This software is for educational and research purposes only. Trading involves substantial risk of loss. Past performance does not guarantee future results.

---

## Acknowledgments

Built with Python, pandas, scikit-learn, and Streamlit.
