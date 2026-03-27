# MoonWire Experiments (Geometry Baseline)

**Status:** Experiment 1 implemented  
**Last Updated:** 2026-03-05

---

## Experiment 1 — MoonWire-Only Baseline

### Purpose

Evaluate a **MoonWire-only** strategy on BTC using the canonical geometry simulator (no harness). MoonWire drives exposure directly (long/flat; short optional). No Core gating, no Core macro filter, no portfolio allocator. Single-asset BTC only. Closed-bar deterministic: decision at bar *t* applies to return *t*→*t*+1.

### Required Environment Variables

| Variable | Required | Default | Description |
|----------|----------|--------|-------------|
| `MOONWIRE_SIGNAL_FILE` | **Yes** (for BTC_MOONWIRE_ONLY) | — | Path to MoonWire signal feed (JSONL with `timestamp`, `probability`). Feed is produced by moonwire-backend; Itera consumes only. Ensure feed: `python scripts/ensure_moonwire_signal_feed.py`. See `docs/MOONWIRE_INTEGRATION.md`. |
| `MOONWIRE_LONG_THRESH` | No | `0.65` | Probability ≥ this → LONG |
| `MOONWIRE_SHORT_THRESH` | No | `0.35` | Probability ≤ this → SHORT (only if shorts enabled) |
| `MOONWIRE_ALLOW_SHORT` | No | `0` | `1` = enable SHORT signals, `0` = long/flat only |
| `MOONWIRE_MAX_EXPOSURE` | No | `0.985` | Cap on desired exposure (leave room for fees) |
| `MOONWIRE_STRICT_TS_MATCH` | No | `1` | `1` = raise if bar timestamp missing in feed; `0` = treat as HOLD (exposure 0) |
| `MOONWIRE_REQUIRE_EXACT_TS` | No | `1` | Legacy alias for strict timestamp behavior |

### Exact Run Commands (PowerShell, Windows)

From repo root:

```powershell
# 1) Set MoonWire feed (required to include BTC_MOONWIRE_ONLY in output)
#    Feed lives in moonwire-backend repo under feeds/ (sibling to this repo, or set to your export path)
$env:MOONWIRE_SIGNAL_FILE = "C:\Users\admin\OneDrive\Desktop\Desktop\moonwire-backend\feeds\btc_signals.jsonl"

# 2) Optional: cost regime (default is custom 10/5 bps)
$env:MOONWIRE_LONG_THRESH = "0.65"
$env:MOONWIRE_MAX_EXPOSURE = "0.985"

# 3) Run geometry validation — one command for full CSV including BTC_MOONWIRE_ONLY
python research/experiments/portfolio_geometry_validation.py `
  --btc_data_file "data/btcusd_3600s_2019-01-01_to_2025-12-30.csv" `
  --eth_data_file "data/ethusd_3600s_2019-01-01_to_2025-12-30.csv" `
  --cost_regime custom `
  --out_csv "research/experiments/output/portfolio_geometry_validation.csv"
```

To run all cost regimes (retail_launch, pro_target, institutional) with distinct output files:

```powershell
$env:MOONWIRE_SIGNAL_FILE = "C:\Users\admin\OneDrive\Desktop\Desktop\moonwire-backend\feeds\btc_signals.jsonl"
python research/experiments/portfolio_geometry_validation.py `
  --btc_data_file "data/btcusd_3600s_2019-01-01_to_2025-12-30.csv" `
  --eth_data_file "data/ethusd_3600s_2019-01-01_to_2025-12-30.csv" `
  --run_all_cost_regimes
```

### Outputs

- **Consolidated CSV:** `research/experiments/output/portfolio_geometry_validation.csv` (or `*__retail_launch.csv` etc. when using `--run_all_cost_regimes`). Includes scenario `BTC_MOONWIRE_ONLY` with window rows: `full_cycle`, `crash_window`, `post_crash`.
- **Columns:** `scenario`, `window`, `cost_regime`, `fee_bps`, `slippage_bps`, `CAGR`, `MaxDD`, `Calmar`, `Sortino`, `UlcerIndex`, `TimeToRecoveryBars`, `AvgGrossExposure`, `Turnover`, `final_equity_usd`.
- **Trace CSV (MoonWire-only):** `debug/geometry_btc_moonwire_trace.csv` (or `*__<cost_regime>.csv` when using `--run_all_cost_regimes`). Columns: `timestamp`, `close`, `desired_exposure_frac`, `applied_exposure`, `bar_return_px`, `bar_return_applied`, `fee_slippage_this_bar`, `equity_index`, `equity_usd`, `rebalanced`, `cost_regime`.

### Behavior

- If `MOONWIRE_SIGNAL_FILE` is **not** set, the run **skips** scenario `BTC_MOONWIRE_ONLY` (other scenarios unchanged). No error.
- Same inputs (BTC/ETH data, feed, env) → identical output (deterministic).
- Fees/slippage and net mode are applied the same way as for other geometry scenarios.

### Strategy Module

- **Path:** `runtime.argus.research.strategies.sg_moonwire_intent_v1` (or `research.strategies.sg_moonwire_intent_v1` when `runtime/argus` is on `sys.path`).
- **API:** `generate_intent(df, ctx, closed_only=True)` → dict with `desired_exposure_frac`, etc., same contract as Core for the geometry runner.

See also: [MOONWIRE_INTEGRATION.md](MOONWIRE_INTEGRATION.md) for feed export and integration details.
