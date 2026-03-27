#!/usr/bin/env bash
# VB dry-run loop wrapper for systemd.
# - Uses venv Python
# - Invokes run_live_loop_volatility_breakout_v1.py with fixed server paths
# - No broker; Coinbase public candles only (inside live_data.py)
set -euo pipefail

VENV_PY="${VENV_PY:-/opt/argus/venv/bin/python}"
LOOP_SCRIPT="${LOOP_SCRIPT:-/opt/argus/runtime/argus/run_live_loop_volatility_breakout_v1.py}"

exec "$VENV_PY" "$LOOP_SCRIPT" \
  --data-store /opt/argus/runtime/argus/data/btc_live_dry_run.csv \
  --state /opt/argus/vb_state.json \
  --log /opt/argus/vb_live_log.jsonl \
  --lookback 200 \
  --cap 1.0 \
  --interval 300
