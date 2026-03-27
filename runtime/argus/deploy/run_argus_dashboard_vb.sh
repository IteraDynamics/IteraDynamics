#!/usr/bin/env bash
# Streamlit dashboard for VB dry-run monitoring (no RealBroker required if env is set).
set -euo pipefail

STREAMLIT="${STREAMLIT:-/opt/argus/venv/bin/streamlit}"
DASHBOARD="${DASHBOARD:-/opt/argus/runtime/argus/dashboard.py}"
PORT="${STREAMLIT_PORT:-8501}"

export VB_DRY_RUN_STATE_PATH="${VB_DRY_RUN_STATE_PATH:-/opt/argus/vb_state.json}"
export VB_DRY_RUN_LOG_PATH="${VB_DRY_RUN_LOG_PATH:-/opt/argus/vb_live_log.jsonl}"
export VB_DRY_RUN_DATA_STORE="${VB_DRY_RUN_DATA_STORE:-/opt/argus/runtime/argus/data/btc_live_dry_run.csv}"

cd /opt/argus/runtime/argus

exec "$STREAMLIT" run "$DASHBOARD" \
  --server.address 0.0.0.0 \
  --server.port "$PORT" \
  --server.headless true \
  --browser.gatherUsageStats false
