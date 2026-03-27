# VB dry-run runtime — server deployment prep

This document describes how to deploy the **validated VB dry-run** stack to a Linux server. It is **deployment preparation only**: no strategy changes, no broker, no Prime, no MoonWire, no Core.

## What this is (and is not)

- **Separate from Prime**: This path uses `run_live_once_volatility_breakout_v1.py` / `run_live_loop_volatility_breakout_v1.py` only. It does **not** use `signal_generator`, Prime state, or order placement.
- **Dry-run only**: No real trades. No `RealBroker` / Coinbase trading API.
- **Market data**: `live_data.py` uses **Coinbase Exchange public** hourly candles only (`https://api.exchange.coinbase.com/.../candles`). No API keys required for candles.

## Assumptions

| Item | Value |
|------|--------|
| Server layout | Mono-repo–style tree under `/opt/argus` (same relative paths as this repo) |
| Python venv | `/opt/argus/venv` |
| Working directory for manual runs | `/opt/argus` or `/opt/argus/runtime/argus` (see commands below) |

## Files to copy to the server

Copy these paths **preserving directory structure** under `/opt/argus` (i.e. they land as `/opt/argus/runtime/argus/...`).

### VB runner + data fetch

| Local path (repo) | Server path |
|-------------------|-------------|
| `runtime/argus/run_live_once_volatility_breakout_v1.py` | `/opt/argus/runtime/argus/run_live_once_volatility_breakout_v1.py` |
| `runtime/argus/run_live_loop_volatility_breakout_v1.py` | `/opt/argus/runtime/argus/run_live_loop_volatility_breakout_v1.py` |
| `runtime/argus/live_data.py` | `/opt/argus/runtime/argus/live_data.py` |

### Strategy (Layer 2 — do not edit on server)

| Local path | Server path |
|------------|-------------|
| `runtime/argus/research/strategies/sg_volatility_breakout_v1.py` | `/opt/argus/runtime/argus/research/strategies/sg_volatility_breakout_v1.py` |

### Python package scaffolding + harness loader (required imports)

| Local path | Server path |
|------------|-------------|
| `runtime/argus/research/__init__.py` | `/opt/argus/runtime/argus/research/__init__.py` |
| `runtime/argus/research/strategies/__init__.py` | `/opt/argus/runtime/argus/research/strategies/__init__.py` |
| `runtime/argus/research/harness/__init__.py` | `/opt/argus/runtime/argus/research/harness/__init__.py` |
| `runtime/argus/research/harness/backtest_runner.py` | `/opt/argus/runtime/argus/research/harness/backtest_runner.py` |

### Optional (recommended for operators)

| Local path | Server path |
|------------|-------------|
| `runtime/argus/VB_DRY_RUN.md` | `/opt/argus/runtime/argus/VB_DRY_RUN.md` |
| `runtime/argus/deploy/vb_dry_run.service` | `/opt/argus/runtime/argus/deploy/vb_dry_run.service` |
| `runtime/argus/deploy/run_vb_dry_run.sh` | `/opt/argus/runtime/argus/deploy/run_vb_dry_run.sh` |

### Streamlit dashboard (VB dry-run monitor, no broker)

| Local path | Server path |
|------------|-------------|
| `runtime/argus/dashboard.py` | `/opt/argus/runtime/argus/dashboard.py` |
| `runtime/argus/deploy/run_argus_dashboard_vb.sh` | `/opt/argus/runtime/argus/deploy/run_argus_dashboard_vb.sh` |
| `runtime/argus/deploy/argus_dashboard_vb.service` | `/opt/argus/runtime/argus/deploy/argus_dashboard_vb.service` |

The dashboard reads **`vb_state.json`**, **`vb_live_log.jsonl`**, and the CSV store. Set **`VB_DRY_RUN_STATE_PATH`** (etc.) or place `vb_state.json` under `/opt/argus/` so the app can start **without** `src/real_broker.py` on the server.

### Runtime data (created on first run; directory should exist)

Create on server:

```bash
sudo mkdir -p /opt/argus/runtime/argus/data
sudo chown -R YOUR_DEPLOY_USER:YOUR_DEPLOY_USER /opt/argus/runtime/argus/data
```

State and log paths used below:

- `/opt/argus/vb_state.json`
- `/opt/argus/vb_live_log.jsonl`
- `/opt/argus/runtime/argus/data/btc_live_dry_run.csv`

## Python dependencies (server venv)

The runner needs at least: **Python 3.10+**, **pandas**, **numpy**, **requests**.

The Streamlit dashboard additionally needs: **streamlit**, **plotly**, **python-dotenv**.

Example:

```bash
/opt/argus/venv/bin/pip install pandas numpy requests
/opt/argus/venv/bin/pip install streamlit plotly python-dotenv
```

## SCP template (from your laptop)

Replace `USER`, `SERVER`, and adjust repo root if needed.

**Option A — rsync (recommended, preserves tree):**

```bash
RSYNC="rsync -avz --relative"
REPO="/path/to/IteraDynamics_Mono"
DEST="USER@SERVER:/opt/argus/"

$RSYNC "$REPO/./runtime/argus/run_live_once_volatility_breakout_v1.py" "$DEST"
$RSYNC "$REPO/./runtime/argus/run_live_loop_volatility_breakout_v1.py" "$DEST"
$RSYNC "$REPO/./runtime/argus/live_data.py" "$DEST"
$RSYNC "$REPO/./runtime/argus/research/__init__.py" "$DEST"
$RSYNC "$REPO/./runtime/argus/research/strategies/__init__.py" "$DEST"
$RSYNC "$REPO/./runtime/argus/research/strategies/sg_volatility_breakout_v1.py" "$DEST"
$RSYNC "$REPO/./runtime/argus/research/harness/__init__.py" "$DEST"
$RSYNC "$REPO/./runtime/argus/research/harness/backtest_runner.py" "$DEST"
$RSYNC "$REPO/./runtime/argus/deploy/" "$DEST"
$RSYNC "$REPO/./runtime/argus/dashboard.py" "$DEST"
```

**Option B — single scp recursive for `runtime/argus` subtree** (if you sync the whole `runtime/argus` folder from repo):

```bash
scp -r /path/to/IteraDynamics_Mono/runtime/argus USER@SERVER:/opt/argus/runtime/
```

Then remove anything on the server you do not want (only if you copied more than the list above).

## SSH verification (on server)

```bash
ssh USER@SERVER

# Layout
ls -la /opt/argus/runtime/argus/run_live_once_volatility_breakout_v1.py
ls -la /opt/argus/runtime/argus/live_data.py
ls -la /opt/argus/runtime/argus/research/strategies/sg_volatility_breakout_v1.py
ls -la /opt/argus/venv/bin/python

# Quick import check (no network required for import)
cd /opt/argus/runtime/argus
/opt/argus/venv/bin/python -c "import run_live_once_volatility_breakout_v1 as m; print('ok', m.LiveConfig)"

# One dry-run cycle (requires outbound HTTPS to Coinbase)
/opt/argus/venv/bin/python /opt/argus/runtime/argus/run_live_once_volatility_breakout_v1.py \
  --data-store /opt/argus/runtime/argus/data/btc_live_dry_run.csv \
  --state /opt/argus/vb_state.json \
  --log /opt/argus/vb_live_log.jsonl \
  --lookback 200 \
  --cap 1.0
```

Expected: one JSON line on stdout; `vb_state.json` updated; `btc_live_dry_run.csv` created/updated; optional lines appended to `vb_live_log.jsonl`.

## Manual: one VB dry-run cycle (server)

```bash
/opt/argus/venv/bin/python /opt/argus/runtime/argus/run_live_once_volatility_breakout_v1.py \
  --data-store /opt/argus/runtime/argus/data/btc_live_dry_run.csv \
  --state /opt/argus/vb_state.json \
  --log /opt/argus/vb_live_log.jsonl \
  --lookback 200 \
  --cap 1.0
```

## Manual: VB loop (foreground)

```bash
/opt/argus/venv/bin/python /opt/argus/runtime/argus/run_live_loop_volatility_breakout_v1.py \
  --data-store /opt/argus/runtime/argus/data/btc_live_dry_run.csv \
  --state /opt/argus/vb_state.json \
  --log /opt/argus/vb_live_log.jsonl \
  --lookback 200 \
  --cap 1.0 \
  --interval 300
```

Stop with `Ctrl+C`.

## systemd (recommended service name)

- **Unit file (template):** `runtime/argus/deploy/vb_dry_run.service`
- **Suggested service name:** `vb-dry-run.service`

Make the wrapper executable after copy:

```bash
chmod +x /opt/argus/runtime/argus/deploy/run_vb_dry_run.sh
```

Edit `/etc/systemd/system/vb-dry-run.service` and set `User=` / `Group=` to the account that should own `vb_state.json`, logs, and the data directory.

Install:

```bash
sudo cp /opt/argus/runtime/argus/deploy/vb_dry_run.service /etc/systemd/system/vb-dry-run.service
sudo systemctl daemon-reload
sudo systemctl enable vb-dry-run.service
sudo systemctl start vb-dry-run.service
sudo systemctl status vb-dry-run.service
```

Logs:

```bash
journalctl -u vb-dry-run.service -f
```

## Streamlit dashboard (remote monitoring)

**Goal:** Open `http://YOUR_SERVER:8501` in a browser without SSH port-forwarding (after firewall / security steps below).

### 1) Install dashboard dependencies (venv)

```bash
/opt/argus/venv/bin/pip install streamlit plotly python-dotenv
```

### 2) One-off test (foreground)

```bash
chmod +x /opt/argus/runtime/argus/deploy/run_argus_dashboard_vb.sh
/opt/argus/runtime/argus/deploy/run_argus_dashboard_vb.sh
```

The wrapper sets `VB_DRY_RUN_*` to the same paths as the VB loop and binds **`0.0.0.0`** so the UI is reachable from other machines.

### 3) systemd (recommended)

```bash
sudo cp /opt/argus/runtime/argus/deploy/argus_dashboard_vb.service /etc/systemd/system/argus-dashboard-vb.service
# Edit User=/Group= if needed (must read vb_state + log + CSV)
sudo systemctl daemon-reload
sudo systemctl enable --now argus-dashboard-vb.service
sudo systemctl status argus-dashboard-vb.service
journalctl -u argus-dashboard-vb.service -f
```

### 4) Security (do not skip)

Exposing Streamlit on a public IP without protection is risky (anyone who can reach the port sees your monitor).

Pick **at least one**:

| Approach | Notes |
|----------|--------|
| **Firewall** | Allow TCP `8501` (or your `STREAMLIT_PORT`) only from your home IP / VPN. |
| **Tailscale / WireGuard** | Bind stays `0.0.0.0` but only mesh peers can route to the host. |
| **Nginx or Caddy** | Terminate TLS on 443, reverse-proxy to `127.0.0.1:8501`, add HTTP basic auth or IP allowlist. |
| **SSH tunnel** | `ssh -L 8501:127.0.0.1:8501 user@server` and run Streamlit bound to `127.0.0.1` only (change wrapper `--server.address`). |

Override listen port via environment on the unit or wrapper: `STREAMLIT_PORT=8502`.

## Rollback

1. Stop and disable the service:

```bash
sudo systemctl stop vb-dry-run.service
sudo systemctl disable vb-dry-run.service
```

2. Remove the unit (optional):

```bash
sudo rm -f /etc/systemd/system/vb-dry-run.service
sudo systemctl daemon-reload
```

3. Remove or archive artifacts (optional):

```bash
mv /opt/argus/vb_state.json /opt/argus/vb_state.json.bak.$(date +%Y%m%d) 2>/dev/null || true
mv /opt/argus/vb_live_log.jsonl /opt/argus/vb_live_log.jsonl.bak.$(date +%Y%m%d) 2>/dev/null || true
```

VB runner files can remain on disk; they are inert if nothing executes them.

## Post-deploy checklist

- [ ] Outbound HTTPS allowed to `api.exchange.coinbase.com`
- [ ] `vb_state.json` and `vb_live_log.jsonl` writable by the service user
- [ ] `runtime/argus/data/` exists and is writable (CSV store)
- [ ] venv has `pandas`, `numpy`, `requests`
- [ ] (If using dashboard) venv has `streamlit`, `plotly`, `python-dotenv`; firewall / TLS / VPN considered for port 8501
