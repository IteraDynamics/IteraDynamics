# /opt/argus/run_live.py
# 🦅 ARGUS LIVE SCHEDULER - PRIME READY (FULL FILE REPLACEMENT)
#
# - Pins /opt/argus first in sys.path
# - Loads /opt/argus/.env WITHOUT overriding systemd-provided env
# - Schedules generate_signals() at the top of every hour
# - Heartbeat support unchanged
#
# NOTE: Strategy selection happens in apex_core/signal_generator.py via ARGUS_MODE.

from __future__ import annotations

import os
import sys
import time
import signal
import schedule
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

# ---------------------------
# PATH / ENV
# ---------------------------

_THIS_FILE = Path(__file__).resolve()
_PROJECT_ROOT = _THIS_FILE.parent  # /opt/argus

# Force local project root to win imports over site-packages
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Load /opt/argus/.env if present, but NEVER override systemd EnvironmentFile vars.
_ENV_PATH = _PROJECT_ROOT / ".env"
if _ENV_PATH.exists():
    load_dotenv(_ENV_PATH, override=False)
else:
    load_dotenv(override=False)

# ---------------------------
# IMPORT CONFIG + TRADING CYCLE
# ---------------------------

from config import log_startup, LOG_PATH  # noqa: E402
from apex_core.signal_generator import generate_signals  # noqa: E402


# ---------------------------
# LOGGING (namespaced per product when not BTC-USD)
# ---------------------------

class Logger:
    def __init__(self, logfile: str = None):
        self.terminal = sys.stdout
        path = logfile or str(LOG_PATH)
        self.log = open(path, "a", encoding="utf-8")

    def write(self, message: str):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()


sys.stdout = Logger()
sys.stderr = sys.stdout

# Minimal diagnostics: product_id and resolved paths
log_startup()


# ---------------------------
# TIME
# ---------------------------

def utc_now_str() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


# ---------------------------
# HEALTHCHECKS HEARTBEAT
# ---------------------------

def _mask_url(url: str) -> str:
    if not url:
        return ""
    try:
        parts = url.split("/")
        if len(parts) >= 3:
            return f"{parts[0]}//{parts[2]}/…"
    except Exception:
        pass
    return "…"


def hc_ping(url: Optional[str], timeout_sec: float) -> None:
    if not url:
        return
    try:
        req = urllib.request.Request(
            url=url,
            method="GET",
            headers={"User-Agent": "Argus/heartbeat"},
        )
        with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
            _ = resp.read(0)
        print(f"[{utc_now_str()}] ✅ HEARTBEAT SENT -> {_mask_url(url)}")
    except Exception as e:
        print(f"[{utc_now_str()}] ⚠️ HEARTBEAT FAILED -> {_mask_url(url)} | {e}")


ARGUS_ALERTS_ENABLED = os.getenv("ARGUS_ALERTS_ENABLED", "true").strip().lower() in {"1", "true", "yes", "y", "on"}
HC_PING_URL = os.getenv("ARGUS_HEARTBEAT_PING_URL", "").strip()
HC_FAIL_URL = os.getenv("ARGUS_HEARTBEAT_FAIL_URL", "").strip()
ALERT_TIMEOUT_SEC = float(os.getenv("ARGUS_ALERT_TIMEOUT_SEC", "2.5"))

# ---------------------------
# SHUTDOWN CONTROL
# ---------------------------

_shutdown_requested = False


def _handle_shutdown(signum, frame):
    global _shutdown_requested
    print(f"[{utc_now_str()}] ⚠️ RECEIVED SIGNAL {signum} — GRACEFUL SHUTDOWN REQUESTED")
    _shutdown_requested = True


signal.signal(signal.SIGTERM, _handle_shutdown)
signal.signal(signal.SIGINT, _handle_shutdown)


# ---------------------------
# JOB WRAPPER
# ---------------------------

def job():
    print(f"\n[{utc_now_str()}] 🚀 EXECUTION WINDOW OPEN...")
    try:
        print(f"[{utc_now_str()}] 🦅 ARGUS EXECUTION CYCLE...")
        generate_signals()
        print(f"[{utc_now_str()}] 💤 Cycle complete. Sleeping...")

        if ARGUS_ALERTS_ENABLED:
            hc_ping(HC_PING_URL, timeout_sec=ALERT_TIMEOUT_SEC)

    except Exception as e:
        print(f"[{utc_now_str()}] ❌ SCHEDULER ERROR: {e}")
        if ARGUS_ALERTS_ENABLED and HC_FAIL_URL:
            hc_ping(HC_FAIL_URL, timeout_sec=ALERT_TIMEOUT_SEC)


# Run at the top of every hour
schedule.every().hour.at(":00").do(job)


if __name__ == "__main__":
    mode = os.getenv("ARGUS_MODE", "legacy").strip().lower()
    product = os.getenv("ARGUS_PRODUCT_ID", "BTC-USD")
    print("🦅 ARGUS LIVE SCHEDULER ONLINE")
    print(f"⏰ UTC Time: {utc_now_str()}")
    print(f"🧭 ARGUS_MODE: {mode}  ARGUS_PRODUCT_ID: {product}")

    while not _shutdown_requested:
        schedule.run_pending()
        time.sleep(1)

    print(f"[{utc_now_str()}] 🛑 ARGUS SCHEDULER SHUTDOWN COMPLETE")
