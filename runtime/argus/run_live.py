# runtime/argus/run_live.py
# 🦅 ARGUS LIVE SCHEDULER - V2.3 (HEALTHCHECKS HEARTBEAT)

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

# --- ENV ---
# Load .env from repo root if present; safe no-op if missing.
_THIS_FILE = Path(__file__).resolve()
_REPO_ROOT = _THIS_FILE.parents[2] if len(_THIS_FILE.parents) >= 3 else _THIS_FILE.parent
load_dotenv(_REPO_ROOT / ".env")


# --- IMPORT TRADING CYCLE ---
from apex_core.signal_generator import generate_signals  # noqa: E402


# --- LOGGING ---
class Logger:
    def __init__(self, logfile: str = "argus.log"):
        self.terminal = sys.stdout
        self.log = open(logfile, "a", encoding="utf-8")

    def write(self, message: str):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()


sys.stdout = Logger()
sys.stderr = sys.stdout


# --- TIME ---
def utc_now_str() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def utc_time_str() -> str:
    return datetime.now(timezone.utc).strftime("%H:%M:%S")


# --- HEALTHCHECKS HEARTBEAT ---
def _mask_url(url: str) -> str:
    # Avoid leaking full tokenized URLs in logs
    if not url:
        return ""
    try:
        # Keep scheme + host only
        # e.g. https://hc-ping.com/UUID -> https://hc-ping.com/…
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
            # Consume response to complete request cleanly
            _ = resp.read(0)
        print(f"[{utc_now_str()}] ✅ HEARTBEAT SENT -> {_mask_url(url)}")
    except Exception as e:
        # Best-effort: never block or raise
        print(f"[{utc_now_str()}] ⚠️ HEARTBEAT FAILED -> {_mask_url(url)} | {e}")


ARGUS_ALERTS_ENABLED = os.getenv("ARGUS_ALERTS_ENABLED", "true").strip().lower() in {"1", "true", "yes", "y", "on"}
HC_PING_URL = os.getenv("ARGUS_HEARTBEAT_PING_URL", "").strip()
HC_FAIL_URL = os.getenv("ARGUS_HEARTBEAT_FAIL_URL", "").strip()
ALERT_TIMEOUT_SEC = float(os.getenv("ARGUS_ALERT_TIMEOUT_SEC", "2.5"))


# --- SHUTDOWN CONTROL ---
_shutdown_requested = False


def _handle_shutdown(signum, frame):
    global _shutdown_requested
    print(f"[{utc_now_str()}] ⚠️ RECEIVED SIGNAL {signum} — GRACEFUL SHUTDOWN REQUESTED")
    _shutdown_requested = True


signal.signal(signal.SIGTERM, _handle_shutdown)
signal.signal(signal.SIGINT, _handle_shutdown)


# --- JOB WRAPPER ---
def job():
    print(f"\n[{utc_now_str()}] 🚀 EXECUTION WINDOW OPEN...")
    try:
        print(f"[{utc_now_str()}] 🦅 ARGUS EXECUTION CYCLE...")
        generate_signals()
        print(f"[{utc_now_str()}] 💤 Cycle complete. Sleeping...")

        # Heartbeat only after successful cycle completion
        if ARGUS_ALERTS_ENABLED:
            hc_ping(HC_PING_URL, timeout_sec=ALERT_TIMEOUT_SEC)

    except Exception as e:
        # Log and emit "fail" heartbeat if configured; never raise
        print(f"[{utc_now_str()}] ❌ SCHEDULER ERROR: {e}")
        if ARGUS_ALERTS_ENABLED and HC_FAIL_URL:
            hc_ping(HC_FAIL_URL, timeout_sec=ALERT_TIMEOUT_SEC)


# Run at the top of every hour
schedule.every().hour.at(":00").do(job)


if __name__ == "__main__":
    print("🦅 ARGUS LIVE SCHEDULER ONLINE")
    print(f"⏰ UTC Time: {utc_now_str()}")

    while not _shutdown_requested:
        schedule.run_pending()
        time.sleep(1)

    print(f"[{utc_now_str()}] 🛑 ARGUS SCHEDULER SHUTDOWN COMPLETE")
