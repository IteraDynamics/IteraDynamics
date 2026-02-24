# runtime/argus/config.py
# Product and namespaced paths for Argus Prime (BTC-USD / ETH-USD independent instances).
# - ARGUS_PRODUCT_ID env (default BTC-USD) controls product and file namespacing.
# - Legacy filenames used when product is BTC-USD for backward compatibility.

from __future__ import annotations

import os
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent

PRODUCT_ID = os.getenv("ARGUS_PRODUCT_ID", "BTC-USD").strip()
PRODUCT_SLUG = PRODUCT_ID.lower().replace("-", "_")

_USE_LEGACY_NAMES = PRODUCT_SLUG == "btc_usd"


def _base(name: str, namespaced: str) -> Path:
    return _PROJECT_ROOT / (name if _USE_LEGACY_NAMES else namespaced)


# Flight recorder and state paths (safe on Windows and Linux: slug uses '_')
FLIGHT_RECORDER_PATH = _base("flight_recorder.csv", f"flight_recorder_{PRODUCT_SLUG}.csv")
TRADE_STATE_PATH = _base("trade_state.json", f"trade_state_{PRODUCT_SLUG}.json")
TRADE_STATE_TMP_PATH = _PROJECT_ROOT / (
    "trade_state.json.tmp" if _USE_LEGACY_NAMES else f"trade_state_{PRODUCT_SLUG}.json.tmp"
)
TRADE_LEDGER_PATH = _base("trade_ledger.jsonl", f"trade_ledger_{PRODUCT_SLUG}.jsonl")

PRIME_STATE_LIVE_PATH = _base("prime_state.json", f"prime_state_{PRODUCT_SLUG}.json")
PRIME_STATE_LIVE_TMP_PATH = _PROJECT_ROOT / (
    "prime_state.json.tmp" if _USE_LEGACY_NAMES else f"prime_state_{PRODUCT_SLUG}.json.tmp"
)
PRIME_STATE_PAPER_PATH = _base("paper_prime_state.json", f"paper_prime_state_{PRODUCT_SLUG}.json")
PRIME_STATE_PAPER_TMP_PATH = _PROJECT_ROOT / (
    "paper_prime_state.json.tmp"
    if _USE_LEGACY_NAMES
    else f"paper_prime_state_{PRODUCT_SLUG}.json.tmp"
)

CORTEX_PATH = _base("cortex.json", f"cortex_{PRODUCT_SLUG}.json")
CORTEX_TMP_PATH = _PROJECT_ROOT / (
    "cortex.json.tmp" if _USE_LEGACY_NAMES else f"cortex_{PRODUCT_SLUG}.json.tmp"
)

LOG_PATH = _PROJECT_ROOT / ("argus.log" if _USE_LEGACY_NAMES else f"argus_{PRODUCT_SLUG}.log")


def log_startup() -> None:
    """Print product_id and resolved paths at startup (minimal diagnostics)."""
    print(f"[config] ARGUS_PRODUCT_ID={PRODUCT_ID} (slug={PRODUCT_SLUG})")
    print(f"[config] flight_recorder={FLIGHT_RECORDER_PATH}")
    print(f"[config] prime_state(live)={PRIME_STATE_LIVE_PATH} paper={PRIME_STATE_PAPER_PATH}")
    print(f"[config] trade_state={TRADE_STATE_PATH} ledger={TRADE_LEDGER_PATH}")
    print(f"[config] cortex={CORTEX_PATH} log={LOG_PATH}")
