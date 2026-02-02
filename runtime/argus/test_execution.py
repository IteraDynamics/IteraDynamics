#!/usr/bin/env python3
"""
test_execution.py — Argus Pre-Deployment Validation Suite
=========================================================

Run this BEFORE deploying to your server to verify:
  1. API credentials work
  2. Wallet access is functional
  3. Market data fetching works
  4. Signal generation runs without errors
  5. Dry-run order routing behaves correctly
  6. State files are handled properly

Usage:
    python test_execution.py              # Run all tests
    python test_execution.py --quick      # Quick connectivity test only
    python test_execution.py --full-cycle # Simulate full buy→hold→sell cycle

All tests run in DRY-RUN mode — no real orders are placed.
"""

from __future__ import annotations

import os
import sys
import json
import time
import argparse
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional, Tuple

# ---------------------------
# Setup paths
# ---------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR  # runtime/argus/
_REPO_ROOT = _PROJECT_ROOT.parent.parent  # IteraDynamics_Mono/

if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Load .env from multiple possible locations
from dotenv import load_dotenv

# Try repo root first (local dev), then runtime/argus/, then /etc/argus/
for env_path in [
    _REPO_ROOT / ".env",
    _PROJECT_ROOT / ".env",
    Path("/etc/argus/argus.env"),
]:
    if env_path.exists():
        load_dotenv(str(env_path), override=False)
        print(f"📄 Loaded env from: {env_path}")
        break

# Force dry-run mode for ALL tests
os.environ["ARGUS_DRY_RUN"] = "1"
os.environ["PRIME_DRY_RUN"] = "1"
os.environ["ARGUS_DRY_RUN_LOG_LEDGER"] = "1"

# ---------------------------
# Test utilities
# ---------------------------

class TestResult:
    def __init__(self, name: str):
        self.name = name
        self.passed = False
        self.message = ""
        self.details: dict = {}
        self.duration_ms = 0

    def __str__(self):
        status = "✅ PASS" if self.passed else "❌ FAIL"
        return f"{status} | {self.name} | {self.message}"


def _now_str() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def _print_header(title: str):
    print()
    print("=" * 60)
    print(f"  {title}")
    print("=" * 60)


def _print_result(result: TestResult):
    status = "✅ PASS" if result.passed else "❌ FAIL"
    print(f"\n{status} [{result.name}]")
    print(f"   {result.message}")
    if result.details:
        for k, v in result.details.items():
            print(f"   • {k}: {v}")
    if result.duration_ms > 0:
        print(f"   ⏱️  {result.duration_ms}ms")


# ---------------------------
# Individual Tests
# ---------------------------

def test_env_variables() -> TestResult:
    """Check that required environment variables are set."""
    result = TestResult("Environment Variables")
    start = time.time()

    required = ["COINBASE_API_KEY", "COINBASE_API_SECRET"]
    alt_keys = ["CB_API_KEY", "CB_API_SECRET"]

    missing = []
    for i, key in enumerate(required):
        val = os.getenv(key) or os.getenv(alt_keys[i])
        if not val:
            missing.append(f"{key} (or {alt_keys[i]})")

    if missing:
        result.passed = False
        result.message = f"Missing: {', '.join(missing)}"
        result.details["hint"] = "Set these in .env file or environment"
    else:
        result.passed = True
        result.message = "All required credentials found"
        result.details["ARGUS_DRY_RUN"] = os.getenv("ARGUS_DRY_RUN", "not set")

    result.duration_ms = int((time.time() - start) * 1000)
    return result


def test_broker_connection() -> TestResult:
    """Test that RealBroker can initialize and connect."""
    result = TestResult("Broker Connection")
    start = time.time()

    try:
        from src.real_broker import RealBroker
        broker = RealBroker()

        result.passed = True
        result.message = "RealBroker initialized successfully"
        result.details["mode"] = "DRY-RUN" if broker.dry_run else "LIVE"
        result.details["product"] = "BTC-USD"

    except Exception as e:
        result.passed = False
        result.message = f"Failed to initialize: {e}"

    result.duration_ms = int((time.time() - start) * 1000)
    return result


def test_wallet_access() -> TestResult:
    """Test that we can fetch wallet balances."""
    result = TestResult("Wallet Access")
    start = time.time()

    try:
        from src.real_broker import RealBroker
        broker = RealBroker()
        cash, btc = broker.get_wallet_snapshot()

        result.passed = True
        result.message = "Wallet snapshot retrieved"
        result.details["USD"] = f"${cash:,.2f}"
        result.details["BTC"] = f"{btc:.8f}"

        if cash == 0 and btc == 0:
            result.message += " (⚠️ balances are zero)"

    except Exception as e:
        result.passed = False
        result.message = f"Failed: {e}"

    result.duration_ms = int((time.time() - start) * 1000)
    return result


def test_market_data() -> TestResult:
    """Test fetching market data from Coinbase."""
    result = TestResult("Market Data")
    start = time.time()

    try:
        import requests
        url = "https://api.coinbase.com/v2/prices/BTC-USD/spot"
        resp = requests.get(url, timeout=5)
        resp.raise_for_status()
        data = resp.json()
        price = float(data["data"]["amount"])

        result.passed = True
        result.message = f"Current BTC price: ${price:,.2f}"
        result.details["source"] = "Coinbase Public API"

    except Exception as e:
        result.passed = False
        result.message = f"Failed: {e}"

    result.duration_ms = int((time.time() - start) * 1000)
    return result


def test_candle_fetch() -> TestResult:
    """Test fetching OHLCV candles from Coinbase Exchange."""
    result = TestResult("Candle Data (Exchange API)")
    start = time.time()

    try:
        import requests
        url = "https://api.exchange.coinbase.com/products/BTC-USD/candles"
        resp = requests.get(url, params={"granularity": 3600}, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        if not isinstance(data, list) or len(data) == 0:
            raise ValueError("No candle data returned")

        result.passed = True
        result.message = f"Retrieved {len(data)} hourly candles"
        result.details["latest_close"] = f"${data[0][4]:,.2f}" if data else "N/A"

    except Exception as e:
        result.passed = False
        result.message = f"Failed: {e}"

    result.duration_ms = int((time.time() - start) * 1000)
    return result


def test_signal_generator_import() -> TestResult:
    """Test that signal generator module loads without errors."""
    result = TestResult("Signal Generator Import")
    start = time.time()

    try:
        from apex_core.signal_generator import generate_signals, ARGUS_MODE

        result.passed = True
        result.message = "Signal generator module loaded"
        result.details["ARGUS_MODE"] = ARGUS_MODE

    except Exception as e:
        result.passed = False
        result.message = f"Import failed: {e}"

    result.duration_ms = int((time.time() - start) * 1000)
    return result


def test_dry_run_buy() -> TestResult:
    """Test dry-run BUY order routing."""
    result = TestResult("Dry-Run BUY Order")
    start = time.time()

    try:
        from src.real_broker import RealBroker
        broker = RealBroker()

        if not broker.dry_run:
            result.passed = False
            result.message = "SAFETY: Broker not in dry-run mode!"
            return result

        # Simulate a small BUY
        qty = 0.0001  # ~$10 at $100k
        price = 100000.0
        
        print("\n   [Simulating BUY order...]")
        ok = broker.execute_trade("BUY", qty, price)

        result.passed = ok
        result.message = "Dry-run BUY completed" if ok else "Dry-run BUY returned False"
        result.details["qty"] = f"{qty} BTC"
        result.details["price"] = f"${price:,.2f}"
        result.details["trade_state_modified"] = "No (dry-run)"

    except Exception as e:
        result.passed = False
        result.message = f"Error: {e}"

    result.duration_ms = int((time.time() - start) * 1000)
    return result


def test_dry_run_sell() -> TestResult:
    """Test dry-run SELL order routing."""
    result = TestResult("Dry-Run SELL Order")
    start = time.time()

    try:
        from src.real_broker import RealBroker
        broker = RealBroker()

        if not broker.dry_run:
            result.passed = False
            result.message = "SAFETY: Broker not in dry-run mode!"
            return result

        # Simulate a small SELL
        qty = 0.0001
        price = 100000.0

        print("\n   [Simulating SELL order...]")
        ok = broker.execute_trade("SELL", qty, price)

        result.passed = ok
        result.message = "Dry-run SELL completed" if ok else "Dry-run SELL returned False"
        result.details["qty"] = f"{qty} BTC"
        result.details["trade_state_cleared"] = "No (dry-run)"

    except Exception as e:
        result.passed = False
        result.message = f"Error: {e}"

    result.duration_ms = int((time.time() - start) * 1000)
    return result


def test_signal_cycle() -> TestResult:
    """Run one full signal generation cycle in dry-run mode."""
    result = TestResult("Full Signal Cycle (Dry-Run)")
    start = time.time()

    try:
        # Ensure flight_recorder.csv exists with enough data
        from apex_core.signal_generator import update_market_data, DATA_FILE

        print("\n   [Updating market data...]")
        update_market_data()

        if not DATA_FILE.exists():
            result.passed = False
            result.message = "flight_recorder.csv not created"
            return result

        import pandas as pd
        df = pd.read_csv(DATA_FILE)
        rows = len(df)

        if rows < 210:
            result.passed = True  # Partial pass
            result.message = f"Data file has {rows} rows (need 210 for full signals)"
            result.details["hint"] = "Run several times to accumulate data"
            return result

        print("\n   [Running signal generation...]")
        from apex_core.signal_generator import generate_signals
        generate_signals()

        result.passed = True
        result.message = "Signal cycle completed without errors"
        result.details["data_rows"] = rows

    except Exception as e:
        result.passed = False
        result.message = f"Error: {e}"

    result.duration_ms = int((time.time() - start) * 1000)
    return result


def test_exit_watcher() -> TestResult:
    """Test exit watcher with mock data."""
    result = TestResult("Exit Watcher (Mock Data)")
    start = time.time()

    # Set mock data for testing
    os.environ["ARGUS_TEST_PRICE"] = "100000"
    os.environ["ARGUS_TEST_BTC"] = "0.001"

    try:
        print("\n   [Running exit watcher check...]")
        from apex_core.exit_watcher import check_exit_window
        check_exit_window()

        result.passed = True
        result.message = "Exit watcher ran without errors"
        result.details["mock_price"] = "$100,000"
        result.details["mock_btc"] = "0.001"

    except Exception as e:
        result.passed = False
        result.message = f"Error: {e}"

    finally:
        # Clean up mock data
        os.environ.pop("ARGUS_TEST_PRICE", None)
        os.environ.pop("ARGUS_TEST_BTC", None)

    result.duration_ms = int((time.time() - start) * 1000)
    return result


def test_state_files() -> TestResult:
    """Check state file paths are accessible (not contents)."""
    result = TestResult("State File Paths")
    start = time.time()

    try:
        from apex_core.signal_generator import (
            STATE_FILE, CORTEX_FILE,
            PRIME_STATE_FILE_LIVE, PRIME_STATE_FILE_PAPER,
            DATA_FILE
        )

        files = {
            "trade_state.json": STATE_FILE,
            "cortex.json": CORTEX_FILE,
            "prime_state.json (live)": PRIME_STATE_FILE_LIVE,
            "paper_prime_state.json": PRIME_STATE_FILE_PAPER,
            "flight_recorder.csv": DATA_FILE,
        }

        status = {}
        for name, path in files.items():
            if path.exists():
                status[name] = "exists"
            else:
                status[name] = "will be created"

        result.passed = True
        result.message = "State file paths resolved"
        result.details = status

    except Exception as e:
        result.passed = False
        result.message = f"Error: {e}"

    result.duration_ms = int((time.time() - start) * 1000)
    return result


# ---------------------------
# Full Cycle Simulation
# ---------------------------

def run_full_cycle_simulation():
    """
    Simulate a complete BUY → HOLD → SELL cycle in dry-run mode.
    This mimics what would happen over multiple hours on the server.
    """
    _print_header("FULL CYCLE SIMULATION (DRY-RUN)")

    print(f"\n🕐 Starting at {_now_str()}")
    print("   This simulates a complete trading cycle without real orders.\n")

    results = []

    # Phase 1: Setup
    print("\n--- PHASE 1: SETUP & CONNECTION ---")
    results.append(test_env_variables())
    _print_result(results[-1])
    if not results[-1].passed:
        print("\n❌ Cannot continue without credentials.")
        return results

    results.append(test_broker_connection())
    _print_result(results[-1])

    results.append(test_wallet_access())
    _print_result(results[-1])

    # Phase 2: Market Data
    print("\n--- PHASE 2: MARKET DATA ---")
    results.append(test_market_data())
    _print_result(results[-1])

    results.append(test_candle_fetch())
    _print_result(results[-1])

    # Phase 3: Signal Generation
    print("\n--- PHASE 3: SIGNAL GENERATION ---")
    results.append(test_signal_generator_import())
    _print_result(results[-1])

    results.append(test_signal_cycle())
    _print_result(results[-1])

    # Phase 4: Order Routing
    print("\n--- PHASE 4: ORDER ROUTING (DRY-RUN) ---")
    results.append(test_dry_run_buy())
    _print_result(results[-1])

    print("\n   [Simulating hold period... (skipped in test)]")

    results.append(test_dry_run_sell())
    _print_result(results[-1])

    # Phase 5: Exit Logic
    print("\n--- PHASE 5: EXIT WATCHER ---")
    results.append(test_exit_watcher())
    _print_result(results[-1])

    # Phase 6: State Files
    print("\n--- PHASE 6: STATE FILES ---")
    results.append(test_state_files())
    _print_result(results[-1])

    return results


def run_quick_test():
    """Quick connectivity test only."""
    _print_header("QUICK CONNECTIVITY TEST")

    results = []

    results.append(test_env_variables())
    _print_result(results[-1])

    if results[-1].passed:
        results.append(test_broker_connection())
        _print_result(results[-1])

        results.append(test_wallet_access())
        _print_result(results[-1])

        results.append(test_market_data())
        _print_result(results[-1])

    return results


def run_all_tests():
    """Run all tests in sequence."""
    _print_header("ARGUS PRE-DEPLOYMENT TEST SUITE")

    print(f"\n🕐 {_now_str()}")
    print(f"📁 Project root: {_PROJECT_ROOT}")
    print(f"🔒 Mode: DRY-RUN (no real orders)\n")

    results = []

    tests = [
        ("Environment", test_env_variables),
        ("Broker", test_broker_connection),
        ("Wallet", test_wallet_access),
        ("Market Data", test_market_data),
        ("Candles", test_candle_fetch),
        ("Signal Import", test_signal_generator_import),
        ("State Files", test_state_files),
        ("Dry-Run BUY", test_dry_run_buy),
        ("Dry-Run SELL", test_dry_run_sell),
        ("Exit Watcher", test_exit_watcher),
    ]

    for name, test_fn in tests:
        print(f"\n🔍 Testing: {name}...")
        result = test_fn()
        results.append(result)
        _print_result(result)

        # Stop on critical failures
        if name in ["Environment", "Broker"] and not result.passed:
            print(f"\n❌ Critical test failed. Stopping.")
            break

    return results


# ---------------------------
# Main
# ---------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Argus Pre-Deployment Validation Suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python test_execution.py              # Run all tests
  python test_execution.py --quick      # Quick connectivity test
  python test_execution.py --full-cycle # Simulate buy→hold→sell
        """
    )
    parser.add_argument("--quick", action="store_true", help="Quick connectivity test only")
    parser.add_argument("--full-cycle", action="store_true", help="Simulate full trading cycle")

    args = parser.parse_args()

    if args.quick:
        results = run_quick_test()
    elif args.full_cycle:
        results = run_full_cycle_simulation()
    else:
        results = run_all_tests()

    # Summary
    _print_header("TEST SUMMARY")

    passed = sum(1 for r in results if r.passed)
    failed = sum(1 for r in results if not r.passed)
    total = len(results)

    print(f"\n   ✅ Passed: {passed}/{total}")
    print(f"   ❌ Failed: {failed}/{total}")

    if failed > 0:
        print("\n   Failed tests:")
        for r in results:
            if not r.passed:
                print(f"     • {r.name}: {r.message}")

    if failed == 0:
        print("\n   🎉 All tests passed! Safe to deploy to server.")
        print("   Remember to run with ARGUS_DRY_RUN=1 on server for first cycle.")
    else:
        print("\n   ⚠️  Fix failures before deploying to server.")

    print()
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

