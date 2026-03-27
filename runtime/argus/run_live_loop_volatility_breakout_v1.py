"""
Lightweight polling wrapper for VB dry-run: calls run_once() on a schedule.

No real trades. Use --data-store for live data or --csv for static. Same CLI as run_live_once
plus --interval (seconds between cycles). Duplicate-bar protection is in run_once (skips same bar).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Ensure we can import the single-step runner
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from run_live_once_volatility_breakout_v1 import LiveConfig, run_once


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Run VB dry-run in a loop (polling). No real trades."
    )
    ap.add_argument("--csv", type=str, default=None, help="Static OHLCV CSV. Omit if using --data-store.")
    ap.add_argument("--data-store", type=str, default=None, help="Rolling CSV for live data (Coinbase).")
    ap.add_argument(
        "--coinbase-product",
        type=str,
        default=None,
        help="Coinbase product (e.g. SOL-USD). Else env COINBASE_PRODUCT_ID / ARGUS_COINBASE_ASSET; default BTC-USD.",
    )
    ap.add_argument("--state", type=str, default="vb_state.json", help="State JSON path.")
    ap.add_argument("--log", type=str, default=None, help="Optional JSONL log path.")
    ap.add_argument("--lookback", type=int, default=200, help="Lookback bars.")
    ap.add_argument("--cap", type=float, default=1.0, help="Exposure cap.")
    ap.add_argument("--interval", type=int, default=300, help="Seconds between cycles (default: 300).")
    args = ap.parse_args()

    data_store_path = Path(args.data_store).resolve() if args.data_store else None
    csv_path = Path(args.csv).resolve() if args.csv else None
    if data_store_path is None and csv_path is None:
        ap.error("Provide either --csv or --data-store")
    if data_store_path is not None and csv_path is not None:
        ap.error("Provide only one of --csv or --data-store")

    cp = args.coinbase_product
    coinbase_product_id = str(cp).strip() if cp else None

    cfg = LiveConfig(
        state_path=Path(args.state).resolve(),
        lookback=int(args.lookback),
        exposure_cap=float(args.cap),
        closed_only=True,
        log_path=Path(args.log).resolve() if args.log else None,
        csv_path=csv_path,
        data_store_path=data_store_path,
        coinbase_product_id=coinbase_product_id,
    )

    interval = max(60, int(args.interval))
    while True:
        try:
            run_once(cfg)
        except KeyboardInterrupt:
            print("Stopped by user.")
            return 0
        except Exception as e:
            print(f"Cycle error: {e}", flush=True)

        try:
            time.sleep(interval)
        except KeyboardInterrupt:
            print("Stopped by user.")
            return 0


if __name__ == "__main__":
    sys.exit(main())
