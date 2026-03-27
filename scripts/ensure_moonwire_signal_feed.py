"""
Ensure a MoonWire signal feed exists for Itera to consume.

Itera does NOT generate MoonWire signals. This script only:
  - Checks whether MOONWIRE_SIGNAL_FILE exists
  - Validates freshness and schema (timestamp, probability)
  - If missing or stale, calls moonwire-backend/scripts/export_signal_feed.py
    as an external process

Requires MOONWIRE_BACKEND_ROOT when the feed must be exported. No inference or
signal generation logic runs inside Itera.

Usage (from repo root):
  python scripts/ensure_moonwire_signal_feed.py
  python scripts/ensure_moonwire_signal_feed.py --csv data/btcusd_3600s_2019-01-01_to_2025-12-30.csv
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

# Repo root on path so we can import runtime.argus.integrations
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from runtime.argus.integrations.moonwire_feed import ensure_feed


def _derive_start_end_from_csv(csv_path: Path) -> tuple[str, str]:
    import pandas as pd
    df = pd.read_csv(csv_path)
    if "Timestamp" not in df.columns:
        raise ValueError(f"CSV must have a Timestamp column: {csv_path}")
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], utc=True)
    return df["Timestamp"].min().strftime("%Y-%m-%d"), df["Timestamp"].max().strftime("%Y-%m-%d")


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser(description="Ensure MoonWire signal feed exists (check/validate/call backend).")
    ap.add_argument("--signal-file", default=os.environ.get("MOONWIRE_SIGNAL_FILE"), help="Target JSONL path (or set MOONWIRE_SIGNAL_FILE)")
    ap.add_argument("--backend-root", default=os.environ.get("MOONWIRE_BACKEND_ROOT"), help="Path to moonwire-backend repo (or set MOONWIRE_BACKEND_ROOT)")
    ap.add_argument("--csv", help="OHLCV CSV to derive start/end date range for export")
    ap.add_argument("--start", default="2019-01-01", help="Start date for export (YYYY-MM-DD)")
    ap.add_argument("--end", default="2025-12-30", help="End date for export (YYYY-MM-DD)")
    ap.add_argument("--product", default="BTC-USD", help="Product for export")
    ap.add_argument("--bar-seconds", type=int, default=3600, help="Bar size in seconds")
    ap.add_argument("--horizon-hours", type=int, default=3, help="Horizon for export")
    ap.add_argument("--model-dir", default="models/standard", help="Model dir in moonwire-backend")
    ap.add_argument("--max-age-seconds", type=int, default=None, help="Consider feed stale if older than this (or MOONWIRE_FEED_MAX_AGE_SECONDS)")
    ap.add_argument("--no-skip", action="store_true", help="Raise if backend missing instead of exiting with code 1")
    args = ap.parse_args()

    signal_file = args.signal_file
    if not signal_file:
        print("MOONWIRE_SIGNAL_FILE not set and --signal-file not provided.", file=sys.stderr)
        return 1
    signal_path = Path(signal_file)
    if not signal_path.is_absolute():
        signal_path = (_REPO_ROOT / signal_path).resolve()

    backend_root = args.backend_root
    if not backend_root and (Path(_REPO_ROOT).parent / "moonwire-backend").exists():
        backend_root = str(Path(_REPO_ROOT).parent / "moonwire-backend")
    if backend_root:
        backend_root = Path(backend_root).resolve()

    start, end = args.start, args.end
    if args.csv:
        csv_path = _REPO_ROOT / args.csv if not Path(args.csv).is_absolute() else Path(args.csv)
        if csv_path.exists():
            start, end = _derive_start_end_from_csv(csv_path)

    max_age = args.max_age_seconds
    if max_age is None and os.environ.get("MOONWIRE_FEED_MAX_AGE_SECONDS"):
        try:
            max_age = int(os.environ["MOONWIRE_FEED_MAX_AGE_SECONDS"])
        except ValueError:
            pass

    ok, path = ensure_feed(
        signal_file=signal_path,
        backend_root=backend_root,
        max_age_seconds=max_age,
        product=args.product,
        bar_seconds=args.bar_seconds,
        start=start,
        end=end,
        horizon_hours=args.horizon_hours,
        model_dir=args.model_dir,
        skip_export_if_missing_backend=not args.no_skip,
    )
    if ok:
        print(f"Feed ready: {path}")
        return 0
    print("Feed not available. Set MOONWIRE_BACKEND_ROOT and run export from moonwire-backend if needed.", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
