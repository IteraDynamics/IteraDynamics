"""
Dev-only: generate a PLACEHOLDER btc_signals.jsonl from the backtest OHLCV CSV.

NOT part of the canonical MoonWire consumption path. MoonWire owns signal
generation (moonwire-backend/scripts/export_signal_feed.py). Itera consumes only;
use scripts/ensure_moonwire_signal_feed.py to check/validate/call backend.

This script writes one line per bar with a constant probability (default 0.6)
for local alignment testing only. Output: JSONL with timestamp, probability.

Usage (from repo root):
  python scripts/generate_btc_signals_jsonl.py
  python scripts/generate_btc_signals_jsonl.py --csv data/other.csv --out data/other_signals.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate timestamp-aligned btc_signals.jsonl from OHLCV CSV.")
    ap.add_argument("--csv", default="data/btcusd_3600s_2019-01-01_to_2025-12-30.csv", help="Path to OHLCV CSV with Timestamp column")
    ap.add_argument("--out", default="data/btc_signals.jsonl", help="Output JSONL path")
    ap.add_argument("--probability", type=float, default=0.6, help="Default probability (0.6 = bullish at 0.55 thresh)")
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    csv_path = repo_root / args.csv
    out_path = repo_root / args.out

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    if "Timestamp" not in df.columns:
        raise ValueError(f"CSV must have a Timestamp column: {csv_path}")
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], utc=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(out_path, "w") as f:
        for ts in df["Timestamp"]:
            unix_sec = int(ts.timestamp())
            line = json.dumps({"timestamp": unix_sec, "probability": args.probability}) + "\n"
            f.write(line)
            n += 1

    print(f"Wrote {n} lines to {out_path}")

    # If moonwire-backend sibling exists, copy there so MOONWIRE_SIGNAL_FILE path works
    sibling = repo_root.parent / "moonwire-backend" / "feeds" / "btc_signals.jsonl"
    if repo_root.parent.exists() and (repo_root.parent / "moonwire-backend").exists():
        sibling.parent.mkdir(parents=True, exist_ok=True)
        import shutil
        shutil.copy2(out_path, sibling)
        print(f"Copied to {sibling}")


if __name__ == "__main__":
    main()
