# download_btc_history.py
# Asset-agnostic hourly candle history via Coinbase Exchange public API.
#
# Usage examples:
#   python download_btc_history.py
#       -> product from env (COINBASE_PRODUCT_ID / ARGUS_COINBASE_ASSET) or BTC-USD
#
#   set ARGUS_COINBASE_ASSET=sol && python download_btc_history.py --outfile data/solusd_3600s_2019-01-01_to_2025-12-30.csv
#
# This script ONLY uses Coinbase's public market-data endpoint (no API keys).
# It does NOT touch RealBroker or your live Argus runtime files.

from __future__ import annotations

import argparse
import csv
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Dict, Any

import requests


COINBASE_EXCHANGE_URL = "https://api.exchange.coinbase.com"
GRANULARITY_DEFAULT = 3600  # 1 hour in seconds
MAX_CANDLES_PER_REQUEST = 300  # per Coinbase docs


def _resolve_argus_import():
    """Load shared Coinbase product resolver from runtime/argus."""
    repo_root = Path(__file__).resolve().parents[2]
    argus_dir = repo_root / "runtime" / "argus"
    if str(argus_dir) not in sys.path:
        sys.path.insert(0, str(argus_dir))
    from coinbase_product import resolve_coinbase_product_id

    return resolve_coinbase_product_id


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _parse_date_utc(date_str: str) -> datetime:
    """
    Parse a date string like '2021-01-01' or full ISO into a UTC datetime.
    """
    try:
        # Try YYYY-MM-DD first
        dt = datetime.strptime(date_str, "%Y-%m-%d")
        return dt.replace(tzinfo=timezone.utc)
    except ValueError:
        # Fallback: let fromisoformat handle full ISO strings
        dt = datetime.fromisoformat(date_str)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)


def fetch_candles(
    product_id: str,
    start: datetime,
    end: datetime,
    granularity: int,
    max_retries: int = 3,
    retry_backoff_sec: float = 2.0,
) -> List[List[Any]]:
    """
    Fetch candles from Coinbase Exchange public API.

    Returns a list of lists, each:
        [ time, low, high, open, close, volume ]

    time is epoch seconds (UTC).
    """
    url = f"{COINBASE_EXCHANGE_URL}/products/{product_id}/candles"

    params = {
        "granularity": granularity,
        "start": start.isoformat(),
        "end": end.isoformat(),
    }

    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.get(url, params=params, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            if not isinstance(data, list):
                raise ValueError(f"Unexpected payload type from Coinbase: {type(data)}")
            return data
        except Exception as e:
            if attempt == max_retries:
                raise
            wait_for = retry_backoff_sec * attempt
            print(
                f"⚠️  Request failed (attempt {attempt}/{max_retries}): {e} "
                f"— retrying in {wait_for:.1f}s...",
                file=sys.stderr,
            )
            time.sleep(wait_for)

    return []  # Should not reach here


def build_output_path(
    project_root: Path,
    outfile_arg: str | None,
    product_id: str,
    granularity: int,
    start: datetime,
    end: datetime,
) -> Path:
    if outfile_arg:
        out_path = Path(outfile_arg)
        if not out_path.is_absolute():
            out_path = project_root / out_path
        return out_path

    # Default: data/<product>_<granularity>s_<start>_to_<end>.csv
    data_dir = project_root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    start_str = start.date().isoformat()
    end_str = end.date().isoformat()
    step_name = f"{granularity}s"  # e.g. "3600s"

    filename = f"{product_id.replace('-', '').lower()}_{step_name}_{start_str}_to_{end_str}.csv"
    return data_dir / filename


def main() -> None:
    current_file = Path(__file__).resolve()
    project_root = current_file.parents[2]
    resolve_coinbase_product_id = _resolve_argus_import()

    parser = argparse.ArgumentParser(
        description="Download historical candles from Coinbase Exchange (public API). Product: --product-id or env (see runtime/argus/coinbase_product.py)."
    )
    parser.add_argument(
        "--product-id",
        type=str,
        default=None,
        help="Coinbase product (e.g. SOL-USD). If omitted, uses COINBASE_PRODUCT_ID / ARGUS_COINBASE_ASSET env; default BTC-USD.",
    )
    parser.add_argument(
        "--granularity",
        type=int,
        default=GRANULARITY_DEFAULT,
        help=f"Candle size in seconds (default: {GRANULARITY_DEFAULT} = 1h).",
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default="2019-01-01",
        help="Start date (UTC) for history, e.g. '2019-01-01' or full ISO (default: 2019-01-01).",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default=None,
        help="End date (UTC). If omitted, uses now().",
    )
    parser.add_argument(
        "--outfile",
        type=str,
        default=None,
        help="Optional output CSV path (relative to project root). "
        "If omitted, will write to data/<auto_name>.csv",
    )

    args = parser.parse_args()

    product_id = resolve_coinbase_product_id(args.product_id)
    granularity = args.granularity
    start = _parse_date_utc(args.start_date)
    end = _utc_now() if args.end_date is None else _parse_date_utc(args.end_date)

    if end <= start:
        print("❌ End date must be after start date.", file=sys.stderr)
        sys.exit(1)

    out_path = build_output_path(
        project_root=project_root,
        outfile_arg=args.outfile,
        product_id=product_id,
        granularity=granularity,
        start=start,
        end=end,
    )

    print(f"📅 Downloading {product_id} candles from {start.isoformat()} to {end.isoformat()}")
    print(f"🕒 Granularity: {granularity} seconds")
    print(f"💾 Output file: {out_path}")

    # Coinbase returns up to MAX_CANDLES_PER_REQUEST candles.
    # For hourly candles, that's 300 hours (~12.5 days) per request.
    step = timedelta(seconds=granularity * MAX_CANDLES_PER_REQUEST)

    all_candles: List[List[Any]] = []

    cursor = start
    req_count = 0

    while cursor < end:
        chunk_end = min(cursor + step, end)
        print(
            f"   → Fetching {product_id} from {cursor.isoformat()} to {chunk_end.isoformat()}...",
            flush=True,
        )

        try:
            candles = fetch_candles(
                product_id=product_id,
                start=cursor,
                end=chunk_end,
                granularity=granularity,
            )
        except Exception as e:
            print(f"❌ Failed to fetch candles for window [{cursor} → {chunk_end}]: {e}", file=sys.stderr)
            sys.exit(1)

        req_count += 1
        all_candles.extend(candles)

        # Advance
        cursor = chunk_end

        # Small pause to be polite to the API
        time.sleep(0.2)

    if not all_candles:
        print("❌ No candles returned. Check parameters / network / Coinbase status.", file=sys.stderr)
        sys.exit(1)

    print(f"✅ Raw candles fetched: {len(all_candles)} across {req_count} request(s)")

    # Coinbase usually returns most-recent-first; sort ascending by time.
    # Candle format: [ time, low, high, open, close, volume ]
    all_candles.sort(key=lambda c: c[0])

    # Deduplicate on timestamp (in case of overlapping windows)
    normalized_rows: List[Dict[str, Any]] = []
    seen_ts = set()

    for c in all_candles:
        ts_epoch = c[0]
        if ts_epoch in seen_ts:
            continue
        seen_ts.add(ts_epoch)

        dt = datetime.fromtimestamp(ts_epoch, tz=timezone.utc)
        row = {
            "Timestamp": dt.strftime("%Y-%m-%d %H:%M:%S"),
            "Open": float(c[3]),
            "High": float(c[2]),
            "Low": float(c[1]),
            "Close": float(c[4]),
            "Volume": float(c[5]),
        }
        normalized_rows.append(row)

    print(f"📈 Normalized candles (deduplicated): {len(normalized_rows)}")

    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["Timestamp", "Open", "High", "Low", "Close", "Volume"],
        )
        writer.writeheader()
        writer.writerows(normalized_rows)

    print(f"💾 Done. Wrote {len(normalized_rows)} rows to: {out_path}")
    print("   You can now use this file for backtesting (e.g., Sniper).")


if __name__ == "__main__":
    main()
