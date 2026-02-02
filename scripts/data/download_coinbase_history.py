# download_coinbase_history.py
#
# One-shot downloader for Coinbase public candles (no API key).
# Writes a CSV in the SAME schema as flight_recorder.csv:
#   Timestamp,Open,High,Low,Close,Volume
#
# Example (PowerShell):
#   python .\download_coinbase_history.py --product BTC-USD --granularity 3600 --days 730 --out .\data\historical\btc_usd_1h.csv
#
# Notes:
# - Coinbase candles endpoint returns max ~300 candles per request, so we paginate.
# - Uses UTC ISO timestamps.
# - No pandas required. Pure stdlib + requests.
#
# Install dependency:
#   python -m pip install requests

from __future__ import annotations

import argparse
import csv
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Optional, Tuple

import requests


COINBASE_CANDLES_URL = "https://api.exchange.coinbase.com/products/{product}/candles"


@dataclass
class Candle:
    ts: datetime  # UTC
    low: float
    high: float
    open: float
    close: float
    volume: float

    def to_row(self) -> List[str]:
        # Match flight_recorder schema/casing
        return [
            self.ts.strftime("%Y-%m-%d %H:%M:%S"),
            f"{self.open:.8f}",
            f"{self.high:.8f}",
            f"{self.low:.8f}",
            f"{self.close:.8f}",
            f"{self.volume:.8f}",
        ]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Download Coinbase historical candles to flight_recorder-compatible CSV.")
    p.add_argument("--product", type=str, default="BTC-USD", help="Coinbase product id (e.g. BTC-USD, ETH-USD).")
    p.add_argument("--granularity", type=int, default=3600, help="Seconds per candle. 3600=1h, 86400=1d, etc.")
    p.add_argument("--days", type=int, default=365, help="How many days of history to fetch.")
    p.add_argument("--out", type=str, default=None, help="Output CSV path.")
    p.add_argument("--sleep", type=float, default=0.25, help="Seconds to sleep between requests (rate-limit friendly).")
    p.add_argument("--timeout", type=float, default=15.0, help="HTTP timeout seconds.")
    return p.parse_args()


def iso_z(dt: datetime) -> str:
    # Coinbase accepts ISO8601; use 'Z' suffix
    dt = dt.astimezone(timezone.utc)
    return dt.replace(microsecond=0).isoformat().replace("+00:00", "Z")


def fetch_chunk(
    session: requests.Session,
    product: str,
    granularity: int,
    start: datetime,
    end: datetime,
    timeout: float,
) -> List[Candle]:
    """
    Fetch a chunk of candles. Coinbase returns:
      [ time, low, high, open, close, volume ]
    in reverse chronological order.
    """
    url = COINBASE_CANDLES_URL.format(product=product)
    params = {
        "start": iso_z(start),
        "end": iso_z(end),
        "granularity": int(granularity),
    }

    resp = session.get(url, params=params, timeout=timeout, headers={"User-Agent": "offline-research/1.0"})
    if resp.status_code == 429:
        raise RuntimeError("Rate limited (429). Increase --sleep and rerun.")
    if resp.status_code >= 400:
        raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:200]}")

    data = resp.json()
    if not isinstance(data, list):
        raise RuntimeError(f"Unexpected response type: {type(data)}")

    candles: List[Candle] = []
    for row in data:
        # row: [ time, low, high, open, close, volume ]
        if not isinstance(row, (list, tuple)) or len(row) < 6:
            continue
        ts = datetime.fromtimestamp(int(row[0]), tz=timezone.utc)
        candles.append(
            Candle(
                ts=ts,
                low=float(row[1]),
                high=float(row[2]),
                open=float(row[3]),
                close=float(row[4]),
                volume=float(row[5]),
            )
        )

    return candles


def daterange_chunks(end: datetime, start: datetime, chunk_seconds: int) -> List[Tuple[datetime, datetime]]:
    """
    Build [start,end] chunks going backwards.
    Coinbase limit: ~300 candles; so chunk_seconds = granularity * 300 is safe.
    """
    chunks: List[Tuple[datetime, datetime]] = []
    cur_end = end
    while cur_end > start:
        cur_start = max(start, cur_end - timedelta(seconds=chunk_seconds))
        chunks.append((cur_start, cur_end))
        cur_end = cur_start
    # We'll fetch oldest->newest for nicer progress, so reverse
    chunks.reverse()
    return chunks


def dedupe_sort(candles: List[Candle]) -> List[Candle]:
    # Dedupe by timestamp; keep last occurrence
    by_ts = {}
    for c in candles:
        by_ts[c.ts] = c
    out = list(by_ts.values())
    out.sort(key=lambda c: c.ts)
    return out


def write_csv(path: Path, candles: List[Candle]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Timestamp", "Open", "High", "Low", "Close", "Volume"])
        for c in candles:
            w.writerow(c.to_row())


def main() -> None:
    args = parse_args()

    root = Path.cwd()
    out_path = Path(args.out) if args.out else (root / "data" / "historical" / f"{args.product.replace('-','_').lower()}_{args.granularity}s.csv")

    gran = int(args.granularity)
    if gran <= 0:
        raise ValueError("--granularity must be > 0")
    if args.days <= 0:
        raise ValueError("--days must be > 0")

    end = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    start = end - timedelta(days=int(args.days))

    # Safe chunk size: 300 candles max per request
    chunk_seconds = gran * 300

    chunks = daterange_chunks(end=end, start=start, chunk_seconds=chunk_seconds)

    print(f"PRODUCT: {args.product}")
    print(f"GRAN   : {gran} seconds ({gran/3600:.2f} hours)")
    print(f"START  : {start.isoformat()}")
    print(f"END    : {end.isoformat()}")
    print(f"CHUNKS : {len(chunks)} (<=300 candles each)")
    print(f"OUT    : {out_path.resolve()}")

    all_candles: List[Candle] = []

    with requests.Session() as session:
        for idx, (s, e) in enumerate(chunks, start=1):
            try:
                candles = fetch_chunk(session, args.product, gran, s, e, timeout=float(args.timeout))
                all_candles.extend(candles)
                print(f"[{idx:03d}/{len(chunks):03d}] {iso_z(s)} → {iso_z(e)}  got={len(candles)}  total={len(all_candles)}")
            except Exception as ex:
                print(f"[{idx:03d}/{len(chunks):03d}] ERROR {iso_z(s)} → {iso_z(e)}  {type(ex).__name__}: {ex}", file=sys.stderr)
                print("Tip: rerun with larger --sleep (e.g. 1.0) if rate limited.", file=sys.stderr)
                raise
            time.sleep(float(args.sleep))

    all_candles = dedupe_sort(all_candles)

    # Basic sanity: enforce monotonic hourly spacing if 1h
    print(f"DONE. candles_deduped={len(all_candles)}")
    if len(all_candles) >= 2:
        span_days = (all_candles[-1].ts - all_candles[0].ts).total_seconds() / 86400.0
        print(f"SPAN_DAYS: {span_days:.2f}")
        print(f"FIRST   : {all_candles[0].ts.isoformat()}")
        print(f"LAST    : {all_candles[-1].ts.isoformat()}")

    write_csv(out_path, all_candles)
    print(f"WROTE: {out_path.resolve()}")


if __name__ == "__main__":
    main()
