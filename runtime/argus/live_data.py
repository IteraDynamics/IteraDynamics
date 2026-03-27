"""
Live market data for dry-run (no auth, no trading).

Fetches hourly candles from Coinbase Exchange public API for a configurable product
and maintains a rolling CSV store: init from existing, fetch newer candles, append,
dedupe by timestamp, UTC only.

Product selection (override order):
  1) Argument to update_coinbase_store(..., product_id=...)
  2) COINBASE_PRODUCT_ID or ARGUS_PRODUCT_ID (e.g. SOL-USD)
  3) ARGUS_COINBASE_ASSET or COINBASE_ASSET short code (e.g. sol -> SOL-USD)
  4) Default BTC-USD

See runtime/argus/coinbase_product.py
"""

from __future__ import annotations

import importlib.util
from datetime import timezone
from pathlib import Path
from typing import Any, List

import pandas as pd
import requests

_pkg_dir = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location(
    "coinbase_product",
    _pkg_dir / "coinbase_product.py",
)
if _spec is None or _spec.loader is None:  # pragma: no cover
    raise ImportError("coinbase_product.py not found next to live_data.py")
_coinbase_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_coinbase_mod)
resolve_coinbase_product_id = _coinbase_mod.resolve_coinbase_product_id

GRANULARITY_HOUR = 3600
MAX_CANDLES_PER_REQUEST = 300  # Coinbase limit


def fetch_recent_hourly_candles(
    product_id: str | None = None,
    granularity: int = GRANULARITY_HOUR,
    timeout_sec: int = 15,
) -> List[List[Any]]:
    """
    Fetch recent hourly candles from Coinbase Exchange (public, no API key).
    Returns list of [time_unix_sec, low, high, open, close, volume].
    """
    pid = resolve_coinbase_product_id(product_id)
    url = f"https://api.exchange.coinbase.com/products/{pid}/candles"
    params = {"granularity": granularity}
    resp = requests.get(url, params=params, timeout=timeout_sec)
    resp.raise_for_status()
    data = resp.json()
    if not isinstance(data, list):
        raise ValueError(f"Coinbase returned non-list: {type(data)}")
    return data


def _candle_row_to_series(c: list) -> dict:
    """Convert Coinbase candle [time, low, high, open, close, volume] to harness row."""
    ts = pd.Timestamp(c[0], unit="s", tz="UTC")
    return {
        "Timestamp": ts.strftime("%Y-%m-%d %H:%M:%S"),
        "Open": float(c[3]),
        "High": float(c[2]),
        "Low": float(c[1]),
        "Close": float(c[4]),
        "Volume": float(c[5]) if len(c) > 5 else 0.0,
    }


def load_existing_store(csv_path: Path) -> pd.DataFrame:
    """Load existing CSV; return empty DataFrame with correct columns if missing/invalid."""
    cols = ["Timestamp", "Open", "High", "Low", "Close", "Volume"]
    if not csv_path.exists():
        return pd.DataFrame(columns=cols)
    try:
        df = pd.read_csv(csv_path)
        if "Timestamp" not in df.columns or "Close" not in df.columns:
            return pd.DataFrame(columns=cols)
        df["Timestamp"] = pd.to_datetime(df["Timestamp"], utc=True, errors="coerce")
        df = df.dropna(subset=["Timestamp"])
        for c in ("Open", "High", "Low", "Close"):
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        if "Volume" not in df.columns:
            df["Volume"] = 0.0
        else:
            df["Volume"] = pd.to_numeric(df["Volume"], errors="coerce").fillna(0.0)
        return df.dropna(subset=["Open", "High", "Low", "Close"]).reset_index(drop=True)
    except Exception:
        return pd.DataFrame(columns=cols)


def update_coinbase_store(
    csv_path: Path,
    product_id: str | None = None,
) -> pd.DataFrame:
    """
    Initialize from existing CSV if present, fetch newer hourly candles from Coinbase,
    append and deduplicate by timestamp (UTC), then write back to csv_path.
    Returns the updated DataFrame (sorted by Timestamp ascending).

    product_id: explicit Coinbase product (e.g. SOL-USD); if None, uses env / default.
    """
    pid = resolve_coinbase_product_id(product_id)
    df = load_existing_store(csv_path)
    try:
        raw = fetch_recent_hourly_candles(product_id=pid, granularity=GRANULARITY_HOUR)
    except Exception as e:
        if df.empty:
            raise
        # Fetch failed; return existing store so runner can still use last known data
        return df

    if not raw:
        return df.sort_values("Timestamp").reset_index(drop=True) if not df.empty else df

    raw.sort(key=lambda x: x[0])
    new_rows = [_candle_row_to_series(c) for c in raw]
    new_df = pd.DataFrame(new_rows)
    new_df["Timestamp"] = pd.to_datetime(new_df["Timestamp"], utc=True, errors="coerce")

    if df.empty:
        combined = new_df.dropna(subset=["Timestamp", "Open", "High", "Low", "Close"])
    else:
        combined = pd.concat([df, new_df], ignore_index=True)
        combined["Timestamp"] = pd.to_datetime(combined["Timestamp"], utc=True, errors="coerce")
        combined = combined.drop_duplicates(subset=["Timestamp"], keep="last")
        combined = combined.sort_values("Timestamp").reset_index(drop=True)

    combined = combined.dropna(subset=["Open", "High", "Low", "Close"])
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    out = combined.copy()
    out["Timestamp"] = out["Timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")
    out.to_csv(csv_path, index=False)
    combined["Timestamp"] = pd.to_datetime(combined["Timestamp"], utc=True)
    return combined.sort_values("Timestamp").reset_index(drop=True)


def update_btc_store(csv_path: Path, product_id: str | None = None) -> pd.DataFrame:
    """Backward-compatible alias for update_coinbase_store."""
    return update_coinbase_store(csv_path, product_id=product_id)
