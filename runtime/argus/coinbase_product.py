"""
Resolve Coinbase Exchange product id (e.g. BTC-USD, SOL-USD) from env or explicit override.

Supported environment variables (first non-empty wins within each group):

  Full product id:
    COINBASE_PRODUCT_ID
    ARGUS_PRODUCT_ID

  Short base asset (appended with -USD):
    ARGUS_COINBASE_ASSET
    COINBASE_ASSET

Examples:
  set COINBASE_PRODUCT_ID=SOL-USD
  set ARGUS_COINBASE_ASSET=sol   -> SOL-USD

Default if nothing set: BTC-USD
"""

from __future__ import annotations

import os


def normalize_to_product_id(value: str) -> str:
    """
    Accept either a full product id (BTC-USD) or a short code (btc, SOL).
    Coinbase spot USD pairs use BASE-USD.
    """
    s = str(value).strip().upper()
    if not s:
        return "BTC-USD"
    if "-" in s:
        return s
    return f"{s}-USD"


def resolve_coinbase_product_id(explicit: str | None = None) -> str:
    """Resolve product id: explicit arg overrides all env; else env; else BTC-USD."""
    if explicit is not None and str(explicit).strip():
        return normalize_to_product_id(str(explicit).strip())

    for key in ("COINBASE_PRODUCT_ID", "ARGUS_PRODUCT_ID"):
        v = os.getenv(key, "").strip()
        if v:
            return normalize_to_product_id(v)

    for key in ("ARGUS_COINBASE_ASSET", "COINBASE_ASSET"):
        v = os.getenv(key, "").strip()
        if v:
            return normalize_to_product_id(v)

    return "BTC-USD"
