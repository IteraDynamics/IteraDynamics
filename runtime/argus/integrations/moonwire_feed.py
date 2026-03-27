"""
MoonWire signal feed orchestration for Itera.

Itera CONSUMES MoonWire outputs only. This module does NOT perform any MoonWire
feature engineering, model loading, or inference. It:
  - Checks whether a MoonWire signal feed file exists
  - Validates freshness and schema (timestamp, probability; optional metadata)
  - If missing or stale, calls moonwire-backend/scripts/export_signal_feed.py
    as an external process (subprocess) and uses the produced JSONL path

Env:
  MOONWIRE_BACKEND_ROOT  Path to moonwire-backend repo (required when export needed)
  MOONWIRE_SIGNAL_FILE   Target JSONL feed path (required)
  MOONWIRE_FEED_MAX_AGE_SECONDS  If set, feed older than this is considered stale (optional)
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

# Required fields per JSONL row; other fields (symbol, model_version, etc.) are allowed
REQUIRED_KEYS = {"timestamp", "probability"}


def _resolve_path(path: str | Path, base: Optional[Path] = None) -> Path:
    p = Path(path)
    if not p.is_absolute() and base is not None:
        p = (base / p).resolve()
    else:
        p = p.resolve()
    return p


def validate_schema(filepath: Path, max_lines: int = 10) -> Tuple[bool, str]:
    """
    Validate that the JSONL file has required fields (timestamp, probability) per line.
    Optional metadata fields are allowed. Returns (valid, message).
    """
    if not filepath.exists():
        return False, "file does not exist"
    try:
        with open(filepath, "r") as f:
            for i, line in enumerate(f):
                if i >= max_lines and max_lines > 0:
                    break
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if not isinstance(obj, dict):
                    return False, f"line {i+1}: expected object, got {type(obj).__name__}"
                for key in REQUIRED_KEYS:
                    if key not in obj:
                        return False, f"line {i+1}: missing required key '{key}'"
                ts = obj["timestamp"]
                prob = obj["probability"]
                if not isinstance(ts, (int, float)):
                    return False, f"line {i+1}: timestamp must be numeric, got {type(ts).__name__}"
                if not isinstance(prob, (int, float)):
                    return False, f"line {i+1}: probability must be numeric, got {type(prob).__name__}"
                if not (0 <= float(prob) <= 1):
                    return False, f"line {i+1}: probability must be in [0,1], got {prob}"
        return True, "schema valid"
    except json.JSONDecodeError as e:
        return False, f"invalid JSON: {e}"
    except OSError as e:
        return False, f"read error: {e}"


def is_fresh(filepath: Path, max_age_seconds: Optional[int] = None) -> bool:
    """Return True if file exists and (if max_age_seconds set) mtime is within max_age_seconds of now."""
    if not filepath.exists():
        return False
    if max_age_seconds is None:
        return True
    try:
        mtime = filepath.stat().st_mtime
        import time
        return (time.time() - mtime) <= max_age_seconds
    except OSError:
        return False


def call_export_signal_feed(
    backend_root: Path,
    signal_file: Path,
    *,
    product: str = "BTC-USD",
    bar_seconds: int = 3600,
    start: str = "2019-01-01",
    end: str = "2025-12-30",
    horizon_hours: int = 3,
    model_dir: str = "models/standard",
) -> bool:
    """
    Call moonwire-backend/scripts/export_signal_feed.py as external process.
    Does NOT import any MoonWire code. Returns True if export succeeded and
    output was written to (or copied to) signal_file.
    """
    export_script = backend_root / "scripts" / "export_signal_feed.py"
    if not export_script.exists():
        logger.error("MoonWire export script not found: %s", export_script)
        return False
    out_stem = backend_root / "feeds" / "btc_signals"
    out_stem.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(export_script),
        "--product", product,
        "--bar_seconds", str(bar_seconds),
        "--start", start,
        "--end", end,
        "--out", str(out_stem),
        "--format", "jsonl",
        "--horizon_hours", str(horizon_hours),
        "--model_dir", model_dir,
    ]
    logger.info("Exporting new feed via moonwire-backend: %s", " ".join(cmd))
    result = subprocess.run(cmd, cwd=str(backend_root))
    if result.returncode != 0:
        logger.error("MoonWire export failed with exit code %s", result.returncode)
        return False
    src = backend_root / "feeds" / "btc_signals.jsonl"
    if not src.exists():
        logger.error("MoonWire export did not produce %s", src)
        return False
    # If target is outside backend, copy into place
    if src.resolve() != signal_file.resolve():
        signal_file.parent.mkdir(parents=True, exist_ok=True)
        import shutil
        shutil.copy2(src, signal_file)
    logger.info("Export complete: %s", signal_file)
    return True


def ensure_feed(
    signal_file: Optional[str | Path] = None,
    backend_root: Optional[str | Path] = None,
    *,
    max_age_seconds: Optional[int] = None,
    product: str = "BTC-USD",
    bar_seconds: int = 3600,
    start: str = "2019-01-01",
    end: str = "2025-12-30",
    horizon_hours: int = 3,
    model_dir: str = "models/standard",
    skip_export_if_missing_backend: bool = True,
) -> Tuple[bool, Optional[Path]]:
    """
    Ensure a MoonWire signal feed exists and is valid. Itera consumes only;
    no inference is done here.

    Returns (success, path). If success is True, path is the resolved feed path.
    If feed is missing or stale and backend is available, calls export_signal_feed
    via subprocess. If backend is missing and skip_export_if_missing_backend is True,
    returns (False, None) without raising.

    Env (used when args not provided):
      MOONWIRE_SIGNAL_FILE   target JSONL path
      MOONWIRE_BACKEND_ROOT  path to moonwire-backend repo
      MOONWIRE_FEED_MAX_AGE_SECONDS  optional freshness (seconds)
    """
    path = signal_file or os.environ.get("MOONWIRE_SIGNAL_FILE")
    if not path:
        logger.warning("feed missing: MOONWIRE_SIGNAL_FILE not set")
        return False, None
    path = _resolve_path(path)
    backend = backend_root or os.environ.get("MOONWIRE_BACKEND_ROOT")
    if backend is not None:
        backend = _resolve_path(backend)
    max_age = max_age_seconds
    if max_age is None and os.environ.get("MOONWIRE_FEED_MAX_AGE_SECONDS"):
        try:
            max_age = int(os.environ["MOONWIRE_FEED_MAX_AGE_SECONDS"])
        except ValueError:
            pass

    if path.exists():
        valid, msg = validate_schema(path)
        if not valid:
            logger.warning("feed schema invalid: %s", msg)
        else:
            logger.info("feed exists; %s", msg)
            if is_fresh(path, max_age):
                logger.info("feed fresh")
                return True, path
            logger.info("feed stale (or max_age check enabled and exceeded)")
    else:
        logger.info("feed missing: %s", path)

    if not backend or not Path(backend).exists():
        logger.warning("MoonWire backend not available (MOONWIRE_BACKEND_ROOT=%s); cannot export", backend)
        if skip_export_if_missing_backend:
            return False, None
        raise FileNotFoundError("MOONWIRE_BACKEND_ROOT required to create feed but not set or path missing")

    ok = call_export_signal_feed(
        Path(backend),
        path,
        product=product,
        bar_seconds=bar_seconds,
        start=start,
        end=end,
        horizon_hours=horizon_hours,
        model_dir=model_dir,
    )
    if not ok:
        return False, None
    valid, msg = validate_schema(path)
    if not valid:
        logger.warning("feed schema invalid after export: %s", msg)
        return False, None
    logger.info("schema valid")
    return True, path
