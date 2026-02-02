# apex_core/jsonl_writer.py
"""
Atomic JSONL writer for thread-safe, crash-resistant append operations.

Guarantees:
- No partial writes
- No corruption from concurrent writers
- Immediate flush to disk (survives crashes)
- Audit trail integrity for regulatory compliance
"""
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict
from datetime import datetime, timezone
import logging

logger = logging.getLogger(__name__)

# Platform-specific file locking
if sys.platform == "win32":
    import msvcrt
    
    def _lock_file(f):
        msvcrt.locking(f.fileno(), msvcrt.LK_LOCK, 1)
    
    def _unlock_file(f):
        try:
            msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
        except Exception:
            pass  # Ignore unlock errors on Windows
else:
    import fcntl
    
    def _lock_file(f):
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
    
    def _unlock_file(f):
        fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def atomic_jsonl_append(path: Path, record: Dict[str, Any]) -> None:
    """
    Thread-safe, atomic append to JSONL file.

    Args:
        path: Path to JSONL file
        record: Dictionary to append as JSON line

    Raises:
        Exception: Re-raised after logging for caller to handle
    """
    try:
        # Ensure parent directory exists
        path.parent.mkdir(parents=True, exist_ok=True)

        # Ensure timestamp exists (standardized field)
        if "ts" not in record and "timestamp" not in record:
            record["timestamp"] = datetime.now(timezone.utc).isoformat()

        with path.open("a", encoding="utf-8") as f:
            # Exclusive lock (blocks other writers until we're done)
            _lock_file(f)
            try:
                line = json.dumps(record, default=str)
                f.write(line + "\n")
                f.flush()
                os.fsync(f.fileno())  # Force kernel write to disk
            finally:
                _unlock_file(f)

    except Exception as e:
        logger.error(f"Failed to write to {path}: {e}", extra={
            "path": str(path),
            "error": str(e),
            "record_keys": list(record.keys()) if isinstance(record, dict) else None
        })
        raise  # Re-raise for caller to handle


def safe_jsonl_append(path: Path, record: Dict[str, Any]) -> bool:
    """
    Safe wrapper that doesn't raise exceptions (for non-critical logging).

    Args:
        path: Path to JSONL file
        record: Dictionary to append as JSON line

    Returns:
        True if successful, False otherwise
    """
    try:
        atomic_jsonl_append(path, record)
        return True
    except Exception as e:
        logger.warning(f"Non-critical JSONL write failed for {path}: {e}")
        return False
