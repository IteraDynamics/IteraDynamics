"""
Create a deterministic "entry test" CSV for sg_volatility_breakout_v1.

Goal:
- Find the FIRST historical bar where the strategy would enter (or request non-zero exposure)
- Write a truncated CSV that ends at that bar so the live runner will ENTER on first run
- Optionally write a second CSV that includes one bar after entry (useful for testing HOLD)

This is NOT a backtest (no PnL/metrics). It's just scanning + CSV generation.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import pandas as pd


def _setup_argus_path(repo_root: Path) -> None:
    """Ensure runtime/argus is on sys.path so `import research.*` resolves."""
    argus_dir = (repo_root / "runtime" / "argus").resolve()
    if str(argus_dir) not in sys.path:
        sys.path.insert(0, str(argus_dir))


def _load_btc(csv_path: Path) -> pd.DataFrame:
    """
    Load BTC data using the backtest harness loader (canonical CSV contract).
    """
    from research.harness.backtest_runner import load_flight_recorder

    return load_flight_recorder(str(csv_path))


def _normalize_action(a: Any) -> str:
    if not a:
        return "HOLD"
    s = str(a).strip().upper()
    if s in ("ENTER_LONG", "ENTER", "BUY", "LONG"):
        return "ENTER"
    if s in ("EXIT_LONG", "EXIT", "SELL", "CLOSE"):
        return "EXIT"
    return "HOLD"


def _find_first_entry_bar(
    df: pd.DataFrame,
    *,
    lookback: int,
    closed_only: bool,
) -> Tuple[int, pd.Timestamp, Dict[str, Any]]:
    """
    Iterate over dataset and find the first bar where either:
    - intent_action == ENTER_LONG (or BUY/ENTER) OR
    - desired_exposure_frac > 0
    """
    from research.strategies.sg_volatility_breakout_v1 import generate_intent

    if len(df) <= lookback:
        raise ValueError(f"insufficient_bars:{len(df)} (need > {lookback})")

    ctx: Dict[str, Any] = {"mode": "scan_vb_entry_test"}

    for i in range(lookback, len(df)):
        df_slice = df.iloc[: i + 1].copy()
        intent = generate_intent(df_slice, ctx, closed_only=closed_only)
        if not isinstance(intent, dict):
            continue

        action_norm = _normalize_action(intent.get("action"))
        try:
            desired = float(intent.get("desired_exposure_frac") or 0.0)
        except Exception:
            desired = 0.0

        if action_norm == "ENTER" or desired > 0.0:
            ts = pd.to_datetime(df_slice["Timestamp"].iloc[-1], utc=True)
            return i, ts, intent

    raise RuntimeError("no_entry_signal_found")


def _format_timestamp_for_csv(ts: pd.Series) -> pd.Series:
    """
    Write timestamps in a backtest-harness-friendly string form.
    Keep it simple: 'YYYY-MM-DD HH:MM:SS' (UTC).
    """
    t = pd.to_datetime(ts, utc=True, errors="coerce")
    return t.dt.strftime("%Y-%m-%d %H:%M:%S")


def main() -> int:
    repo_root = Path(__file__).resolve().parents[2]
    _setup_argus_path(repo_root)

    ap = argparse.ArgumentParser(description="Create truncated CSVs to force ENTRY/HOLD for sg_volatility_breakout_v1.")
    ap.add_argument("--csv", required=True, help="Input BTC CSV (backtest harness format).")
    ap.add_argument("--lookback", type=int, default=200, help="Lookback bars for scanning (default: 200).")
    ap.add_argument("--out_dir", type=str, default=".", help="Output directory for generated CSVs.")
    ap.add_argument("--no_post", action="store_true", help="Do not generate the post-entry CSV.")
    args = ap.parse_args()

    in_path = Path(args.csv).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    df = _load_btc(in_path)

    entry_idx, entry_ts, entry_intent = _find_first_entry_bar(df, lookback=int(args.lookback), closed_only=True)
    print(f"FIRST ENTRY BAR: index={entry_idx} ts_utc={entry_ts.isoformat()}")
    print(f"  intent.action={entry_intent.get('action')} desired_exposure_frac={entry_intent.get('desired_exposure_frac')} reason={entry_intent.get('reason')}")

    df_entry = df.iloc[: entry_idx + 1].copy()
    df_entry["Timestamp"] = _format_timestamp_for_csv(df_entry["Timestamp"])
    entry_out = out_dir / "btc_vb_entry_test.csv"
    df_entry.to_csv(entry_out, index=False)
    print(f"Wrote: {entry_out} rows={len(df_entry)}")

    if not args.no_post:
        post_end = min(len(df), entry_idx + 2)  # include one additional bar if available
        df_post = df.iloc[:post_end].copy()
        df_post["Timestamp"] = _format_timestamp_for_csv(df_post["Timestamp"])
        post_out = out_dir / "btc_vb_post_entry_test.csv"
        df_post.to_csv(post_out, index=False)
        print(f"Wrote: {post_out} rows={len(df_post)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

