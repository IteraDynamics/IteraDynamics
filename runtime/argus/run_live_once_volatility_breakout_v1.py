"""
Minimal local "live runtime" (single-step) for Itera Dynamics — VB dry-run.

This script simulates one live decision cycle for a single Layer 2 strategy:
  research.strategies.sg_volatility_breakout_v1.generate_intent(df_slice)

Modes:
- Static CSV: --csv path (load from file, decision at last bar).
- Live dry-run: --data-store path (fetch recent hourly candles from Coinbase for --coinbase-product or env, update store).

No real trades; no Prime/signal_generator; no broker integration.
Duplicate-bar protection: state._meta.last_processed_bar_ts skips reprocessing the same bar.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import pandas as pd


def _setup_path() -> Path:
    """
    Ensure runtime/argus is on sys.path (so `import research.*` works).
    Mirrors research harness bootstrapping.
    """
    this_file = Path(__file__).resolve()
    argus_dir = this_file.parent

    if argus_dir.name == "argus" and (argus_dir / "research").exists():
        if str(argus_dir) not in sys.path:
            sys.path.insert(0, str(argus_dir))
        return argus_dir

    for candidate in [Path.cwd() / "runtime" / "argus", Path.cwd()]:
        if (candidate / "research").exists():
            if str(candidate) not in sys.path:
                sys.path.insert(0, str(candidate))
            return candidate

    return Path.cwd()


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            v = json.load(f)
        return v if isinstance(v, dict) else {}
    except Exception:
        return {}


def _json_sanitize(x: Any) -> Any:
    """
    Convert non-JSON-native types to deterministic JSON-safe values.
    - datetime / pandas Timestamp -> ISO-8601 string
    - dict/list/tuple -> sanitized recursively
    """
    if x is None or isinstance(x, (str, int, float, bool)):
        return x
    if isinstance(x, datetime):
        if x.tzinfo is None:
            x = x.replace(tzinfo=timezone.utc)
        return x.astimezone(timezone.utc).isoformat()
    if isinstance(x, pd.Timestamp):
        ts = x
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        return ts.tz_convert("UTC").isoformat()
    if isinstance(x, dict):
        return {str(k): _json_sanitize(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_json_sanitize(v) for v in x]
    return str(x)


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(_json_sanitize(payload), f, separators=(",", ":"), sort_keys=True)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


@dataclass(frozen=True)
class LiveConfig:
    state_path: Path
    lookback: int = 200
    exposure_cap: float = 1.0
    closed_only: bool = True
    log_path: Optional[Path] = None
    csv_path: Optional[Path] = None
    data_store_path: Optional[Path] = None
    #: Coinbase product id (e.g. SOL-USD). None => use COINBASE_PRODUCT_ID / ARGUS_COINBASE_ASSET env.
    coinbase_product_id: Optional[str] = None


def _load_btc_csv_like_harness(csv_path: Path) -> pd.DataFrame:
    """
    Use the same CSV contract as the backtest harness:
      Timestamp, Open, High, Low, Close, Volume(optional)
    """
    from research.harness.backtest_runner import load_flight_recorder

    return load_flight_recorder(str(csv_path))


def _state_default() -> Dict[str, Any]:
    return {
        "in_position": False,
        "entry_price": None,
        "entry_timestamp": None,
        "current_exposure": 0.0,
        "last_action": "INIT",
        "_meta": {
            "updated_at_utc": None,
            "last_bar_ts_utc": None,
            "last_processed_bar_ts": None,
        },
        "_strategy_ctx": {},
    }


def _coerce_state(raw: Dict[str, Any]) -> Dict[str, Any]:
    st = _state_default()
    if isinstance(raw, dict):
        for k in ("in_position", "entry_price", "entry_timestamp", "current_exposure", "last_action"):
            if k in raw:
                st[k] = raw[k]
        if isinstance(raw.get("_meta"), dict):
            st["_meta"].update(raw["_meta"])
        if isinstance(raw.get("_strategy_ctx"), dict):
            st["_strategy_ctx"] = raw["_strategy_ctx"]

    st["in_position"] = bool(st.get("in_position", False))
    try:
        st["current_exposure"] = float(st.get("current_exposure") or 0.0)
    except Exception:
        st["current_exposure"] = 0.0

    return st


def _normalize_action(a: Any) -> str:
    if not a:
        return "HOLD"
    s = str(a).strip().upper()
    if s in ("BUY", "ENTER", "ENTER_LONG", "LONG"):
        return "ENTER"
    if s in ("EXIT", "EXIT_LONG", "SELL", "CLOSE"):
        return "EXIT"
    return "HOLD"


def _append_log_line(path: Path, line: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(line.rstrip("\n") + "\n")
        f.flush()


def run_once(cfg: LiveConfig) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
    """
    Execute one "live" cycle at the most recent closed bar.

    Data source: cfg.data_store_path (live: fetch + update CSV) or cfg.csv_path (static CSV).
    Duplicate-bar protection: if state._meta.last_processed_bar_ts equals current bar ts, skip and return (None, state).

    Returns: (intent, updated_state) or (None, state) when bar already processed.
    """
    _setup_path()

    # 1) Data: live store or static CSV
    if cfg.data_store_path is not None:
        try:
            from live_data import update_coinbase_store
        except ImportError:
            from runtime.argus.live_data import update_coinbase_store
        df = update_coinbase_store(cfg.data_store_path, product_id=cfg.coinbase_product_id)
        if df is None or df.empty:
            raise ValueError("live_data_empty")
    elif cfg.csv_path is not None:
        df = _load_btc_csv_like_harness(cfg.csv_path)
    else:
        raise ValueError("provide either --csv or --data-store")
    if df is None or df.empty:
        raise ValueError("empty_csv")

    # When using live store, enforce CLOSED hourly candles only:
    # Coinbase candle timestamps are hour-starts; we only act on candles strictly before the current UTC hour.
    if cfg.data_store_path is not None:
        now_utc = datetime.now(timezone.utc)
        current_hour_start = now_utc.replace(minute=0, second=0, microsecond=0)
        ts_series = pd.to_datetime(df["Timestamp"], utc=True, errors="coerce")
        cutoff = pd.Timestamp(current_hour_start).tz_convert("UTC")
        df = df.loc[ts_series < cutoff].copy()
        # Conservative decision bar selection:
        # Coinbase may still "mutate" the most recent candle close/high/low for the latest interval
        # (even if its Timestamp looks like a completed hour). To guarantee closed-bar integrity,
        # we exclude the newest candle from decision eligibility and use the second-most-recent candle.
        # This prevents the scenario where `latest_bar_ts` stays the same but `latest_close` changes
        # across cycles due to store overwrites of the newest row.
        df = df.sort_values("Timestamp").reset_index(drop=True)
        if len(df) < 2:
            raise ValueError(f"insufficient_bars_after_conservatism:{len(df)} (need >= 2)")
        df = df.iloc[:-1].copy()  # drop newest potentially mutable candle
        df["Timestamp"] = pd.to_datetime(df["Timestamp"], utc=True, errors="coerce")

    if df.empty or len(df) < cfg.lookback:
        raise ValueError(f"insufficient_bars:{len(df)} (need >= {cfg.lookback})")

    # Decision point: latest closed bar
    bar_ts = df["Timestamp"].iloc[-1]
    ts = pd.Timestamp(bar_ts)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    bar_ts_utc = ts.isoformat()
    last_close = float(df["Close"].iloc[-1])

    # 2) Load state and duplicate-bar protection
    state_raw = _read_json(cfg.state_path)
    state = _coerce_state(state_raw)
    last_processed = (state.get("_meta") or {}).get("last_processed_bar_ts")
    if last_processed is not None and str(last_processed) == bar_ts_utc:
        skip_log = {
            "ts_utc": _utc_now_iso(),
            "latest_bar_ts": bar_ts_utc,
            "latest_close": last_close,
            "skipped": True,
            "reason": "already_processed_same_bar",
        }
        print(json.dumps(skip_log, separators=(",", ":"), sort_keys=True))
        if cfg.log_path is not None:
            _append_log_line(cfg.log_path, json.dumps(skip_log, separators=(",", ":"), sort_keys=True))
        return None, state

    df_slice = df.tail(cfg.lookback).copy()

    # 3) Call Layer 2 strategy (same interface as harness)
    from research.strategies import sg_volatility_breakout_v1

    ctx: Dict[str, Any] = {
        "mode": "live_local",
        "dry_run": True,
        "now_utc": datetime.now(timezone.utc),
    }
    # Preserve strategy-internal state across cycles (optional but useful for parity).
    if isinstance(state.get("_strategy_ctx"), dict):
        ctx.update(state["_strategy_ctx"])

    intent = sg_volatility_breakout_v1.generate_intent(df_slice, ctx, closed_only=cfg.closed_only)
    if not isinstance(intent, dict):
        raise TypeError("strategy_returned_non_dict_intent")

    action = _normalize_action(intent.get("action"))
    desired = float(intent.get("desired_exposure_frac") or 0.0)
    desired = max(0.0, min(1.0, desired))

    # 4) Layer 3: exposure cap + no duplicate entries + clean exits
    capped_desired = min(desired, max(0.0, min(1.0, float(cfg.exposure_cap))))
    in_pos = bool(state["in_position"])

    next_state = dict(state)
    next_state["_meta"] = dict(state.get("_meta") or {})
    next_state["_meta"]["updated_at_utc"] = _utc_now_iso()
    next_state["_meta"]["last_processed_bar_ts"] = bar_ts_utc
    next_state["_strategy_ctx"] = ctx  # persist any strategy internal fields

    if not in_pos:
        if action == "ENTER" and capped_desired > 0:
            next_state["in_position"] = True
            next_state["entry_price"] = last_close
            next_state["entry_timestamp"] = bar_ts_utc
            next_state["current_exposure"] = capped_desired
            next_state["last_action"] = "ENTER"
        else:
            next_state["in_position"] = False
            next_state["current_exposure"] = 0.0
            next_state["last_action"] = "HOLD"
    else:
        # In position: allow clean exit; otherwise hold and track capped exposure (no re-entry)
        if action == "EXIT" or capped_desired <= 0:
            next_state["in_position"] = False
            next_state["entry_price"] = None
            next_state["entry_timestamp"] = None
            next_state["current_exposure"] = 0.0
            next_state["last_action"] = "EXIT"
        else:
            next_state["in_position"] = True
            next_state["current_exposure"] = capped_desired
            next_state["last_action"] = "HOLD"

    # 5) Output + persist
    applied_action = next_state["last_action"]
    log_obj = {
        "ts_utc": _utc_now_iso(),
        "latest_bar_ts": bar_ts_utc,
        "latest_close": last_close,
        "intent_action": str(intent.get("action")),
        "applied_action": applied_action,
        "in_position": bool(next_state["in_position"]),
        "entry_price": next_state.get("entry_price"),
        "last_action": next_state["last_action"],
        "intent_desired_exposure_frac": desired,
        "layer3_exposure_cap": float(cfg.exposure_cap),
        "layer3_applied_exposure": float(next_state["current_exposure"]),
        "reason": intent.get("reason"),
    }
    line = json.dumps(log_obj, separators=(",", ":"), sort_keys=True)
    print(line)
    if cfg.log_path is not None:
        _append_log_line(cfg.log_path, line)

    _atomic_write_json(cfg.state_path, next_state)
    return intent, next_state


if __name__ == "__main__":
    _setup_path()
    import argparse

    ap = argparse.ArgumentParser(
        description="Run one VB dry-run cycle (static CSV or live data store). No real trades."
    )
    ap.add_argument("--csv", type=str, default=None, help="Static OHLCV CSV (backtest harness format). Omit if using --data-store.")
    ap.add_argument("--data-store", type=str, default=None, help="Rolling CSV path for live data (fetch from Coinbase, append, dedupe).")
    ap.add_argument(
        "--coinbase-product",
        type=str,
        default=None,
        help="Coinbase product id (e.g. SOL-USD). Default: COINBASE_PRODUCT_ID, ARGUS_PRODUCT_ID, or ARGUS_COINBASE_ASSET / COINBASE_ASSET, else BTC-USD.",
    )
    ap.add_argument("--state", type=str, default="vb_state.json", help="State JSON path (default: vb_state.json).")
    ap.add_argument("--log", type=str, default=None, help="Optional JSONL log path.")
    ap.add_argument("--lookback", type=int, default=200, help="Lookback bars (default: 200).")
    ap.add_argument("--cap", type=float, default=1.0, help="Exposure cap (default: 1.0).")
    args = ap.parse_args()

    data_store_path = Path(args.data_store).resolve() if args.data_store else None
    csv_path = Path(args.csv).resolve() if args.csv else None
    if data_store_path is None and csv_path is None:
        ap.error("Provide either --csv or --data-store")
    if data_store_path is not None and csv_path is not None:
        ap.error("Provide only one of --csv or --data-store")

    cfg = LiveConfig(
        state_path=Path(args.state).resolve(),
        lookback=int(args.lookback),
        exposure_cap=float(args.cap),
        closed_only=True,
        log_path=Path(args.log).resolve() if args.log else None,
        csv_path=csv_path,
        data_store_path=data_store_path,
        coinbase_product_id=(str(args.coinbase_product).strip() or None) if args.coinbase_product else None,
    )
    run_once(cfg)
