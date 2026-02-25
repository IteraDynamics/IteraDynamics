# runtime/argus/run_portfolio_live.py
# Portfolio orchestrator: BTC-USD + ETH-USD (and future assets), closed-bar only.
# Single process; executes orders per product in-process. Uses research.portfolio allocator.

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests

# Single path at startup: repo root so research.portfolio and runtime.argus resolve
_THIS_FILE = Path(__file__).resolve()
_PROJECT_ROOT = _THIS_FILE.parent
_REPO_ROOT = _PROJECT_ROOT.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from dotenv import load_dotenv
_ENV_PATH = _PROJECT_ROOT / ".env"
if _ENV_PATH.exists():
    load_dotenv(_ENV_PATH, override=False)
else:
    load_dotenv(override=False)

# Canonical imports (no sys.path alter after this)
from research.portfolio.cross_asset_allocator import (
    PortfolioPolicy,
    PortfolioAllocationDecision,
)
from research.regime.classify_regime import classify_regime
from runtime.argus.apex_core.portfolio_signal_generator import generate_portfolio_decision
from runtime.argus.config import (
    get_paths_for_product,
    PORTFOLIO_MAX_GROSS_EXPOSURE,
    PORTFOLIO_MAX_WEIGHT_PER_ASSET,
    PORTFOLIO_MIN_WEIGHT_PER_ASSET,
    PORTFOLIO_ALLOW_CASH,
)
from runtime.argus.src.real_broker import RealBroker

# ---------------------------------------------------------------------------
# ENV CONTRACT (LOCKED)
# ---------------------------------------------------------------------------

def _get_portfolio_products() -> List[str]:
    raw = os.getenv("ARGUS_PORTFOLIO_PRODUCTS", "").strip()
    if not raw:
        raise ValueError(
            "ARGUS_PORTFOLIO_PRODUCTS is missing or empty. "
            "Set e.g. ARGUS_PORTFOLIO_PRODUCTS=BTC-USD,ETH-USD"
        )
    products = [p.strip().upper() for p in raw.split(",") if p.strip()]
    if not products:
        raise ValueError(
            "ARGUS_PORTFOLIO_PRODUCTS is empty after parsing. "
            "Set e.g. ARGUS_PORTFOLIO_PRODUCTS=BTC-USD,ETH-USD"
        )
    return sorted(products)


# ---------------------------------------------------------------------------
# STATE CONTRACT (LOCKED): portfolio_state.json
# ---------------------------------------------------------------------------

PORTFOLIO_STATE_PATH = _PROJECT_ROOT / "portfolio_state.json"
PORTFOLIO_STATE_TMP_PATH = _PROJECT_ROOT / "portfolio_state.json.tmp"


def _load_portfolio_state() -> Dict[str, Any]:
    if not PORTFOLIO_STATE_PATH.exists():
        return {
            "last_bar_ts_utc": None,
            "prev_target_weights": {},
            "last_decision_meta": None,
        }
    try:
        with open(PORTFOLIO_STATE_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return {"last_bar_ts_utc": None, "prev_target_weights": {}, "last_decision_meta": None}
        data.setdefault("last_bar_ts_utc", None)
        data.setdefault("prev_target_weights", {})
        data.setdefault("last_decision_meta", None)
        return data
    except Exception:
        return {"last_bar_ts_utc": None, "prev_target_weights": {}, "last_decision_meta": None}


def _save_portfolio_state(state: Dict[str, Any]) -> None:
    PORTFOLIO_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(PORTFOLIO_STATE_TMP_PATH, "w", encoding="utf-8") as f:
        json.dump(state, f, separators=(",", ":"), sort_keys=True)
        f.flush()
        os.fsync(f.fileno())
    os.replace(PORTFOLIO_STATE_TMP_PATH, PORTFOLIO_STATE_PATH)


# ---------------------------------------------------------------------------
# Per-product paths
# ---------------------------------------------------------------------------

def _slug(product_id: str) -> str:
    return product_id.strip().upper().replace("-", "_").lower()


def _flight_recorder_path(product_id: str) -> Path:
    s = _slug(product_id)
    name = "flight_recorder.csv" if s == "btc_usd" else f"flight_recorder_{s}.csv"
    return _PROJECT_ROOT / name


def _ledger_path(product_id: str) -> Path:
    return get_paths_for_product(product_id)["ledger_path"]


# ---------------------------------------------------------------------------
# Data: fetch and load (closed-bar only)
# ---------------------------------------------------------------------------

def _update_market_data_for_product(product_id: str) -> None:
    url = f"https://api.exchange.coinbase.com/products/{product_id}/candles"
    path = _flight_recorder_path(product_id)
    try:
        resp = requests.get(url, params={"granularity": 3600}, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        if not isinstance(data, list):
            return
        data.sort(key=lambda x: x[0])
        import pandas as pd
        if path.exists():
            df_existing = pd.read_csv(path)
            if "Timestamp" not in df_existing.columns:
                return
            last_ts = pd.to_datetime(df_existing["Timestamp"], utc=True, errors="coerce").max()
            if pd.isna(last_ts):
                last_ts = pd.Timestamp.min.tz_localize("UTC")
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(columns=["Timestamp", "Open", "High", "Low", "Close", "Volume"]).to_csv(path, index=False)
            last_ts = pd.Timestamp.min.tz_localize("UTC")
        new_rows = []
        for c in data:
            ts = pd.to_datetime(c[0], unit="s", utc=True)
            if ts > last_ts:
                new_rows.append({
                    "Timestamp": ts.strftime("%Y-%m-%d %H:%M:%S"),
                    "Open": c[3], "High": c[2], "Low": c[1], "Close": c[4], "Volume": c[5],
                })
        if new_rows:
            pd.DataFrame(new_rows).to_csv(path, mode="a", header=False, index=False)
    except Exception as e:
        print(f"   >> [PORTFOLIO] Data update failed for {product_id}: {e}")


def _load_df(product_id: str):
    import pandas as pd
    path = _flight_recorder_path(product_id)
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def _last_closed_bar_ts(df) -> Optional[str]:
    if df is None or len(df) < 2:
        return None
    df = df.iloc[:-1]
    ts = df["Timestamp"].iloc[-1]
    return str(ts) if ts is not None and str(ts) != "nan" else None


# ---------------------------------------------------------------------------
# Layer 2 intent
# ---------------------------------------------------------------------------

def _layer2_intent_for_product(product_id: str, df, bar_ts_utc: str) -> Optional[Dict[str, Any]]:
    mod_name = os.getenv("ARGUS_STRATEGY_MODULE", "").strip()
    fn_name = os.getenv("ARGUS_STRATEGY_FUNC", "").strip()
    if not mod_name or not fn_name:
        return None
    try:
        import importlib
        from datetime import datetime, timezone
        mod = importlib.import_module(mod_name)
        fn = getattr(mod, fn_name)
        if not callable(fn):
            return None
        ctx = {
            "mode": "live",
            "dry_run": os.getenv("ARGUS_DRY_RUN", "false").strip().lower() in ("1", "true", "yes"),
            "model_file": os.getenv("ARGUS_MODEL_FILE", "random_forest.pkl"),
            "model_path": str(_PROJECT_ROOT / "models" / os.getenv("ARGUS_MODEL_FILE", "random_forest.pkl")),
            "now_utc": datetime.now(timezone.utc),
        }
        out = fn(df, ctx, closed_only=True)
        if isinstance(out, dict):
            return out
        if hasattr(out, "__dict__"):
            return {k: getattr(out, k) for k in ("action", "confidence", "desired_exposure_frac", "horizon_hours", "reason", "meta") if hasattr(out, k)}
        return None
    except Exception as e:
        print(f"   >> [PORTFOLIO] Layer 2 intent failed for {product_id}: {e}")
        return None


def _get_policy() -> PortfolioPolicy:
    return PortfolioPolicy(
        max_gross_exposure=PORTFOLIO_MAX_GROSS_EXPOSURE,
        max_weight_per_asset=PORTFOLIO_MAX_WEIGHT_PER_ASSET,
        min_weight_per_asset=PORTFOLIO_MIN_WEIGHT_PER_ASSET,
        allow_cash=PORTFOLIO_ALLOW_CASH,
    )


def _utc_now_str() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


# ---------------------------------------------------------------------------
# Portfolio equity (single shared USD, no double counting)
# ---------------------------------------------------------------------------

def _compute_portfolio_equity(
    broker_by_product: Dict[str, RealBroker],
    dfs: Dict[str, Any],
    products: List[str],
) -> Tuple[float, Dict[str, float], Dict[str, Tuple[float, float]]]:
    """
    Use last CLOSED bar price only; one wallet snapshot per product; cash from ONE broker only.
    Returns (portfolio_equity_usd, last_price_by_pid, wallet_by_pid).
    wallet_by_pid[pid] = (cash_usd, base_units).
    """
    last_price_by_pid: Dict[str, float] = {}
    wallet_by_pid: Dict[str, Tuple[float, float]] = {}
    for pid in products:
        df = dfs.get(pid)
        if df is not None and len(df) > 1 and "Close" in df.columns:
            close = df["Close"].iloc[:-1]
            if len(close) > 0:
                last_price_by_pid[pid] = float(close.iloc[-1])
        if last_price_by_pid.get(pid) is None or last_price_by_pid[pid] <= 0:
            raise ValueError(f"invalid_price_for_equity:{pid}")
        broker = broker_by_product.get(pid)
        if broker is None:
            raise ValueError(f"wallet_snapshot_failed:{pid}")
        try:
            cash, base = broker.get_wallet_snapshot()
            wallet_by_pid[pid] = (float(cash), float(base))
        except Exception:
            raise ValueError(f"wallet_snapshot_failed:{pid}")

    # Cash from ONE broker only (first product) — USD wallet is shared
    first_pid = products[0] if products else None
    cash_usd_shared = wallet_by_pid.get(first_pid, (0.0, 0.0))[0] if first_pid else 0.0

    portfolio_equity_usd = cash_usd_shared + sum(
        wallet_by_pid.get(pid, (0.0, 0.0))[1] * last_price_by_pid.get(pid, 0.0)
        for pid in products
    )
    return portfolio_equity_usd, last_price_by_pid, wallet_by_pid


# ---------------------------------------------------------------------------
# Execution: move exposure toward target (closed-bar price only)
# ---------------------------------------------------------------------------

def _execute_target_exposure(
    pid: str,
    broker: RealBroker,
    target_exposure_frac: float,
    portfolio_equity_usd: float,
    last_price_by_pid: Dict[str, float],
    wallet_by_pid: Dict[str, Tuple[float, float]],
    bar_ts_utc: str,
) -> Dict[str, Any]:
    """Place order to move exposure toward target. Uses portfolio-level equity; last closed bar price. Returns result dict for ledger."""
    result: Dict[str, Any] = {"executed": False, "order_id": None, "filled_qty": None, "notional": None, "error": None}
    price = last_price_by_pid.get(pid)
    if price is None or price <= 0:
        result["error"] = "invalid_price"
        return result
    cash_usd, base_units = wallet_by_pid.get(pid, (0.0, 0.0))
    if portfolio_equity_usd <= 0:
        result["error"] = "zero_equity"
        return result
    target_notional_i = portfolio_equity_usd * target_exposure_frac
    current_notional_i = base_units * price
    delta_notional_i = target_notional_i - current_notional_i
    tol = price * 0.0001
    if abs(delta_notional_i) < tol:
        result["executed"] = True
        result["notional"] = current_notional_i
        result["filled_qty"] = base_units
        return result
    try:
        if delta_notional_i > 0:
            # BUY: refresh wallet so we don't overspend
            try:
                cash_usd, base_units = broker.get_wallet_snapshot()
                cash_usd, base_units = float(cash_usd), float(base_units)
            except Exception:
                pass
            max_buy_usd = max(0.0, cash_usd * 0.99)
            buy_notional = min(delta_notional_i, max_buy_usd)
            if buy_notional <= 0:
                result["executed"] = True
                result["notional"] = current_notional_i
                result["filled_qty"] = base_units
                return result
            qty = buy_notional / price
            out = broker.execute_portfolio_trade("BUY", qty, price)
            result["executed"] = bool(out.get("executed", False))
            result["order_id"] = out.get("order_id")
            result["filled_qty"] = out.get("filled_base") if result["executed"] else None
            result["notional"] = out.get("filled_quote") if result["executed"] else None
            if out.get("error"):
                result["error"] = out["error"]
        else:
            sell_qty = min(base_units, abs(delta_notional_i) / price)
            if sell_qty <= 0:
                result["executed"] = True
                result["notional"] = current_notional_i
                result["filled_qty"] = base_units
                return result
            out = broker.execute_portfolio_trade("SELL", sell_qty, price)
            result["executed"] = bool(out.get("executed", False))
            result["order_id"] = out.get("order_id")
            result["filled_qty"] = out.get("filled_base") if result["executed"] else None
            result["notional"] = (out.get("filled_quote") or (out.get("filled_base", 0) * price)) if result["executed"] else None
            if out.get("error"):
                result["error"] = out["error"]
    except Exception as e:
        result["error"] = str(e)
    return result


def _append_ledger(product_id: str, payload: Dict[str, Any]) -> None:
    path = _ledger_path(product_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, separators=(",", ":"), sort_keys=True) + "\n")
        f.flush()


# ---------------------------------------------------------------------------
# Main cycle
# ---------------------------------------------------------------------------

def run_portfolio_cycle() -> None:
    products = _get_portfolio_products()
    policy = _get_policy()
    state = _load_portfolio_state()

    for pid in products:
        _update_market_data_for_product(pid)

    dfs: Dict[str, Any] = {}
    last_closed_by_product: Dict[str, Optional[str]] = {}
    for pid in products:
        df = _load_df(pid)
        dfs[pid] = df
        last_closed_by_product[pid] = _last_closed_bar_ts(df) if df is not None else None

    closed_ts_set = set(last_closed_by_product.values())
    if None in closed_ts_set or len(closed_ts_set) != 1:
        print(f"[{_utc_now_str()}] [PORTFOLIO] Wait: bars not aligned {last_closed_by_product}")
        return
    bar_ts_utc = list(closed_ts_set)[0]

    # Idempotency: already processed this bar
    if state.get("last_bar_ts_utc") == bar_ts_utc:
        print(f"[{_utc_now_str()}] [PORTFOLIO] Already processed bar_ts={bar_ts_utc}; skip.")
        return

    layer1_by_product: Dict[str, Any] = {}
    layer2_by_product: Dict[str, Any] = {}
    for pid in products:
        df = dfs.get(pid)
        if df is None or len(df) < 100:
            print(f"[{_utc_now_str()}] [PORTFOLIO] Skip: insufficient data for {pid}")
            return
        try:
            reg = classify_regime(df, closed_only=True)
            layer1_by_product[pid] = reg
        except Exception as e:
            print(f"[{_utc_now_str()}] [PORTFOLIO] Layer 1 failed for {pid}: {e}")
            return
        intent = _layer2_intent_for_product(pid, df, bar_ts_utc)
        if intent is None:
            intent = {"action": "HOLD", "confidence": 0.0, "desired_exposure_frac": 0.0, "horizon_hours": 0, "reason": "no_strategy", "meta": {}}
        layer2_by_product[pid] = intent

    prev_weights = state.get("prev_target_weights") or {}
    try:
        decision = generate_portfolio_decision(
            dfs_by_product=dfs,
            layer1_outputs_by_product=layer1_by_product,
            layer2_intents_by_product=layer2_by_product,
            policy=policy,
            bar_ts_utc=bar_ts_utc,
            prev_target_weights=prev_weights,
        )
    except ValueError as e:
        print(f"[{_utc_now_str()}] [PORTFOLIO] Allocator error: {e}")
        return

    cap_override = decision.meta.get("cap_override") is True
    print(f"[{_utc_now_str()}] [PORTFOLIO] products={products} bar_ts={bar_ts_utc}")
    print(f"[{_utc_now_str()}] [PORTFOLIO] weights={decision.target_weights} cash_weight={decision.cash_weight} cap_override={cap_override}")

    target_exposure_frac: Dict[str, float] = {
        pid: decision.target_weights.get(pid, 0.0) * policy.max_gross_exposure
        for pid in products
    }

    broker_by_product: Dict[str, RealBroker] = {pid: RealBroker(product_id=pid) for pid in products}
    portfolio_equity_usd, last_price_by_pid, wallet_by_pid = _compute_portfolio_equity(
        broker_by_product, dfs, products
    )
    execution_results: Dict[str, Dict[str, Any]] = {}
    all_ok = True
    for pid in products:
        res = _execute_target_exposure(
            pid,
            broker_by_product[pid],
            target_exposure_frac[pid],
            portfolio_equity_usd,
            last_price_by_pid,
            wallet_by_pid,
            bar_ts_utc,
        )
        execution_results[pid] = res
        if res.get("error") and not res.get("executed"):
            all_ok = False
        elif res.get("executed"):
            try:
                cash, base = broker_by_product[pid].get_wallet_snapshot()
                wallet_by_pid[pid] = (float(cash), float(base))
            except Exception:
                pass

    for pid in products:
        res = execution_results[pid]
        _append_ledger(pid, {
            "ts": _utc_now_str(),
            "bar_ts_utc": bar_ts_utc,
            "source": "run_portfolio_live",
            "target_weight": decision.target_weights.get(pid, 0.0),
            "target_exposure_frac": target_exposure_frac[pid],
            "cash_weight": decision.cash_weight,
            "cap_override": cap_override,
            "reason": decision.reason,
            "executed": res.get("executed", False),
            "order_id": res.get("order_id"),
            "filled_qty": res.get("filled_qty"),
            "notional": res.get("notional"),
            "error": res.get("error"),
        })

    if not all_ok:
        print(f"[{_utc_now_str()}] [PORTFOLIO] One or more executions failed; state NOT updated.")
        return

    _save_portfolio_state({
        "last_bar_ts_utc": bar_ts_utc,
        "prev_target_weights": dict(decision.target_weights),
        "last_decision_meta": dict(decision.meta),
    })
    print(f"[{_utc_now_str()}] [PORTFOLIO] Cycle complete; state updated.")


if __name__ == "__main__":
    run_portfolio_cycle()