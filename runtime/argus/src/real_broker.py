# src/real_broker.py
# 🦅 ARGUS REAL BROKER - V9.5.1 (FULL REPLACEMENT)
# (V9.5 + MAKER BUY SIZING FIX: maker entry now uses SAME USD BUDGET as v9.4 market entry
#  + SAFE MAKER BUY PATH: POST-ONLY LIMIT ENTRY + TIMEOUT/CANCEL + MARKET FALLBACK
#  + PRESERVES: SOP-SAFE ENV LOADING + ATOMIC STATE + IDEMPOTENCY + TZ-AWARE UTC + DRY-RUN SUPPORT
#  + FILL/FEES LEDGER + ORDER-ID RECONCILIATION VIA client_order_id)

import os
import json
import uuid
import time
from decimal import Decimal, ROUND_DOWN
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Tuple, Optional, Dict, List

from dotenv import load_dotenv
from coinbase.rest import RESTClient


# ---------------------------
# Env loading (SOP-safe)
# ---------------------------
# systemd provides EnvironmentFile=/etc/argus/argus.env, so env vars should already be set in the service.
# We still load dotenv for ad-hoc manual runs; we do NOT override existing env vars.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
ETC_ENV = Path("/etc/argus/argus.env")
LOCAL_ENV = PROJECT_ROOT / ".env"

if ETC_ENV.exists():
    load_dotenv(str(ETC_ENV), override=False)
if LOCAL_ENV.exists():
    load_dotenv(str(LOCAL_ENV), override=False)


# ---------------------------
# Constants (product and namespaced paths from config)
# ---------------------------
from config import (
    PRODUCT_ID,
    TRADE_STATE_PATH,
    TRADE_STATE_TMP_PATH,
    TRADE_LEDGER_PATH,
)

HARDCODED_UUID = os.getenv("ARGUS_PORTFOLIO_ID", "5bce9ffb-611c-4dcb-9e18-75d3914825a1")

STATE_FILE = TRADE_STATE_PATH
STATE_TMP = TRADE_STATE_TMP_PATH
LEDGER_FILE = TRADE_LEDGER_PATH


# ---------------------------
# Helpers
# ---------------------------
def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw.strip())
    except Exception:
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw.strip())
    except Exception:
        return default


def _get(obj: Any, key: str, default=None):
    """Safe getter for dict OR object attribute."""
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _to_float(x: Any, default: float = 0.0) -> float:
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_write_json(path: Path, tmp_path: Path, payload: dict) -> None:
    tmp_path.parent.mkdir(parents=True, exist_ok=True)
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, separators=(",", ":"), sort_keys=True)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, path)


def _append_jsonl(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, separators=(",", ":"), sort_keys=True) + "\n")
        f.flush()


def _load_state(path: Path) -> Optional[dict]:
    try:
        if not path.exists():
            return None
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


class RealBroker:
    """
    Live Coinbase broker wrapper.

    SOP behaviors:
    - Uses a single portfolio (HARDCODED_UUID).
    - DRY-RUN toggle via ARGUS_DRY_RUN:
        * No live orders are sent
        * trade_state.json is NOT mutated
        * Wallet snapshot still uses real Coinbase balances
    - LIVE:
        * Submits market orders
        * If order_id missing in immediate response, reconciles by client_order_id via list_orders()
        * Polls order until terminal; requires FILLED to mutate state
        * Pulls fills to compute fill-based avg + fees
        * Writes trade_ledger.jsonl for audit/PnL

    V9.5 extension (minimal):
    - Optional MAKER-based BUY entry:
        * Post-only limit buy attempt (maker)
        * Timeout + cancel
        * Market fallback (default enabled)
        * Preserves: missing order_id reconciliation + atomic state + idempotency guard

    V9.5.1 patch:
    - Maker BUY sizing now matches v9.4: derive a USD budget (qty * signal_price, cents-rounded),
      then compute base_size = usd_budget / limit_price (8dp, rounded down).
    """

    def __init__(self):
        self.api_key = os.getenv("COINBASE_API_KEY") or os.getenv("CB_API_KEY")
        self.api_secret = os.getenv("COINBASE_API_SECRET") or os.getenv("CB_API_SECRET")

        if not self.api_key or not self.api_secret:
            raise ValueError("❌ MISSING API KEYS in env (.env or /etc/argus/argus.env)")

        self.api_secret = self.api_secret.replace("\\n", "\n")

        self.dry_run = _env_bool("ARGUS_DRY_RUN", default=False)
        self.dry_run_ledger = _env_bool("ARGUS_DRY_RUN_LOG_LEDGER", default=False)

        # Maker-entry knobs (LIVE only; DRY_RUN unchanged)
        self.maker_entry_enabled = _env_bool("ARGUS_MAKER_ENTRY", default=False)
        self.maker_timeout_s = _env_int("ARGUS_MAKER_TIMEOUT_SEC", default=45)
        self.maker_poll_s = float(_env_float("ARGUS_MAKER_POLL_SEC", default=2.0))
        self.maker_price_bps = float(_env_float("ARGUS_MAKER_PRICE_BPS", default=0.0))
        self.maker_fallback_market = _env_bool("ARGUS_MAKER_FALLBACK_MARKET", default=True)
        self.maker_allow_partial = _env_bool("ARGUS_MAKER_ALLOW_PARTIAL", default=True)

        self._last_cash: float = 0.0
        self._last_btc: float = 0.0

        try:
            self.client = RESTClient(api_key=self.api_key, api_secret=self.api_secret)
            mode_str = "DRY-RUN (no live orders)" if self.dry_run else "LIVE"
            maker_str = " | MAKER_ENTRY=ON" if (not self.dry_run and self.maker_entry_enabled) else ""
            print(f"🔌 RealBroker: Connected. TARGETING UUID: {HARDCODED_UUID} | MODE: {mode_str}{maker_str}")
        except Exception as e:
            print(f"❌ CONNECTION ERROR: {e}")
            raise

    # ---------------------------
    # Wallet / parsing
    # ---------------------------
    def _get_value(self, obj: Any) -> float:
        if obj is None:
            return 0.0
        if isinstance(obj, dict):
            return float(obj.get("value", 0))
        return float(getattr(obj, "value", 0))

    def _get_accounts(self):
        return self.client.get_accounts(limit=250, portfolio_id=HARDCODED_UUID)

    def get_wallet_snapshot(self) -> Tuple[float, float]:
        response = self._get_accounts()
        cash = 0.0
        btc = 0.0
        for acc in getattr(response, "accounts", []):
            ccy = getattr(acc, "currency", "")
            if ccy == "USD":
                cash = self._get_value(getattr(acc, "available_balance", None))
            elif ccy == "BTC":
                btc = self._get_value(getattr(acc, "available_balance", None))

        self._last_cash = cash
        self._last_btc = btc
        return cash, btc

    @property
    def cash(self) -> float:
        return self._last_cash

    @property
    def positions(self) -> float:
        return self._last_btc

    # ---------------------------
    # trade_state.json (atomic)
    # ---------------------------
    def save_trade_state(self, entry: dict) -> None:
        _atomic_write_json(STATE_FILE, STATE_TMP, entry)
        ep = float(entry.get("entry_price", 0.0))
        print(f"💾 Trade state saved: Entry at ${ep:.2f} (fill-based)")

    def clear_trade_state(self) -> None:
        try:
            if STATE_FILE.exists():
                STATE_FILE.unlink()
                print("🗑️ Trade state cleared.")
        except Exception as e:
            print(f"❌ FAILED TO CLEAR TRADE STATE: {e}")
            raise

    # ---------------------------
    # Order + fills helpers
    # ---------------------------
    def _extract_order_id(self, resp: Any) -> Optional[str]:
        """
        Supports:
        - SDK response objects (market_order_* in your current flow)
        - dict responses (some SDK calls return dict like {'success': True, 'success_response': {'order_id': ...}})
        """
        if resp is None:
            return None

        # Dict-style (common in docs/examples)
        if isinstance(resp, dict):
            # direct
            if resp.get("order_id"):
                return str(resp["order_id"])
            # success_response
            sr = resp.get("success_response") or {}
            if isinstance(sr, dict) and sr.get("order_id"):
                return str(sr["order_id"])
            # nested order
            order = resp.get("order") or {}
            if isinstance(order, dict) and order.get("order_id"):
                return str(order["order_id"])

        oid = _get(resp, "order_id", None)
        if oid:
            return str(oid)
        order = _get(resp, "order", None)
        oid = _get(order, "order_id", None)
        if oid:
            return str(oid)
        return None

    def _response_looks_accepted(self, resp: Any) -> bool:
        """
        Coinbase SDK response objects vary. Historically your V9.0 behavior treated:
          - resp.success True OR resp.order_id present
        as "accepted".
        We'll use a conservative heuristic:
          - success True, OR status in {PENDING, OPEN, ACCEPTED}, OR has client_order_id/order_id.
        """
        if resp is None:
            return False

        # Dict-style
        if isinstance(resp, dict):
            if bool(resp.get("success", False)):
                return True
            sr = resp.get("success_response")
            if isinstance(sr, dict) and sr.get("order_id"):
                return True
            order = resp.get("order")
            if isinstance(order, dict):
                st = (order.get("status") or "").upper().strip()
                if st in {"PENDING", "OPEN", "ACCEPTED"}:
                    return True
                if order.get("order_id") or order.get("client_order_id"):
                    return True

        if bool(_get(resp, "success", False)):
            return True

        # Some responses have 'order' sub-object with status
        order = _get(resp, "order", None) or resp
        status = (_get(order, "status", "") or "").upper().strip()
        if status in {"PENDING", "OPEN", "ACCEPTED"}:
            return True

        if _get(order, "order_id", None) or _get(order, "client_order_id", None) or _get(resp, "client_order_id", None):
            return True

        return False

    def _reconcile_order_id_by_client_order_id(
        self,
        client_order_id: str,
        side: str,
        timeout_s: int = 20,
        poll_s: float = 1.0,
        limit: int = 50,
    ) -> Optional[str]:
        """
        When an order placement returns no order_id, Coinbase still records the order.
        We can look it up by client_order_id via list_orders().
        """
        deadline = time.time() + timeout_s
        side_u = side.upper().strip()

        last_seen = None
        while time.time() < deadline:
            resp = self.client.list_orders(
                product_ids=[PRODUCT_ID],
                retail_portfolio_id=HARDCODED_UUID,
                limit=limit,
            )
            orders = _get(resp, "orders", []) or []
            for o in orders:
                if (_get(o, "client_order_id", None) or "") == client_order_id:
                    last_seen = o
                    oid = _get(o, "order_id", None)
                    if oid:
                        return str(oid)

            time.sleep(poll_s)

        # If we found the order but it still had no order_id (unlikely), print debug
        if last_seen is not None:
            print(f"   ⚠️ Reconcile saw order but no order_id yet | client_order_id={client_order_id} | side={side_u}")
        return None

    def _wait_for_terminal_order(self, order_id: str, timeout_s: int = 60, poll_s: float = 1.0) -> Any:
        deadline = time.time() + timeout_s
        last = None
        while time.time() < deadline:
            resp = self.client.get_order(order_id=order_id)
            order = _get(resp, "order", resp)
            status = (_get(order, "status", "") or "").upper().strip()
            last = order
            if status in {"FILLED", "CANCELED", "CANCELLED", "REJECTED", "EXPIRED", "FAILED"}:
                return order
            time.sleep(poll_s)
        return last

    def _get_fills_for_order(self, order_id: str) -> List[Any]:
        fills_resp = self.client.get_fills(
            order_ids=[order_id],
            retail_portfolio_id=HARDCODED_UUID,
            limit=250,
        )
        fills = _get(fills_resp, "fills", []) or []
        return list(fills)

    def _summarize_fills(self, order_id: str) -> Dict[str, float]:
        """
        Coinbase gotcha (critical):
        - If fill.size_in_quote == True:  size is QUOTE (USD), not BTC
        - If fill.size_in_quote == False: size is BASE (BTC)

        Returns:
          - filled_base (BTC)
          - filled_quote (USD)
          - avg_price (USD/BTC)
          - fees_quote (USD)
          - fee_rate (fees_quote / filled_quote)
        """
        fills = self._get_fills_for_order(order_id)

        filled_base = 0.0
        filled_quote = 0.0
        fees_quote = 0.0

        for f in fills:
            price = _to_float(_get(f, "price", None), 0.0)
            if price <= 0:
                continue

            size = _to_float(_get(f, "size", None), 0.0)
            if size <= 0:
                continue

            size_in_quote = bool(_get(f, "size_in_quote", False))
            fee = _to_float(_get(f, "commission", _get(f, "fee", None)), 0.0)

            if size_in_quote:
                quote = size
                base = quote / price
            else:
                base = size
                quote = base * price

            filled_base += base
            filled_quote += quote
            fees_quote += fee

        avg_price = (filled_quote / filled_base) if filled_base > 0 else 0.0
        fee_rate = (fees_quote / filled_quote) if filled_quote > 0 else 0.0

        return {
            "filled_base": filled_base,
            "filled_quote": filled_quote,
            "avg_price": avg_price,
            "fees_quote": fees_quote,
            "fee_rate": fee_rate,
            "num_fills": float(len(fills)),
        }

    # ---------------------------
    # Maker BUY helpers (minimal + defensive)
    # ---------------------------
    def _quantize_price_down(self, price: Decimal, quote_increment: Decimal) -> Decimal:
        if quote_increment <= 0:
            return price
        ticks = (price / quote_increment).to_integral_value(rounding=ROUND_DOWN)
        return ticks * quote_increment

    def _get_quote_increment(self) -> Optional[Decimal]:
        """
        Attempts to pull quote_increment from product metadata.
        If unavailable, returns None (caller must be conservative).
        """
        try:
            prod = self.client.get_product(PRODUCT_ID)
            qi = None
            if isinstance(prod, dict):
                qi = prod.get("quote_increment") or prod.get("quote_increment_value") or prod.get("quote_increment_decimal")
            else:
                qi = _get(prod, "quote_increment", None)
            if qi is None:
                return None
            return Decimal(str(qi))
        except Exception:
            return None

    def _get_best_bid_ask(self) -> Tuple[Optional[Decimal], Optional[Decimal]]:
        """
        Uses best_bid_ask endpoint if available in the SDK.
        Returns (best_bid, best_ask) as Decimals, or (None, None) if unavailable.
        """
        try:
            resp = self.client.get_best_bid_ask(product_ids=[PRODUCT_ID])
            # Response shape varies; handle common patterns:
            # - {"pricebooks":[{"product_id":"BTC-USD","bids":[{"price":"..."}],"asks":[{"price":"..."}]}]}
            # - {"best_bid_asks":[{"product_id":"BTC-USD","bid":"...","ask":"..."}]}
            if isinstance(resp, dict):
                # pattern A
                pbs = resp.get("pricebooks") or resp.get("price_books") or resp.get("pricebook") or []
                if isinstance(pbs, list):
                    for pb in pbs:
                        if (pb.get("product_id") or "") == PRODUCT_ID:
                            bid = None
                            ask = None
                            bids = pb.get("bids") or []
                            asks = pb.get("asks") or []
                            if bids and isinstance(bids, list):
                                bid = bids[0].get("price") if isinstance(bids[0], dict) else None
                            if asks and isinstance(asks, list):
                                ask = asks[0].get("price") if isinstance(asks[0], dict) else None
                            return (Decimal(str(bid)) if bid else None, Decimal(str(ask)) if ask else None)

                # pattern B
                bba = resp.get("best_bid_asks") or resp.get("best_bid_ask") or resp.get("products") or []
                if isinstance(bba, list):
                    for row in bba:
                        if (row.get("product_id") or "") == PRODUCT_ID:
                            bid = row.get("bid") or row.get("best_bid")
                            ask = row.get("ask") or row.get("best_ask")
                            return (Decimal(str(bid)) if bid else None, Decimal(str(ask)) if ask else None)

            # Object-style fallback
            rows = _get(resp, "pricebooks", None) or _get(resp, "best_bid_asks", None) or []
            for row in rows or []:
                if (_get(row, "product_id", "") or "") == PRODUCT_ID:
                    bid = _get(row, "bid", None) or _get(row, "best_bid", None)
                    ask = _get(row, "ask", None) or _get(row, "best_ask", None)
                    return (Decimal(str(bid)) if bid else None, Decimal(str(ask)) if ask else None)
        except Exception:
            pass
        return (None, None)

    def _place_post_only_limit_buy(self, client_order_id: str, base_size: Decimal, limit_price: Decimal) -> Any:
        """
        Place a post-only GTC limit BUY. Uses the SDK convenience method if available,
        otherwise attempts create_order with limit_limit_gtc configuration.
        """
        try:
            # Some SDK versions accept post_only kwarg; some do not.
            try:
                return self.client.limit_order_gtc_buy(
                    client_order_id=client_order_id,
                    product_id=PRODUCT_ID,
                    base_size=str(base_size),
                    limit_price=str(limit_price),
                    post_only=True,
                )
            except TypeError:
                return self.client.limit_order_gtc_buy(
                    client_order_id=client_order_id,
                    product_id=PRODUCT_ID,
                    base_size=str(base_size),
                    limit_price=str(limit_price),
                )
        except Exception:
            return self.client.create_order(
                client_order_id=client_order_id,
                product_id=PRODUCT_ID,
                side="BUY",
                order_configuration={
                    "limit_limit_gtc": {
                        "base_size": str(base_size),
                        "limit_price": str(limit_price),
                        "post_only": True,
                    }
                },
                retail_portfolio_id=HARDCODED_UUID,
            )

    def _cancel_order_best_effort(self, order_id: Optional[str]) -> None:
        """
        Best-effort cancel using cancel_orders(order_ids=[...]).
        Never raises.
        """
        try:
            if not order_id:
                return
            self.client.cancel_orders(order_ids=[order_id])
        except Exception as e:
            print(f"   ⚠️ CANCEL ERROR (non-fatal) | order_id={order_id} | {e}")

    def _wait_for_fill_or_timeout(
        self,
        order_id: str,
        timeout_s: int,
        poll_s: float,
        allow_partial: bool,
    ) -> Tuple[str, Dict[str, float]]:
        """
        Polls until:
          - FILLED (returns status 'FILLED')
          - partial fill detected and allow_partial True (returns status 'PARTIAL')
          - terminal non-filled status (returns that status)
          - timeout (returns 'TIMEOUT')

        Always returns (status, fill_summary) where fill_summary may be zeros.
        """
        deadline = time.time() + max(1, int(timeout_s))
        last_status = "UNKNOWN"
        last_fill = {"filled_base": 0.0, "filled_quote": 0.0, "avg_price": 0.0, "fees_quote": 0.0, "fee_rate": 0.0, "num_fills": 0.0}

        while time.time() < deadline:
            try:
                oresp = self.client.get_order(order_id=order_id)
                order = _get(oresp, "order", oresp)
                last_status = (_get(order, "status", "") or "").upper().strip() or "UNKNOWN"

                # Pull fills each loop (small volume) so we can detect partials quickly.
                last_fill = self._summarize_fills(order_id)

                if last_status == "FILLED":
                    return ("FILLED", last_fill)

                if allow_partial and last_fill.get("filled_base", 0.0) > 0.0:
                    # We have some fill; caller should cancel remainder then re-check.
                    return ("PARTIAL", last_fill)

                if last_status in {"CANCELED", "CANCELLED", "REJECTED", "EXPIRED", "FAILED"}:
                    return (last_status, last_fill)

            except Exception as e:
                print(f"   ⚠️ POLL ERROR (non-fatal) | order_id={order_id} | {e}")

            time.sleep(max(0.25, float(poll_s)))

        return ("TIMEOUT", last_fill)

    def _execute_maker_buy_with_fallback(self, qty_btc: float, signal_price: float, client_order_id: str) -> bool:
        """
        Minimal maker entry:
        - places post-only limit BUY at (best_bid adjusted) (never crosses ask)
        - waits for fill up to timeout
        - cancels on timeout/no fill
        - falls back to existing MARKET BUY path if enabled
        - preserves idempotency guard (caller already checked trade_state.json)

        V9.5.1 sizing patch:
        - computes usd_size exactly like v9.4 (qty * signal_price, cents rounded)
        - derives base_size = usd_size / limit_price (8dp, rounded down)
        """
        # Pull top-of-book + increments (best effort)
        best_bid, best_ask = self._get_best_bid_ask()
        quote_inc = self._get_quote_increment()

        # Compute limit price:
        # - default: best_bid
        # - else: signal_price adjusted down by bps
        bps = Decimal(str(self.maker_price_bps))
        if best_bid is not None and best_bid > 0:
            raw = best_bid * (Decimal("1") - (bps / Decimal("10000")))
        else:
            raw = Decimal(str(signal_price)) * (Decimal("1") - (bps / Decimal("10000")))

        limit_price = raw
        if quote_inc is not None and quote_inc > 0:
            limit_price = self._quantize_price_down(limit_price, quote_inc)

        # Safety: do not cross spread (avoid taker). If our limit >= best_ask, step down one tick if possible.
        if best_ask is not None and best_ask > 0:
            if quote_inc is not None and quote_inc > 0 and limit_price >= best_ask:
                limit_price = self._quantize_price_down(best_ask - quote_inc, quote_inc)
            elif limit_price >= best_ask:
                # If we don't know tick size, fall back to a tiny decrement to avoid crossing
                limit_price = Decimal(str(best_ask)) * Decimal("0.999999")

        if limit_price <= 0:
            raise ValueError(f"Computed limit_price invalid: {limit_price}")

        # ---- V9.5.1 sizing: match v9.4 USD budget, then derive base_size at limit price ----
        usd_size = (Decimal(str(qty_btc)) * Decimal(str(signal_price))).quantize(
            Decimal("0.01"), rounding=ROUND_DOWN
        )
        if usd_size <= 0:
            raise ValueError(f"Computed usd_size invalid: {usd_size}")

        base_size = (usd_size / limit_price).quantize(Decimal("0.00000001"), rounding=ROUND_DOWN)
        if base_size <= 0:
            raise ValueError(f"Computed base_size invalid: {base_size}")

        print(
            f"   🧾 MAKER BUY ATTEMPT: ~${float(usd_size):.2f} => {base_size} BTC @ ${float(limit_price):.2f} "
            f"(bid={float(best_bid) if best_bid else 0.0:.2f}, ask={float(best_ask) if best_ask else 0.0:.2f}) "
            f"(client_order_id={client_order_id})"
        )

        # Optional audit line for attempt (does not mutate trade_state)
        _append_jsonl(
            LEDGER_FILE,
            {
                "ts": _now_utc_iso(),
                "side": "BUY_MAKER_ATTEMPT",
                "client_order_id": str(client_order_id),
                "product_id": PRODUCT_ID,
                "usd_budget": float(usd_size),
                "base_size": float(base_size),
                "limit_price": float(limit_price),
                "maker_timeout_s": int(self.maker_timeout_s),
                "maker_poll_s": float(self.maker_poll_s),
                "maker_price_bps": float(self.maker_price_bps),
                "best_bid": float(best_bid) if best_bid else 0.0,
                "best_ask": float(best_ask) if best_ask else 0.0,
            },
        )

        resp = self._place_post_only_limit_buy(client_order_id=client_order_id, base_size=base_size, limit_price=limit_price)

        order_id = self._extract_order_id(resp)

        if not order_id and self._response_looks_accepted(resp):
            print("   ⚠️ MAKER BUY accepted but missing order_id; reconciling via list_orders(client_order_id)...")
            order_id = self._reconcile_order_id_by_client_order_id(client_order_id, side="BUY")

        if not order_id:
            print("   ❌ MAKER BUY ERROR: No order_id returned and reconcile failed.")
            if self.maker_fallback_market:
                print("   🔁 MAKER BUY fallback → MARKET BUY (existing path).")
                return self._execute_market_buy_existing(qty_btc=qty_btc, signal_price=signal_price, client_order_id=uuid.uuid4().hex)
            return False

        # Poll for fill / partial / timeout
        status, fill_sum = self._wait_for_fill_or_timeout(
            order_id=order_id,
            timeout_s=int(self.maker_timeout_s),
            poll_s=float(self.maker_poll_s),
            allow_partial=bool(self.maker_allow_partial),
        )

        if status == "FILLED":
            # Finalize like v9.4
            entry_state = {
                "entry_timestamp": _now_utc_iso(),
                "entry_price": float(fill_sum["avg_price"]),
                "entry_fees_quote": float(fill_sum["fees_quote"]),
                "entry_fee_rate": float(fill_sum["fee_rate"]),
                "entry_filled_base": float(fill_sum["filled_base"]),
                "entry_filled_quote": float(fill_sum["filled_quote"]),
                "entry_order_id": str(order_id),
                "entry_client_order_id": str(client_order_id),
                "product_id": PRODUCT_ID,
                "execution": {
                    "entry_mode": "maker",
                    "maker_limit_price": float(limit_price),
                    "maker_timeout_s": int(self.maker_timeout_s),
                    "maker_poll_s": float(self.maker_poll_s),
                    "maker_price_bps": float(self.maker_price_bps),
                    "fallback_used": False,
                    "partial_fill": False,
                    "usd_budget": float(usd_size),
                    "base_size": float(base_size),
                },
            }

            self.save_trade_state(entry_state)

            _append_jsonl(
                LEDGER_FILE,
                {
                    "ts": _now_utc_iso(),
                    "side": "BUY",
                    "order_id": str(order_id),
                    "client_order_id": str(client_order_id),
                    "product_id": PRODUCT_ID,
                    "avg_price": float(fill_sum["avg_price"]),
                    "filled_base": float(fill_sum["filled_base"]),
                    "filled_quote": float(fill_sum["filled_quote"]),
                    "fees_quote": float(fill_sum["fees_quote"]),
                    "fee_rate": float(fill_sum["fee_rate"]),
                    "num_fills": int(fill_sum["num_fills"]),
                    "execution_mode": "maker",
                    "limit_price": float(limit_price),
                    "usd_budget": float(usd_size),
                    "base_size": float(base_size),
                },
            )

            print(
                f"   ✅ MAKER BUY FILLED | avg=${fill_sum['avg_price']:.2f} "
                f"| fees=${fill_sum['fees_quote']:.4f} ({fill_sum['fee_rate']*100:.2f}%)"
            )
            return True

        if status == "PARTIAL":
            # Safety: cancel remainder BEFORE committing state, then re-check final status & fills.
            print(f"   ⚠️ MAKER BUY PARTIAL | filled_base={fill_sum.get('filled_base', 0.0):.8f} BTC | canceling remainder...")
            self._cancel_order_best_effort(order_id)

            # Wait briefly for terminal after cancel (could become FILLED during cancel race)
            final_order = self._wait_for_terminal_order(order_id, timeout_s=15, poll_s=1.0)
            final_status = (_get(final_order, "status", "") or "").upper().strip()
            fill_sum = self._summarize_fills(order_id)  # refresh

            if fill_sum.get("filled_base", 0.0) <= 0.0:
                print(f"   ❌ MAKER BUY PARTIAL PATH: Cancelled/terminal but no fills recorded | status={final_status or 'UNKNOWN'}")
                if self.maker_fallback_market:
                    print("   🔁 MAKER BUY fallback → MARKET BUY (existing path).")
                    return self._execute_market_buy_existing(qty_btc=qty_btc, signal_price=signal_price, client_order_id=uuid.uuid4().hex)
                return False

            # Commit partial as entry (prevents duplicate entries); leave remainder unfilled.
            entry_state = {
                "entry_timestamp": _now_utc_iso(),
                "entry_price": float(fill_sum["avg_price"]),
                "entry_fees_quote": float(fill_sum["fees_quote"]),
                "entry_fee_rate": float(fill_sum["fee_rate"]),
                "entry_filled_base": float(fill_sum["filled_base"]),
                "entry_filled_quote": float(fill_sum["filled_quote"]),
                "entry_order_id": str(order_id),
                "entry_client_order_id": str(client_order_id),
                "product_id": PRODUCT_ID,
                "execution": {
                    "entry_mode": "maker",
                    "maker_limit_price": float(limit_price),
                    "maker_timeout_s": int(self.maker_timeout_s),
                    "maker_poll_s": float(self.maker_poll_s),
                    "maker_price_bps": float(self.maker_price_bps),
                    "fallback_used": False,
                    "partial_fill": True,
                    "final_order_status": final_status or "UNKNOWN",
                    "usd_budget": float(usd_size),
                    "base_size": float(base_size),
                },
            }

            self.save_trade_state(entry_state)

            _append_jsonl(
                LEDGER_FILE,
                {
                    "ts": _now_utc_iso(),
                    "side": "BUY",
                    "order_id": str(order_id),
                    "client_order_id": str(client_order_id),
                    "product_id": PRODUCT_ID,
                    "avg_price": float(fill_sum["avg_price"]),
                    "filled_base": float(fill_sum["filled_base"]),
                    "filled_quote": float(fill_sum["filled_quote"]),
                    "fees_quote": float(fill_sum["fees_quote"]),
                    "fee_rate": float(fill_sum["fee_rate"]),
                    "num_fills": int(fill_sum["num_fills"]),
                    "execution_mode": "maker",
                    "limit_price": float(limit_price),
                    "partial_fill": True,
                    "final_order_status": final_status or "UNKNOWN",
                    "usd_budget": float(usd_size),
                    "base_size": float(base_size),
                },
            )

            print(
                f"   ✅ MAKER BUY PARTIAL (ENTERED) | avg=${fill_sum['avg_price']:.2f} "
                f"| filled_base={fill_sum['filled_base']:.8f} BTC "
                f"| fees=${fill_sum['fees_quote']:.4f} ({fill_sum['fee_rate']*100:.2f}%)"
            )
            return True

        if status in {"CANCELED", "CANCELLED", "REJECTED", "EXPIRED", "FAILED"}:
            print(f"   ❌ MAKER BUY NOT FILLED | status={status} | order_id={order_id}")
            # No fill; safe to fallback if enabled
            if self.maker_fallback_market:
                print("   🔁 MAKER BUY fallback → MARKET BUY (existing path).")
                return self._execute_market_buy_existing(qty_btc=qty_btc, signal_price=signal_price, client_order_id=uuid.uuid4().hex)
            return False

        # TIMEOUT or UNKNOWN: cancel and fallback
        print(f"   ⏳ MAKER BUY TIMEOUT/NO-FILL | status={status} | canceling | order_id={order_id}")
        self._cancel_order_best_effort(order_id)

        if self.maker_fallback_market:
            print("   🔁 MAKER BUY fallback → MARKET BUY (existing path).")
            return self._execute_market_buy_existing(qty_btc=qty_btc, signal_price=signal_price, client_order_id=uuid.uuid4().hex)

        return False

    def _execute_market_buy_existing(self, qty_btc: float, signal_price: float, client_order_id: str) -> bool:
        """
        Extracted from v9.4 BUY logic so maker path can fallback without rewriting behavior.
        IMPORTANT: this is the authoritative v9.4 market buy flow (order_id reconcile + fill/fees + atomic state).
        """
        usd_size = (Decimal(str(qty_btc)) * Decimal(str(signal_price))).quantize(
            Decimal("0.01"), rounding=ROUND_DOWN
        )
        if usd_size <= 0:
            raise ValueError(f"Computed usd_size invalid: {usd_size}")

        print(f"   🚀 ROUTING BUY: ${usd_size} (client_order_id={client_order_id})")
        resp = self.client.market_order_buy(
            client_order_id=client_order_id,
            product_id=PRODUCT_ID,
            quote_size=str(usd_size),
        )

        order_id = self._extract_order_id(resp)

        # If missing, but response implies accepted, reconcile by client_order_id
        if not order_id and self._response_looks_accepted(resp):
            print("   ⚠️ BUY accepted but missing order_id; reconciling via list_orders(client_order_id)...")
            order_id = self._reconcile_order_id_by_client_order_id(client_order_id, side="BUY")

        if not order_id:
            print("   ❌ BUY ERROR: No order_id returned and reconcile failed.")
            return False

        final_order = self._wait_for_terminal_order(order_id)
        status = (_get(final_order, "status", "") or "").upper().strip()

        if status != "FILLED":
            print(f"   ❌ BUY NOT FILLED | status={status or 'UNKNOWN'} | order_id={order_id}")
            return False

        fill_sum = self._summarize_fills(order_id)

        entry_state = {
            "entry_timestamp": _now_utc_iso(),
            "entry_price": float(fill_sum["avg_price"]),
            "entry_fees_quote": float(fill_sum["fees_quote"]),
            "entry_fee_rate": float(fill_sum["fee_rate"]),
            "entry_filled_base": float(fill_sum["filled_base"]),
            "entry_filled_quote": float(fill_sum["filled_quote"]),
            "entry_order_id": str(order_id),
            "entry_client_order_id": str(client_order_id),
            "product_id": PRODUCT_ID,
            "execution": {
                "entry_mode": "market",
                "fallback_used": False,
            },
        }

        self.save_trade_state(entry_state)

        _append_jsonl(
            LEDGER_FILE,
            {
                "ts": _now_utc_iso(),
                "side": "BUY",
                "order_id": str(order_id),
                "client_order_id": str(client_order_id),
                "product_id": PRODUCT_ID,
                "avg_price": float(fill_sum["avg_price"]),
                "filled_base": float(fill_sum["filled_base"]),
                "filled_quote": float(fill_sum["filled_quote"]),
                "fees_quote": float(fill_sum["fees_quote"]),
                "fee_rate": float(fill_sum["fee_rate"]),
                "num_fills": int(fill_sum["num_fills"]),
                "execution_mode": "market",
            },
        )

        print(
            f"   ✅ BUY FILLED | avg=${fill_sum['avg_price']:.2f} "
            f"| fees=${fill_sum['fees_quote']:.4f} ({fill_sum['fee_rate']*100:.2f}%)"
        )
        return True

    # ---------------------------
    # Execution
    # ---------------------------
    def execute_trade(self, action: str, qty: float, price: float | None = None) -> bool:
        action_u = action.upper().strip()
        if action_u not in {"BUY", "SELL"}:
            raise ValueError(f"UNKNOWN action={action}")

        if qty is None or qty <= 0:
            raise ValueError(f"INVALID qty={qty}")

        client_order_id = uuid.uuid4().hex

        # ---------------------------
        # DRY-RUN PATH
        # ---------------------------
        if self.dry_run:
            try:
                if action_u == "BUY":
                    if price is None or price <= 0:
                        raise ValueError("BUY requires a valid price to compute quote_size")

                    usd_size = (Decimal(str(qty)) * Decimal(str(price))).quantize(
                        Decimal("0.01"), rounding=ROUND_DOWN
                    )

                    print(
                        f"   [DRY-RUN] Would BUY approx ${usd_size} "
                        f"({qty:.8f} BTC) at ~${float(price):.2f} "
                        f"(client_order_id={client_order_id})"
                    )

                    if self.dry_run_ledger:
                        _append_jsonl(
                            LEDGER_FILE,
                            {
                                "ts": _now_utc_iso(),
                                "mode": "DRY_RUN",
                                "side": "BUY",
                                "client_order_id": client_order_id,
                                "product_id": PRODUCT_ID,
                                "signal_price": float(price),
                                "target_quote_usd": float(usd_size),
                                "target_base_btc": float(qty),
                            },
                        )

                else:  # SELL
                    btc_size = Decimal(str(qty)).quantize(Decimal("0.00000001"), rounding=ROUND_DOWN)
                    print(
                        f"   [DRY-RUN] Would SELL {btc_size} BTC "
                        f"at ~${float(price) if price else 0.0:.2f} "
                        f"(client_order_id={client_order_id})"
                    )

                    if self.dry_run_ledger:
                        _append_jsonl(
                            LEDGER_FILE,
                            {
                                "ts": _now_utc_iso(),
                                "mode": "DRY_RUN",
                                "side": "SELL",
                                "client_order_id": client_order_id,
                                "product_id": PRODUCT_ID,
                                "target_base_btc": float(btc_size),
                                "signal_price": float(price) if price else 0.0,
                            },
                        )

                print("   [DRY-RUN] No live order sent; trade_state.json unchanged.")
                return True

            except Exception as e:
                print(f"   [DRY-RUN] ORDER SIMULATION ERROR: {e}")
                return False

        # ---------------------------
        # LIVE PATH
        # ---------------------------
        try:
            if action_u == "BUY":
                if price is None or price <= 0:
                    raise ValueError("BUY requires a valid price to compute quote_size")

                # Idempotency guard: if we already have a state file, do not double-enter.
                existing = _load_state(STATE_FILE)
                if existing is not None:
                    print("   ❌ BUY BLOCKED: trade_state.json already exists (position may already be open).")
                    return False

                # Optional maker entry path (minimal extension)
                if self.maker_entry_enabled:
                    ok = self._execute_maker_buy_with_fallback(
                        qty_btc=qty,
                        signal_price=float(price),
                        client_order_id=client_order_id,
                    )
                    return bool(ok)

                # Otherwise: v9.4 market buy behavior (unchanged), now routed through extracted helper
                return self._execute_market_buy_existing(qty_btc=qty, signal_price=float(price), client_order_id=client_order_id)

            # SELL (unchanged v9.4 behavior)
            btc_size = Decimal(str(qty)).quantize(Decimal("0.00000001"), rounding=ROUND_DOWN)
            if btc_size <= 0:
                raise ValueError(f"Computed btc_size invalid: {btc_size}")

            print(f"   🚀 ROUTING SELL: {btc_size} BTC (client_order_id={client_order_id})")
            resp = self.client.market_order_sell(
                client_order_id=client_order_id,
                product_id=PRODUCT_ID,
                base_size=str(btc_size),
            )

            order_id = self._extract_order_id(resp)

            if not order_id and self._response_looks_accepted(resp):
                print("   ⚠️ SELL accepted but missing order_id; reconciling via list_orders(client_order_id)...")
                order_id = self._reconcile_order_id_by_client_order_id(client_order_id, side="SELL")

            if not order_id:
                print("   ❌ SELL ERROR: No order_id returned and reconcile failed.")
                return False

            final_order = self._wait_for_terminal_order(order_id)
            status = (_get(final_order, "status", "") or "").upper().strip()

            if status != "FILLED":
                print(f"   ❌ SELL NOT FILLED | status={status or 'UNKNOWN'} | order_id={order_id}")
                return False

            fill_sum = self._summarize_fills(order_id)

            _append_jsonl(
                LEDGER_FILE,
                {
                    "ts": _now_utc_iso(),
                    "side": "SELL",
                    "order_id": str(order_id),
                    "client_order_id": str(client_order_id),
                    "product_id": PRODUCT_ID,
                    "avg_price": float(fill_sum["avg_price"]),
                    "filled_base": float(fill_sum["filled_base"]),
                    "filled_quote": float(fill_sum["filled_quote"]),
                    "fees_quote": float(fill_sum["fees_quote"]),
                    "fee_rate": float(fill_sum["fee_rate"]),
                    "num_fills": int(fill_sum["num_fills"]),
                },
            )

            self.clear_trade_state()

            print(
                f"   ✅ SELL FILLED | avg=${fill_sum['avg_price']:.2f} "
                f"| fees=${fill_sum['fees_quote']:.4f} ({fill_sum['fee_rate']*100:.2f}%)"
            )
            return True

        except Exception as e:
            print(f"   ❌ ORDER ERROR: {e}")
            return False
