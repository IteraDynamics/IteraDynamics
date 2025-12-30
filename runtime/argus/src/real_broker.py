# src/real_broker.py
# 🦅 ARGUS REAL BROKER - V9.0
# (ATOMIC STATE + IDEMPOTENCY + TZ-AWARE UTC + DRY-RUN SUPPORT)

import os
import json
import uuid
from decimal import Decimal, ROUND_DOWN
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Tuple

from dotenv import load_dotenv
from coinbase.rest import RESTClient

load_dotenv()

HARDCODED_UUID = "5bce9ffb-611c-4dcb-9e18-75d3914825a1"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
STATE_FILE = PROJECT_ROOT / "trade_state.json"
STATE_TMP = PROJECT_ROOT / "trade_state.json.tmp"

PRODUCT_ID = "BTC-USD"


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


class RealBroker:
    """
    Live Coinbase broker wrapper.

    Key behaviors:
    - Uses HARDCODED_UUID portfolio.
    - Persists trade_state.json on LIVE BUY / SELL for Argus guardrails.
    - Respects ARGUS_DRY_RUN env:
        * When True:
            - No live orders are sent.
            - trade_state.json is NOT mutated.
            - Wallet snapshot still uses real Coinbase balances.
    """

    def __init__(self):
        self.api_key = os.getenv("COINBASE_API_KEY") or os.getenv("CB_API_KEY")
        self.api_secret = os.getenv("COINBASE_API_SECRET") or os.getenv("CB_API_SECRET")

        if not self.api_key or not self.api_secret:
            raise ValueError("❌ MISSING API KEYS in .env")

        # Allow "\n" literals for multiline secrets
        self.api_secret = self.api_secret.replace("\\n", "\n")

        # DRY-RUN toggle (default: LIVE)
        self.dry_run = _env_bool("ARGUS_DRY_RUN", default=False)

        # Cache for dashboard properties
        self._last_cash: float = 0.0
        self._last_btc: float = 0.0

        try:
            self.client = RESTClient(api_key=self.api_key, api_secret=self.api_secret)
            mode_str = "DRY-RUN (no live orders)" if self.dry_run else "LIVE"
            print(f"🔌 RealBroker: Connected. TARGETING UUID: {HARDCODED_UUID} | MODE: {mode_str}")
        except Exception as e:
            print(f"❌ CONNECTION ERROR: {e}")
            raise

    # ---------------------------
    # Persistence (atomic)
    # ---------------------------

    def _atomic_write_json(self, path: Path, tmp_path: Path, payload: dict) -> None:
        tmp_path.parent.mkdir(parents=True, exist_ok=True)
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, separators=(",", ":"), sort_keys=True)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)

    def save_trade_state(self, price: float) -> None:
        """
        Saves entry data so the Signal Generator can perform SELL guardrails later.
        Stored as tz-aware UTC ISO8601 (+00:00).

        Only used in LIVE mode; DRY-RUN paths do not mutate state.
        """
        state = {
            "entry_timestamp": datetime.now(timezone.utc).isoformat(),
            "entry_price": float(price),
        }
        self._atomic_write_json(STATE_FILE, STATE_TMP, state)
        print(f"💾 Trade state saved: Entry at ${float(price):.2f}")

    def clear_trade_state(self) -> None:
        """
        Clears memory after a successful exit.

        Only used in LIVE mode; DRY-RUN paths do not mutate state.
        """
        try:
            if STATE_FILE.exists():
                STATE_FILE.unlink()
                print("🗑️ Trade state cleared.")
        except Exception as e:
            print(f"❌ FAILED TO CLEAR TRADE STATE: {e}")
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
        # Do not swallow errors; caller decides fail-closed/open policy.
        return self.client.get_accounts(limit=250, portfolio_id=HARDCODED_UUID)

    def get_wallet_snapshot(self) -> Tuple[float, float]:
        """
        Returns (cash_usd, btc_units) from live exchange balances.
        Raises on API/schema failures.
        """
        response = self._get_accounts()
        cash = 0.0
        btc = 0.0
        for acc in getattr(response, "accounts", []):
            ccy = getattr(acc, "currency", "")
            if ccy == "USD":
                cash = self._get_value(getattr(acc, "available_balance", None))
            elif ccy == "BTC":
                btc = self._get_value(getattr(acc, "available_balance", None))

        # Cache for dashboard / properties
        self._last_cash = cash
        self._last_btc = btc

        return cash, btc

    @property
    def cash(self) -> float:
        """
        Last observed USD cash from get_wallet_snapshot().
        """
        return self._last_cash

    @property
    def positions(self) -> float:
        """
        Last observed BTC units from get_wallet_snapshot().
        """
        return self._last_btc

    # ---------------------------
    # Execution
    # ---------------------------

    def execute_trade(self, action: str, qty: float, price: float | None = None) -> bool:
        """
        action: BUY or SELL
        qty: BTC units
        price: last close used to compute quote_size for BUY

        DRY-RUN behavior:
        - Logs what would be done.
        - Does NOT send live orders.
        - Does NOT touch trade_state.json.
        """
        action_u = action.upper().strip()

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

                elif action_u == "SELL":
                    btc_size = Decimal(str(qty)).quantize(Decimal("0.00000001"), rounding=ROUND_DOWN)
                    print(
                        f"   [DRY-RUN] Would SELL {btc_size} BTC "
                        f"at ~${float(price) if price else 0.0:.2f} "
                        f"(client_order_id={client_order_id})"
                    )

                else:
                    raise ValueError(f"UNKNOWN action={action}")

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

                usd_size = (Decimal(str(qty)) * Decimal(str(price))).quantize(
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

                if getattr(resp, "success", False) or getattr(resp, "order_id", None):
                    self.save_trade_state(price)
                    print("   ✅ BUY ORDER ACCEPTED.")
                    return True

            elif action_u == "SELL":
                btc_size = Decimal(str(qty)).quantize(Decimal("0.00000001"), rounding=ROUND_DOWN)
                if btc_size <= 0:
                    raise ValueError(f"Computed btc_size invalid: {btc_size}")

                print(f"   🚀 ROUTING SELL: {btc_size} BTC (client_order_id={client_order_id})")
                resp = self.client.market_order_sell(
                    client_order_id=client_order_id,
                    product_id=PRODUCT_ID,
                    base_size=str(btc_size),
                )

                if getattr(resp, "success", False) or getattr(resp, "order_id", None):
                    self.clear_trade_state()
                    print("   ✅ SELL ORDER ACCEPTED.")
                    return True

            else:
                raise ValueError(f"UNKNOWN action={action}")

        except Exception as e:
            print(f"   ❌ ORDER ERROR: {e}")

        return False
