# src/signal_generator.py
from __future__ import annotations

import sys
# FORCE UTF-8 OUTPUT TO PREVENT SUBPROCESS CRASHES ON WINDOWS
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')

import os
import json
import logging
import pandas as pd
import types
import importlib.util
import requests
import time
from pathlib import Path
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, Optional, List

# --- BOOTSTRAPPER ---
current_dir = Path(__file__).resolve().parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

if "src" not in sys.modules:
    src_pkg = types.ModuleType("src")
    src_pkg.__path__ = [str(current_dir)] 
    sys.modules["src"] = src_pkg

for file_path in current_dir.glob("*.py"):
    module_name = file_path.stem 
    if module_name == "__init__" or module_name == "signal_generator": continue
    full_name = f"src.{module_name}"
    if full_name not in sys.modules:
        try:
            spec = importlib.util.spec_from_file_location(module_name, file_path)
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                sys.modules[full_name] = module
                sys.modules[module_name] = module 
                spec.loader.exec_module(module)
                setattr(src_pkg, module_name, module)
        except Exception:
            pass 

# --- IMPORTS (Fixed Syntax) ---
try: 
    from src.signal_filter import is_signal_valid
except: 
    def is_signal_valid(s): return True

try: 
    from src.cache_instance import cache
except: 
    cache = None

try: 
    from src.sentiment_blended import blend_sentiment_scores
except: 
    def blend_sentiment_scores(): return {}

try: 
    from src.dispatcher import dispatch_alerts
except: 
    def dispatch_alerts(a, s, c): pass

try: 
    from src.jsonl_writer import atomic_jsonl_append
except: 
    def atomic_jsonl_append(p, d): pass

try: 
    from src.observability import failure_tracker
except: 
    class MockTracker:
        def record_failure(self, *args): pass
    failure_tracker = MockTracker()

try: 
    from src.paths import SHADOW_LOG_PATH, GOVERNANCE_PARAMS_PATH
except:
    SHADOW_LOG_PATH = Path("shadow_log.jsonl")
    GOVERNANCE_PARAMS_PATH = Path("governance_params.json")

# --- BROKER IMPORT ---
try:
    from src.paper_broker import PaperBroker
except ImportError:
    try:
        from paper_broker import PaperBroker
    except:
        logging.warning("PAPER BROKER NOT FOUND. Trading disabled.")
        PaperBroker = None

try:
    from src.regime_detector import MarketRegimeDetector, MetaStrategySelector, PositionSizer
except ImportError:
    try:
        from regime_detector import MarketRegimeDetector, MetaStrategySelector, PositionSizer
    except ImportError:
        logging.warning("CRITICAL: Regime Detector not found. Using Dummy Fallbacks.")
        class MarketRegimeDetector:
            def detect_regime(self, df): return "WARMUP"
        class MetaStrategySelector:
            def get_strategy(self, regime): return "Conservative_Trend_Follow"
        class PositionSizer:
            def __init__(self, risk_percent=0.1): pass
            def calculate_size(self, balance): return 1000.0

logger = logging.getLogger(__name__)

_ML_INFER_FN = None
try:
    from signal_engine.ml.infer import infer_asset_signal as _ML_INFER_FN
except ImportError:
    try:
        from src.ml.infer import infer_asset_signal as _ML_INFER_FN
    except ImportError:
        _ML_INFER_FN = None

def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def _read_json(path: Path, default: Any) -> Any:
    try:
        if path.exists(): return json.loads(path.read_text(encoding="utf-8"))
    except Exception: pass
    return default

def load_governance_params(symbol: str) -> Dict[str, Any]:
    default = {"conf_min": 0.55, "debounce_min": 5}
    try: path = GOVERNANCE_PARAMS_PATH
    except NameError: path = Path("governance_params.json")
    data = _read_json(path, {})
    row = data.get(symbol) or {}
    return {
        "conf_min": float(row.get("conf_min", default["conf_min"])),
        "debounce_min": int(row.get("debounce_min", default["debounce_min"])),
    }

def _fetch_live_coingecko_data(asset: str = "bitcoin") -> Dict[str, Any]:
    try:
        url = "https://api.coingecko.com/api/v3/simple/price"
        params = {
            'ids': asset.lower(), 
            'vs_currencies': 'usd',
            'include_24hr_vol': 'true',
            'include_24hr_change': 'true'
        }
        resp = requests.get(url, params=params, timeout=10)
        data = resp.json()
        
        if asset.lower() in data:
            d = data[asset.lower()]
            return {
                "price": float(d.get('usd', 0)),
                "volume_24h": float(d.get('usd_24h_vol', 0)),
                "price_change_24h": float(d.get('usd_24h_change', 0))
            }
    except Exception as e:
        print(f"!! COINGECKO API ERROR: {e}")
    return {}

def _fetch_asset_history(asset: str) -> Optional[pd.DataFrame]:
    try:
        possible_paths = [Path('flight_recorder.csv'), Path('../flight_recorder.csv'), Path(os.getcwd()) / 'flight_recorder.csv']
        csv_path = None
        for p in possible_paths:
            if p.exists():
                csv_path = p
                break
        if csv_path:
            df = pd.read_csv(csv_path)
            if 'Timestamp' in df.columns and 'Price' in df.columns:
                df['Timestamp'] = pd.to_datetime(df['Timestamp'])
                return df
    except Exception as e:
        logger.warning(f"Failed to fetch history for {asset}: {e}")
    return None

def _infer_ml(asset: str, strategy: str = None) -> Dict[str, Any]:
    ml_enabled = str(os.getenv("MW_INFER_ENABLE", "1")).lower() in {"1", "true", "yes"}
    if not ml_enabled: return {"ok": False, "reason": "ml_disabled"}
    if _ML_INFER_FN is None: return {"ok": False, "reason": "ml_unavailable"}

    try:
        # CRITICAL CHANGE: The infer_asset_signal function now returns a dict with model_name
        out = _ML_INFER_FN(asset, strategy=strategy) 
        if not isinstance(out, dict): return {"ok": False, "reason": "ml_bad_return_type"}
        if out.get("error"): return {"ok": False, "reason": f"ml_error:{out['error']}"}
        
        direction = out.get("direction") or out.get("dir")
        conf = out.get("confidence") if out.get("confidence") is not None else out.get("conf")
        
        # New key capture
        model_name = out.get("model_name", "Unknown Model") # <-- CRITICAL CAPTURE

        if (direction is not None and direction not in {"long", "short"}) or conf is None:
            return {"ok": False, "reason": "ml_missing_keys", "raw": out}
            
        return {"ok": True, "dir": direction, "conf": float(conf), "reason": "ok", "raw": out, "model_name": model_name} # <-- CRITICAL RETURN ADDITION
    except Exception as e:
        return {"ok": False, "reason": f"ml_exception:{type(e).__name__}"}

def label_confidence(score: float) -> str:
    if score >= 0.66: return "High Confidence"
    elif score >= 0.33: return "Medium Confidence"
    else: return "Low Confidence"

# --- PERSISTENT COOLDOWN CHECK ---
def check_broker_cooldown(broker, minutes=10):
    if not broker.trade_log:
        return True
    
    last_trade = broker.trade_log[-1]
    last_ts_str = str(last_trade.get("ts", ""))
    try:
        if "T" in last_ts_str:
            last_dt = datetime.fromisoformat(last_ts_str)
        else:
            # Handle standard string conversion format
            last_dt = datetime.strptime(last_ts_str, "%Y-%m-%d %H:%M:%S.%f")
            
        elapsed_min = (datetime.now() - last_dt).total_seconds() / 60
        if elapsed_min < minutes:
            return False
    except:
        return True
    return True

# --- SINGLETONS ---
_regime_detector = MarketRegimeDetector()
_strategy_selector = MetaStrategySelector()
_broker = PaperBroker() if PaperBroker else None 

def generate_signals():
    print(f"[{datetime.now().time()}] Starting Signal Generator (SMART SIZING + COOLDOWN)...")
    stablecoins = {"USDC", "USDT", "DAI", "TUSD", "BUSD"}
    valid_signals: List[dict] = []
    
    live_env = os.getenv("MW_INFER_LIVE", "1")
    live_ml = str(live_env).lower() in {"1", "true", "yes"}

    try:
        assets = []
        if 'cache' in globals() and cache:
            try: assets = [k for k in cache.keys() if not k.endswith('_signals') and not k.endswith('_sentiment')]
            except: pass 
        
        if not assets:
            assets = ["BTC"]
        
        sentiment_scores = blend_sentiment_scores() if 'blend_sentiment_scores' in globals() else {}
        
        for asset in assets:
            if asset in stablecoins: continue
            
            data = {}
            if 'cache' in globals() and cache:
                data = cache.get_signal(asset) or {}
                
            if not data.get("volume_now") or not data.get("price_change_24h"):
                live_data = _fetch_live_coingecko_data("bitcoin" if asset == "BTC" else asset)
                if live_data:
                    data["volume_now"] = live_data["volume_24h"]
                    data["price_change_24h"] = live_data["price_change_24h"]
                    data["price"] = live_data["price"]
                else:
                    print(f"[{asset}] API Failed. Skipping.")
                    continue

            hist_df = _fetch_asset_history(asset)
            current_regime = _regime_detector.detect_regime(hist_df) if hist_df is not None and not hist_df.empty else "WARMUP"
            target_strategy = _strategy_selector.get_strategy(current_regime)
            
            # --- GET REAL WALLET DATA ---
            available_cash = 10000.0
            current_pos_val = 0.0
            total_equity = 10000.0
            
            if _broker:
                available_cash = _broker.cash
                current_price = float(data.get("price", 0.0))
                current_pos_val = _broker.positions * current_price
                total_equity = _broker.get_portfolio_value(current_price)

            print(f"[{asset}] Regime: {current_regime} -> Strategy: {target_strategy}")

            price_change = float(data.get("price_change_24h", 0.0))
            volume = float(data.get("volume_now", 0.0))
            current_price = float(data.get("price", 0.0))
            sentiment = float(sentiment_scores.get(asset, 0.0))

            ml = _infer_ml(asset, strategy=target_strategy)
            
            if not ml.get("ok"):
                print(f"   >> ML FAILED: {ml.get('reason')}")
            else:
                d_str = ml['dir'].upper() if ml['dir'] else "HOLD"
                model_name = ml.get("model_name", "N/A") # <-- CAPTURE MODEL NAME
                
                # CRITICAL CHANGE: Updated print statement to include the algorithm name
                print(f"   >> [ML PREDICTION] {d_str} ({ml['conf']:.2f}) using {target_strategy} ({model_name})") 

            gov = load_governance_params(asset)
            
            # --- DECIDE & EXECUTE ---
            if live_ml and ml.get("ok"):
                direction = ml["dir"]
                confidence = float(ml["conf"] or 0.0)
                
                if direction is None:
                    print("   >> ML Decision: HOLD/CASH (No Signal Generated)")
                    continue

                if confidence < float(gov["conf_min"]): 
                    print(f"   >> ML Filtered: Confidence {confidence:.2f} < Min {gov['conf_min']}")
                    continue
                
                # --- SIZING LOGIC ---
                exposure_pct = current_pos_val / total_equity if total_equity > 0 else 0
                MAX_ALLOCATION = 0.99 
                ENTRY_PERCENT_OF_CASH = 0.20
                trade_dollars = available_cash * ENTRY_PERCENT_OF_CASH
                MIN_TRADE_DOLLARS = 20.0 

                signal = {
                    "asset": asset,
                    "direction": direction,
                    "confidence_score": confidence,
                    "confidence_label": label_confidence(confidence),
                    "regime": current_regime, 
                    "strategy": target_strategy, 
                    "model_name": ml.get("model_name", "N/A"), # <-- ADD MODEL NAME TO SIGNAL DICT
                    "trade_size_limit": trade_dollars, 
                    "price_change": price_change,
                    "volume": volume,
                    "sentiment": sentiment,
                    "timestamp": datetime.now(timezone.utc),
                    "governance": gov,
                    "inference": "ml_live"
                }
                
                try:
                    is_valid = is_signal_valid(signal)
                except Exception as e:
                    print(f"   >> CRITICAL: Signal Filter Crashed: {e}. Defaulting to True.")
                    is_valid = True

                if is_valid:
                    dispatch_alerts(asset, signal, cache)
                    valid_signals.append(signal)
                    print(f"   >> [LIVE SIGNAL GENERATED] {direction}")
                    
                    if _broker and current_price > 0:
                        action = "BUY" if direction.lower() == "long" else "SELL"
                        
                        if action == "BUY":
                            # CHECK COOLDOWN (10 mins)
                            can_trade = check_broker_cooldown(_broker, minutes=10)
                            
                            if not can_trade:
                                print(f"   >> [SKIPPED] Cooldown Active (Last trade < 10 mins ago)")
                            elif exposure_pct >= MAX_ALLOCATION:
                                print(f"   >> [SKIPPED] Max Allocation Reached ({exposure_pct*100:.1f}%)")
                            elif trade_dollars < MIN_TRADE_DOLLARS:
                                print(f"   >> [SKIPPED] Insufficient Cash for Min Trade (${available_cash:.2f})")
                            else:
                                qty = trade_dollars / current_price
                                print(f"   >> [EXECUTING] {action} {qty:.6f} {asset} (${trade_dollars:.2f})")
                                executed = _broker.execute_trade(action, qty, current_price)
                                if executed:
                                    equity = _broker.get_portfolio_value(current_price)
                                    print(f"   >> [PORTFOLIO] Cash: ${round(_broker.cash, 2)} | Equity: ${round(equity, 2)}")
                        
                        elif action == "SELL":
                            if _broker.positions > 0:
                                qty = _broker.positions
                                print(f"   >> [EXECUTING] {action} {qty:.6f} {asset} (CLOSE POSITION)")
                                executed = _broker.execute_trade(action, qty, current_price)
                                if executed:
                                    equity = _broker.get_portfolio_value(current_price)
                                    print(f"   >> [PORTFOLIO] Cash: ${round(_broker.cash, 2)} | Equity: ${round(equity, 2)}")
                            else:
                                print("   >> [SKIPPED] No position to sell.")

                else:
                    print(f"   >> [SIGNAL REJECTED] Filter Blocked (Vol: {volume}, Change: {price_change:.2f}%)")

    except Exception as e:
        print(f"Generator Exception: {e}")
        failure_tracker.record_failure("signal_generator", str(e))

    return valid_signals

if __name__ == "__main__":
    generate_signals()
    print("Signal Generation Complete.")