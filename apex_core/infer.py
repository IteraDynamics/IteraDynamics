# apex_core/infer.py
from __future__ import annotations
import json
import os
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple
import joblib
import numpy as np
from datetime import datetime, timedelta, timezone

from apex_core.paths import (
    MODELS_DIR,
    RETRAINING_LOG_PATH,
    RETRAINING_TRIGGERED_LOG_PATH
)
from apex_core.jsonl_writer import atomic_jsonl_append
from apex_core.signal_engine.analytics.origin_utils import normalize_origin as _norm

logger = logging.getLogger(__name__)

# Module-level paths that tests need to override via env vars
# These are defined HERE (not in src.paths) so that importlib.reload()
# will re-evaluate them with updated env vars
_TRIGGER_LOG_PATH = Path(os.getenv("TRIGGER_LOG_PATH", str(MODELS_DIR / "trigger_history.jsonl")))
_TRAINING_VERSION_FILE = Path(os.getenv("TRAINING_VERSION_FILE", str(MODELS_DIR / "training_version.txt")))

# Filenames (legacy trigger-likelihood models retained)
_LOGI_MODEL = "trigger_likelihood_v0.joblib"
_LOGI_META  = "trigger_likelihood_v0.meta.json"
_COV_NAME   = "feature_coverage.json"

_RF_MODEL   = "trigger_likelihood_rf.joblib"
_RF_META    = "trigger_likelihood_rf.meta.json"

_GB_MODEL   = "trigger_likelihood_gb.joblib"
_GB_META    = "trigger_likelihood_gb.meta.json"

# ---------- tiny utils ----------
def _read_model_version() -> str:
    try:
        if _TRAINING_VERSION_FILE.exists():
            return _TRAINING_VERSION_FILE.read_text(encoding="utf-8").strip() or "unknown"
    except Exception:
        pass
    return "unknown"

def _artifact_paths(model_name: str, meta_name: str, models_dir: Path | None = None) -> Tuple[Path, Path]:
    md = models_dir or MODELS_DIR
    return md / model_name, md / meta_name

def _load_model_and_meta(model_name: str, meta_name: str, models_dir: Path | None = None):
    mpath, jpath = _artifact_paths(model_name, meta_name, models_dir)
    model = joblib.load(mpath)
    with jpath.open("r", encoding="utf-8") as f:
        meta = json.load(f)
    return model, meta

def _load_cov(models_dir: Path | None = None) -> Dict[str, Any]:
    cpath = (models_dir or MODELS_DIR) / _COV_NAME
    if not cpath.exists():
        return {}
    try:
        with cpath.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

# ---------- public metadata helpers (legacy) ----------
def model_metadata(models_dir: Path | None = None) -> Dict[str, Any]:
    """Logistic metadata (+ coverage merged) for back-compat."""
    try:
        _, meta = _load_model_and_meta(_LOGI_MODEL, _LOGI_META, models_dir)
        cov = _load_cov(models_dir)
        out = dict(meta)
        if cov:
            out["feature_coverage"] = cov
        return out
    except Exception:
        return {}

def model_metadata_all(models_dir: Path | None = None) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    # logistic
    L = model_metadata(models_dir)
    if L:
        out["logistic"] = L
    # rf
    try:
        _, m = _load_model_and_meta(_RF_MODEL, _RF_META, models_dir)
        out["rf"] = m
    except Exception:
        pass
    # gb
    try:
        _, m = _load_model_and_meta(_GB_MODEL, _GB_META, models_dir)
        out["gb"] = m
    except Exception:
        pass
    return out

# ---------- scoring (legacy trigger-likelihood) ----------
def _vectorize(features: Dict[str, Any], feat_order: List[str]) -> np.ndarray:
    return np.array([[float(features.get(k, 0.0) or 0.0) for k in feat_order]], dtype=float)

def _contributions_linear(model, xrow: np.ndarray, feat_order: List[str], top_n: int | None) -> Dict[str, float]:
    try:
        coef = model.coef_.ravel()
        contrib = {feat_order[i]: float(coef[i] * xrow[0, i]) for i in range(len(feat_order))}
        if top_n is not None:
            return dict(sorted(contrib.items(), key=lambda kv: abs(kv[1]), reverse=True)[:top_n])
        return contrib
    except Exception:
        return {}

def infer_score(payload: Dict[str, Any], *, explain: bool = False, top_n: int = 5, models_dir: Path | None = None) -> Dict[str, Any]:
    try:
        model, meta = _load_model_and_meta(_LOGI_MODEL, _LOGI_META, models_dir)
    except Exception:
        if payload.get("features"):
            bz = float(payload["features"].get("burst_z", 0.0))
            p = 1 / (1 + np.exp(-0.1 * bz))
            res = {"prob_trigger_next_6h": float(p), "demo": True}
            if explain:
                res["contributions"] = {"burst_z": 0.1 * bz}
            return res
        return {"prob_trigger_next_6h": 0.062, "demo": True}

    feats = payload.get("features")
    if feats is None:
        return {"prob_trigger_next_6h": 0.062, "note": "origin path not wired", "demo": meta.get("demo", False)}

    feat_order = meta.get("feature_order") or []
    x = _vectorize(feats, feat_order)
    proba = float(model.predict_proba(x)[0, 1])
    out = {"prob_trigger_next_6h": proba}
    if explain:
        out["contributions"] = _contributions_linear(model, x, feat_order, top_n=top_n)
    return out

def infer_score_ensemble(payload: Dict[str, Any], *, models_dir: Path | None = None) -> Dict[str, Any]:
    votes: Dict[str, float] = {}
    demo = False
    feats: Dict[str, Any] = payload.get("features") or {}
    try:
        L_model, L_meta = _load_model_and_meta(_LOGI_MODEL, _LOGI_META, models_dir)
        order = L_meta.get("feature_order") or []
        votes["logistic"] = float(L_model.predict_proba(_vectorize(feats, order))[0, 1])
    except Exception:
        pass
    try:
        RF_model, RF_meta = _load_model_and_meta(_RF_MODEL, _RF_META, models_dir)
        order = RF_meta.get("feature_order") or []
        votes["rf"] = float(RF_model.predict_proba(_vectorize(feats, order))[0, 1])
    except Exception:
        pass
    try:
        GB_model, GB_meta = _load_model_and_meta(_GB_MODEL, _GB_META, models_dir)
        order = GB_meta.get("feature_order") or []
        votes["gb"] = float(GB_model.predict_proba(_vectorize(feats, order))[0, 1])
    except Exception:
        pass
    if not votes:
        bz = float(feats.get("burst_z", 0.0))
        p_demo = 1 / (1 + np.exp(-0.08 * bz))
        votes["logistic"] = p_demo
        demo = True
    probs = list(votes.values())
    mean = float(np.mean(probs))
    low = float(min(probs))
    high = float(max(probs))
    res: Dict[str, Any] = {
        "prob_trigger_next_6h": mean,
        "low": low,
        "high": high,
        "votes": votes,
        "models": list(votes.keys()),
        "demo": demo,
    }
    # logging (never throws)
    try:
        origin_for_log = (
            payload.get("origin")
            or feats.get("_origin")
            or feats.get("origin")
            or payload.get("source")
            or os.getenv("TL_LOG_FALLBACK_ORIGIN", "unknown")
        )
        origin_for_log = str(origin_for_log).strip().lower() if origin_for_log is not None else "unknown"
        exp = res.get("explanation") or payload.get("explanation") or {}
        adjusted_score = float(exp.get("adjusted_score", res.get("prob_trigger_next_6h", 0.0)) or 0.0)
        threshold_used = exp.get("threshold_after_volatility")
        decision = exp.get("decision", "not_triggered" if adjusted_score < (threshold_used or 1.0) else "triggered")
        regime = exp.get("volatility_regime")
        drifted = exp.get("drifted_features", [])
        top_feats = exp.get("top_contributors", [])
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "origin": origin_for_log,
            "adjusted_score": adjusted_score,
            "threshold": threshold_used,
            "decision": decision,
            "volatility_regime": regime,
            "drifted_features": drifted,
            "top_contributors": top_feats,
            "model_version": _read_model_version(),
        }
        atomic_jsonl_append(_TRIGGER_LOG_PATH, entry)
    except Exception as e:
        logger.warning(f"Failed to log trigger inference: {e}")
    return res

# ---------- Volatility-aware thresholds ----------
def compute_volatility_adjusted_threshold(base_threshold: float, regime: str) -> Dict[str, Any]:
    try:
        mults = {
            "calm": float(os.getenv("TL_REGIME_THRESH_MULT_CALM", "0.9")),
            "normal": float(os.getenv("TL_REGIME_THRESH_MULT_NORMAL", "1.0")),
            "turbulent": float(os.getenv("TL_REGIME_THRESH_MULT_TURBULENT", "1.1")),
        }
        multiplier = mults.get(str(regime).strip().lower(), 1.0)
    except Exception:
        multiplier = 1.0
    adjusted = base_threshold * multiplier
    return {
        "base_threshold": base_threshold,
        "volatility_regime": regime,
        "regime_multiplier": multiplier,
        "threshold_after_volatility": adjusted,
    }

# Back-compat alias
def score(payload: dict, explain: bool = False):
    return infer_score(payload, explain=explain)

__all__ = [
    "infer_score", "infer_score_ensemble", "score",
    "model_metadata", "model_metadata_all",
    "compute_volatility_adjusted_threshold",
]

# ---------- Online backtest (legacy utility) ----------
def _load_jsonl(path: Path) -> List[dict]:
    try:
        return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
    except Exception:
        return []

def _parse_ts(v):
    try:
        return datetime.fromtimestamp(float(v), tz=timezone.utc)
    except Exception:
        try:
            s = str(v); s = s[:-1] + "+00:00" if s.endswith("Z") else s
            return datetime.fromisoformat(s).astimezone(timezone.utc)
        except Exception:
            return None

def _label_has_trigger_between(triggers, origin: str, t0: datetime, t1: datetime) -> int:
    o = _norm(origin)
    for r in triggers:
        if _norm(r.get("origin","")) != o: continue
        ts = _parse_ts(r.get("timestamp"))
        if ts and t0 < ts <= t1:
            return 1
    return 0

def live_backtest_last_24h(interval: str = "hour", threshold: float = 0.5) -> Dict[str, Any]:
    now = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    buckets = [now - timedelta(hours=i) for i in range(24, 0, -1)]
    flags = _load_jsonl(RETRAINING_LOG_PATH)
    origins = sorted({_norm(r.get("origin","unknown")) for r in flags if _parse_ts(r.get("timestamp")) and _parse_ts(r.get("timestamp")) >= now - timedelta(hours=24)}) or ["twitter","reddit","rss_news"]
    triggers = _load_jsonl(RETRAINING_TRIGGERED_LOG_PATH)

    per_origin = []
    for o in origins[:10]:
        tp=fp=fn=tn=0
        for t in buckets:
            t_iso = t.isoformat()
            try:
                p = score({"origin": o, "timestamp": t_iso}).get("prob_trigger_next_6h", 0.0)
            except Exception:
                p = 0.0
            y = _label_has_trigger_between(triggers, o, t, t + timedelta(hours=6))
            yhat = 1 if p >= threshold else 0
            if   yhat==1 and y==1: tp+=1
            elif yhat==1 and y==0: fp+=1
            elif yhat==0 and y==1: fn+=1
            else: tn+=1
        prec = tp/float(tp+fp) if (tp+fp)>0 else 0.0
        rec  = tp/float(tp+fn) if (tp+fn)>0 else 0.0
        per_origin.append({"origin": o, "precision": round(prec,3), "recall": round(rec,3), "tp":tp,"fp":fp,"fn":fn,"tn":tn})

    tp=sum(po["tp"] for po in per_origin); fp=sum(po["fp"] for po in per_origin)
    fn=sum(po["fn"] for po in per_origin); tn=sum(po["tn"] for po in per_origin)
    prec = tp/float(tp+fp) if (tp+fp)>0 else 0.0
    rec  = tp/float(tp+fn) if (tp+fn)>0 else 0.0

    return {
        "window_hours": 24,
        "threshold": threshold,
        "overall": {"precision": round(prec,3), "recall": round(rec,3), "tp":tp,"fp":fp,"fn":fn,"tn":tn},
        "per_origin": per_origin[:3],
    }

# ==== MoonWire asset inference bridge (models/current per-symbol) ===========
from pathlib import Path as _Path
import json as _json
import joblib as _joblib
import numpy as _np

def _load_current_bundle(models_dir: _Path | None = None, symbol: str | None = None):
    """
    Load per-symbol bundle if present:
      models/current/{SYMBOL}/{manifest.json,features.json,model.joblib}
    Fallbacks:
      models/current/default/...
      models/current/...
    Returns (model, feature_order, manifest_dict)
    """
    root = _Path("models/current") if models_dir is None else _Path(models_dir)

    search = []
    if symbol:
        search.append(root / str(symbol).upper())
    search.append(root / "default")
    search.append(root)  # legacy

    for base in search:
        man_p = base / "manifest.json"
        feat_p = base / "features.json"
        mdl_p = base / "model.joblib"
        if man_p.exists() and mdl_p.exists():
            manifest = _json.loads(man_p.read_text(encoding="utf-8"))
            feats = []
            if feat_p.exists():
                fj = _json.loads(feat_p.read_text(encoding="utf-8"))
                feats = fj.get("feature_order") or fj.get("features") or []
            if not feats:
                feats = manifest.get("feature_order") or manifest.get("features") or []
            if not feats:
                raise RuntimeError("feature list missing in bundle")
            model = _joblib.load(mdl_p)
            return model, [str(f) for f in feats], manifest

    raise FileNotFoundError(f"inference bundle not found for symbol={symbol} under {root}")

def _build_latest_features(symbol: str, manifest: dict) -> dict:
    """
    Recompute the latest feature row for `symbol` so inference matches training.
    """
    from scripts.ml.data_loader import load_prices
    from scripts.ml.feature_builder import build_features
    lookback = int(manifest.get("lookback_days", 180) or 180)
    prices = load_prices([symbol], lookback_days=lookback)
    feats_map = build_features(prices)
    df = feats_map[symbol]
    if df is None or len(df) == 0:
        raise RuntimeError("no features built for symbol")
    return df.iloc[-1].to_dict()

def infer_asset_signal(symbol: str, *, models_dir: _Path | None = None) -> dict:
    """
    Returns:
      {
        "symbol": "BTC",
        "y_pred": 0|1,
        "direction": "long"|"short",
        "confidence": float,  # probability of long
        "p_long": float,
        "feature_order": [...],
        "model_type": "...",
      }
    """
    model, feat_order, manifest = _load_current_bundle(models_dir, symbol=symbol)
    row = _build_latest_features(symbol, manifest)
    x = _np.array([[float(row.get(k, 0.0) or 0.0) for k in feat_order]], dtype=float)

    if hasattr(model, "predict_proba"):
        p_long = float(model.predict_proba(x)[0, 1])
    elif hasattr(model, "decision_function"):
        z = float(model.decision_function(x)[0])
        p_long = 1.0 / (1.0 + _np.exp(-z))
    else:
        raise RuntimeError("model has neither predict_proba nor decision_function")

    y_pred = 1 if p_long >= 0.5 else 0
    return {
        "symbol": symbol,
        "y_pred": y_pred,
        "direction": "long" if y_pred == 1 else "short",
        "confidence": p_long,
        "p_long": p_long,
        "feature_order": feat_order,
        "model_type": manifest.get("model_type", "unknown"),
    }

try:
    __all__.append("infer_asset_signal")  # type: ignore
except Exception:
    __all__ = [*globals().get("__all__", []), "infer_asset_signal"]
# ===========================================================================