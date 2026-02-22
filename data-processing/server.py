"""
HeartScape Flask server.

Run from the data-processing/ directory:
    pip install flask flask-cors
    python server.py

Endpoints:
    GET  /api/health          — liveness check
    POST /api/simulate        — { "conditions": ["VSD", …] } → analysis JSON
    POST /api/upload          — multipart .nii.gz file       → analysis JSON + obj_url
    GET  /scan_outputs/<file> — serve generated OBJ meshes
"""

import json
import os
import sys
import tempfile

import numpy as np
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

# ── ensure data-processing/ is on the path ───────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))

from backend_pipeline import HeartBackend, extract_features_from_nii
from condition_analysis import (
    run_pipeline,
    scaling_factors_for_viewer,
    FEATURE_ORDER,
)

# ── app setup ─────────────────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app)  # frontend on :8080, server on :5000 — need CORS

SCAN_OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "scan_outputs")
os.makedirs(SCAN_OUTPUT_DIR, exist_ok=True)

# ── lazy-initialise backend once at startup ───────────────────────────────────
print("[HeartScape] Loading backend…")
_backend = HeartBackend(
    csv_path=os.path.join(os.path.dirname(__file__), "heart_features.csv"),
    vectorai_host="localhost:50051",
)

# pre-load condition data for condition_scores computation
_df, _reference, _condition_effects, _condition_counts, _condition_multipliers = run_pipeline(
    os.path.join(os.path.dirname(__file__), "heart_features.csv")
)

# patient ID → ground-truth category lookup from clinical CSV
import re as _re
import pandas as _pd
_clinical: dict = {}
try:
    _cdf = _pd.read_csv(os.path.join(os.path.dirname(__file__), "hvsmr_clinical.csv"))
    _clinical = {int(r["Pat"]): str(r["Category"]) for _, r in _cdf.iterrows()}
    print(f"[HeartScape] Clinical data loaded for {len(_clinical)} patients.")
except Exception as _e:
    print(f"[HeartScape] Warning: could not load clinical CSV ({_e})")

print("[HeartScape] Backend ready.")


# ── helpers ───────────────────────────────────────────────────────────────────

def _json_safe(obj):
    """Recursively make an object JSON-serialisable (numpy → plain Python)."""
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        return None if (obj != obj) else float(obj)  # NaN → None
    if isinstance(obj, np.ndarray):
        return _json_safe(obj.tolist())
    return obj


def _condition_scores(features: dict, top_n: int = 8) -> dict:
    """
    Proximity-based soft condition attribution using softmax over distances.
    Softmax with auto-scaled temperature amplifies relative differences so the
    closest condition stands out instead of everything clustering at ~40%.
    """
    ref_vec = np.array([max(_reference.get(k, 1e-6), 1e-6) for k in FEATURE_ORDER])
    p_vec   = np.array([features.get(k, 0) for k in FEATURE_ORDER])
    raw_dists = {}
    for cond, cond_feats in _condition_effects.items():
        if _condition_counts.get(cond, 0) < 2:
            continue
        c_vec = np.array([cond_feats.get(k, _reference.get(k, 0)) for k in FEATURE_ORDER])
        raw_dists[cond] = float(np.linalg.norm((p_vec - c_vec) / ref_vec))

    if not raw_dists:
        return {}

    # Softmax over negative distances with auto-scaled temperature
    conds = list(raw_dists.keys())
    dists = np.array([raw_dists[c] for c in conds])
    spread = dists.max() - dists.min()
    temp   = min(3.0 / max(spread, 0.05), 6.0)  # softer cap so runner-ups still show
    exp_w  = np.exp(-dists * temp)
    exp_w /= exp_w.sum()

    scores = {c: round(float(w), 4) for c, w in zip(conds, exp_w)}
    return dict(sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_n])


def _best_condition_scores(features: dict, vector_search: dict | None) -> dict:
    """
    Use VectorAI soft_attribution when available, otherwise proximity fallback.
    Either way, apply softmax over raw distances so scores are actually spread out
    rather than clustering at the same value via 1/(1+d) compression.
    """
    if vector_search and vector_search.get("soft_attribution"):
        attrs = [a for a in vector_search["soft_attribution"] if "condition" in a and "distance" in a]
        if attrs:
            conds = [a["condition"] for a in attrs]
            dists  = np.array([float(a["distance"]) for a in attrs])
            spread = dists.max() - dists.min()
            temp   = min(3.0 / max(spread, 0.05), 6.0)
            exp_w  = np.exp(-dists * temp)
            exp_w /= exp_w.sum()
            scores = {c: round(float(w), 4) for c, w in zip(conds, exp_w)}
            return dict(sorted(scores.items(), key=lambda x: x[1], reverse=True))
    return _condition_scores(features)


def _build_response(raw: dict, features: dict, patient_id=None, ground_truth=None) -> dict:
    """
    Shape the backend payload to match pat{N}_analysis.json so the existing
    displayScanAnalysis() function works without any changes.
    """
    pca_result = raw.get("pca", {})
    nearest    = raw.get("nearest") or raw.get("similarity") or []
    return {
        "patient_id":       patient_id,
        "ground_truth":     ground_truth or "Unknown",
        "source":           raw.get("source", "unknown"),
        "conditions":       raw.get("conditions", []),
        "obj_url":          raw.get("obj_url"),          # only present for scan uploads
        "features":         {k: round(float(features.get(k, 0)), 4) for k in FEATURE_ORDER},
        "mesh_scales":      {k: round(float(v), 4) for k, v in raw.get("mesh_scales", {}).items()},
        "pca": {
            "pc1": round(float(pca_result.get("pc1", 0)), 4),
            "pc2": round(float(pca_result.get("pc2", 0)), 4),
        },
        "severity":         round(float(raw.get("severity", 0)), 4),
        "nearest":          nearest,
        "condition_scores": _best_condition_scores(features, raw.get("vector_search")),
        "vector_search":    raw.get("vector_search"),
    }


# ── routes ────────────────────────────────────────────────────────────────────

@app.get("/api/health")
def health():
    return jsonify({
        "status":   "ok",
        "vectorai": _backend.vectorai is not None,
    })


@app.post("/api/simulate")
def simulate():
    body       = request.get_json(force=True, silent=True) or {}
    conditions = body.get("conditions", [])
    if not isinstance(conditions, list):
        return jsonify({"error": "conditions must be a list"}), 400

    raw      = _backend.simulate_heart(conditions)
    features = raw.get("extracted_features", {})
    resp     = _build_response(raw, features, ground_truth="Simulation")
    return jsonify(_json_safe(resp))


@app.post("/api/upload")
def upload():
    f = request.files.get("file")
    if not f:
        return jsonify({"error": "no file field in request"}), 400
    if not (f.filename.endswith(".nii.gz") or f.filename.endswith(".nii")):
        return jsonify({"error": "expected a .nii.gz or .nii file"}), 400

    # Save to scan_outputs/ with original filename
    safe_name = os.path.basename(f.filename).replace(" ", "_")
    save_path = os.path.join(SCAN_OUTPUT_DIR, safe_name)
    f.save(save_path)

    raw      = _backend.analyze_scan(save_path, output_dir=SCAN_OUTPUT_DIR)
    features = raw.get("extracted_features", {})

    # obj_url — browser-accessible URL for the generated OBJ
    obj_filename = os.path.basename(raw.get("obj_path", ""))
    raw["obj_url"] = f"/scan_outputs/{obj_filename}" if obj_filename else None

    # Resolve patient ID and ground truth from filename + clinical CSV
    pid_match = _re.search(r'pat(\d+)', safe_name, _re.IGNORECASE)
    pid = int(pid_match.group(1)) if pid_match else None
    gt  = _clinical.get(pid, "Unknown") if pid is not None else "Unknown"

    resp = _build_response(raw, features, patient_id=pid, ground_truth=gt)
    return jsonify(_json_safe(resp))


@app.get("/scan_outputs/<path:filename>")
def serve_scan_output(filename):
    return send_from_directory(SCAN_OUTPUT_DIR, filename)


# ── run ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("[HeartScape] Server running at http://localhost:5000")
    app.run(host="0.0.0.0", port=5000, debug=False)
