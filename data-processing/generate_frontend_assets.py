"""Generate static JSON/OBJ assets for the HeartScape frontend.

Run from the data-processing/ directory:
    python generate_frontend_assets.py

Outputs (in ../models/):
    condition_effects.json     -- condition multipliers for Simulate mode
    pca_landscape.json         -- PCA coordinates for all training patients
    pat{N}.obj                 -- heart mesh for each holdout patient
    pat{N}_analysis.json       -- pre-computed PCA/condition analysis per holdout patient
"""
import json
import os
import sys

import numpy as np
import pandas as pd
import nibabel as nib

from condition_analysis import (
    run_pipeline,
    export_for_viewer,
    FEATURE_ORDER,
    scaling_factors_for_viewer,
)
from morphology_engine import MorphologyEngine
from mesh_rendering import save_segmented_obj, LABEL_INFO

HOLDOUT_PATIENTS = [1, 14, 16, 28, 57]
MODELS_DIR = "../models"
NII_PATTERN = "./cropped/cropped/pat{}_cropped_seg.nii.gz"
FEATURES_CSV = "heart_features.csv"
CLINICAL_CSV = "hvsmr_clinical.csv"


def nii_path(pid):
    return NII_PATTERN.format(pid)


def load_ground_truth():
    try:
        df = pd.read_csv(CLINICAL_CSV)
        col_pat = next((c for c in df.columns if c.lower() == "patient"), None)
        col_cat = next((c for c in df.columns if c.lower() == "category"), None)
        if not col_pat or not col_cat:
            return {}
        return {int(row[col_pat]): str(row[col_cat]) for _, row in df.iterrows()}
    except Exception as e:
        print(f"  Warning: could not load ground truth ({e})")
        return {}


def compute_condition_scores(features, reference, condition_effects, condition_counts, top_n=8):
    """Proximity-based soft condition attribution — no VectorAI required."""
    ref_vec = np.array([max(reference.get(k, 1e-6), 1e-6) for k in FEATURE_ORDER])
    scores = {}
    for cond, cond_feats in condition_effects.items():
        if condition_counts.get(cond, 0) < 2:
            continue
        c_vec = np.array([cond_feats.get(k, reference.get(k, 0)) for k in FEATURE_ORDER])
        p_vec = np.array([features.get(k, 0) for k in FEATURE_ORDER])
        dist = float(np.linalg.norm((p_vec - c_vec) / ref_vec))
        scores[cond] = round(1.0 / (1.0 + dist), 4)
    return dict(sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_n])


def generate_analysis(pid, reference, condition_effects, condition_counts, engine, gt):
    """Build analysis dict for one holdout patient."""
    df = pd.read_csv(FEATURES_CSV)
    row = df[df["Patient"] == pid]
    if not row.empty:
        r = row.iloc[0]
        features = {k: float(r[k]) for k in FEATURE_ORDER if k in r}
    else:
        print(f"    Patient {pid} not in CSV — extracting from NIfTI")
        from backend_pipeline import extract_features_from_nii
        features = extract_features_from_nii(nii_path(pid))

    mesh_scales = scaling_factors_for_viewer(features, reference)
    pca_result = engine.analyze(features)
    cond_scores = compute_condition_scores(features, reference, condition_effects, condition_counts)

    return {
        "patient_id":       pid,
        "ground_truth":     gt.get(pid, "Unknown"),
        "obj_path":         f"models/pat{pid}.obj",
        "features":         {k: round(float(features[k]), 4) for k in FEATURE_ORDER},
        "mesh_scales":      {name: round(float(v), 4) for name, v in mesh_scales.items()},
        "pca":              {"pc1": round(pca_result["pc1"], 4), "pc2": round(pca_result["pc2"], 4)},
        "severity":         round(pca_result["severity"], 4),
        "nearest":          pca_result["nearest"],
        "condition_scores": cond_scores,
    }


def main():
    os.makedirs(MODELS_DIR, exist_ok=True)

    # 1. condition_effects.json
    print("1/4  Generating condition_effects.json ...")
    df, reference, condition_effects, condition_counts, condition_multipliers = run_pipeline(FEATURES_CSV)
    out = os.path.join(MODELS_DIR, "condition_effects.json")
    export_for_viewer(reference, condition_multipliers, condition_counts, out)
    print(f"     Saved {out}")

    # 2. PCA landscape
    print("2/4  Generating pca_landscape.json ...")
    if not (os.path.exists("../models/pca_model.pkl") and os.path.exists("../models/scaler_model.pkl")):
        print("     ERROR: PCA models not found. Run feature_processing.py first.")
        sys.exit(1)
    engine = MorphologyEngine()
    patients = [
        {
            "patient_id": int(engine.patient_ids[i]),
            "pc1":        round(float(engine.dataset_pc[i, 0]), 4),
            "pc2":        round(float(engine.dataset_pc[i, 1]), 4),
            "category":   str(engine.categories[i]),
        }
        for i in range(len(engine.patient_ids))
    ]
    pc1 = engine.dataset_pc[:, 0]
    pc2 = engine.dataset_pc[:, 1]
    landscape = {
        "patients": patients,
        "bounds": {
            "pc1_min": round(float(pc1.min()) - 0.5, 3),
            "pc1_max": round(float(pc1.max()) + 0.5, 3),
            "pc2_min": round(float(pc2.min()) - 0.5, 3),
            "pc2_max": round(float(pc2.max()) + 0.5, 3),
        },
    }
    lpath = os.path.join(MODELS_DIR, "pca_landscape.json")
    with open(lpath, "w") as f:
        json.dump(landscape, f)
    print(f"     Saved {lpath}  ({len(patients)} training patients)")

    # 3. OBJ meshes for holdout patients
    print("3/4  Generating OBJ meshes for holdout patients ...")
    for pid in HOLDOUT_PATIENTS:
        np_ = nii_path(pid)
        obj = os.path.join(MODELS_DIR, f"pat{pid}.obj")
        if not os.path.exists(np_):
            print(f"     pat{pid}: NIfTI not found ({np_}) -- skipping OBJ")
            continue
        print(f"     pat{pid}: generating OBJ ...", end=" ", flush=True)
        try:
            seg = nib.load(np_)
            save_segmented_obj(obj, seg.get_fdata(), seg.header.get_zooms(), LABEL_INFO)
            print(f"saved to {obj}")
        except Exception as e:
            print(f"FAILED: {e}")

    # 4. Analysis JSONs
    print("4/4  Generating analysis JSONs ...")
    gt = load_ground_truth()
    for pid in HOLDOUT_PATIENTS:
        print(f"     pat{pid} ...", end=" ", flush=True)
        try:
            analysis = generate_analysis(pid, reference, condition_effects, condition_counts, engine, gt)
            apath = os.path.join(MODELS_DIR, f"pat{pid}_analysis.json")
            with open(apath, "w") as f:
                json.dump(analysis, f, indent=2)
            top3 = list(analysis["condition_scores"].keys())[:3]
            print(f"severity={analysis['severity']:.2f}  top={top3}")
        except Exception as e:
            print(f"FAILED: {e}")

    print("\nDone! To serve the app:")
    print("  cd E:\\hacklytics2026")
    print("  python -m http.server 8000")
    print("  Then open http://localhost:8000")


if __name__ == "__main__":
    main()
