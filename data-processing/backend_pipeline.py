"""
HeartBackend — orchestration layer.

Two entry points:
    simulate_heart(conditions)  — generates a morphology by blending condition multipliers
    analyze_scan(nii_path)      — processes a real segmented NIfTI file end-to-end

Both return the same payload shape so visualization code is identical for both.
"""

import os
from typing import List, Dict, Any, Optional

import nibabel as nib
import numpy as np

from condition_analysis import (
    run_pipeline,
    estimate_features,
    scaling_factors_for_viewer,
    FEATURE_ORDER,
)
from morphology_engine import MorphologyEngine
from vectorai_client import VectorAIClient


# ── feature extraction from a single NIfTI file ──────────────────────────────

def extract_features_from_nii(nii_path: str) -> dict:
    """
    Extract per-label volumetric features from a segmented NIfTI file.
    Labels 1–8 map to LV, RV, LA, RA, Aorta, PA, SVC, IVC.
    Returns a dict with the same keys as heart_features.csv.
    """
    seg   = nib.load(nii_path)
    data  = seg.get_fdata()
    vox   = np.prod(seg.header.get_zooms())   # voxel volume in mm³

    features = {}
    for label_id in range(1, 9):
        vol_mm3 = float(np.sum(data == label_id)) * vox
        features[f"Label_{label_id}_vol_ml"] = vol_mm3 / 1000.0

    features["Total_heart_vol"] = sum(
        features[f"Label_{i}_vol_ml"] for i in range(1, 9)
    )

    lv = features["Label_1_vol_ml"]
    rv = features["Label_2_vol_ml"]
    la = features["Label_3_vol_ml"]
    ra = features["Label_4_vol_ml"]
    tv = features["Total_heart_vol"]

    features["LV_RV_ratio"] = lv / rv  if rv  > 0 else float("inf")
    features["LA_RA_ratio"] = la / ra  if ra  > 0 else float("inf")
    features["AO_fraction"] = features["Label_5_vol_ml"] / tv if tv > 0 else 0.0
    features["LV_fraction"] = lv / tv  if tv  > 0 else 0.0

    return features


# ── backend ───────────────────────────────────────────────────────────────────

class HeartBackend:
    """
    Full backend orchestration layer.

    On init:
        - Loads dataset and computes condition multipliers
        - Loads PCA + scaler models
        - Optionally connects to VectorAI DB

    simulate_heart()  — synthetic morphology from condition selection
    analyze_scan()    — real morphology from uploaded .nii.gz file
    """

    def __init__(self, csv_path: str, vectorai_host: Optional[str] = "localhost:50051"):
        _, self.reference, _, _, self.condition_multipliers = run_pipeline(csv_path)
        self.morph_engine = MorphologyEngine()

        self.vectorai = None
        if vectorai_host:
            client = VectorAIClient(host=vectorai_host)
            if client.is_available():
                self.vectorai = client
                print(f"VectorAI DB connected at {vectorai_host}")
            else:
                print(f"VectorAI DB not reachable at {vectorai_host} — running without vector search")

    # ── shared analysis core ──────────────────────────────────────────────────

    def _analyze_features(self, features: dict, source_meta: dict) -> Dict[str, Any]:
        """Run PCA, mesh scaling, and VectorAI on any feature dict."""
        analysis      = self.morph_engine.analyze(features)
        scales        = scaling_factors_for_viewer(features, self.reference)
        vector_search = self.vectorai.multi_resolution_query(features) if self.vectorai else None

        return {
            **source_meta,
            "extracted_features": features,
            "mesh_scales":        scales,
            "pca": {
                "pc1": analysis["pc1"],
                "pc2": analysis["pc2"],
            },
            "similarity":    analysis["nearest"],
            "severity":      analysis["severity"],
            "vector_search": vector_search,
        }

    # ── entry point 1: condition simulation ──────────────────────────────────

    def simulate_heart(self, selected_conditions: List[str]) -> Dict[str, Any]:
        """
        Blend condition multipliers to produce an estimated feature vector,
        then run full analysis. Used by the condition-selector UI.
        """
        features = estimate_features(
            selected_conditions,
            self.reference,
            self.condition_multipliers,
        )
        return self._analyze_features(features, {
            "source":     "simulation",
            "conditions": selected_conditions,
        })

    # ── entry point 2: real scan upload ──────────────────────────────────────

    def analyze_scan(
        self,
        nii_path: str,
        output_dir: str = "scan_outputs",
    ) -> Dict[str, Any]:
        """
        Process a real segmented NIfTI file end-to-end:
            1. Extract volumetric features from voxel label counts
            2. Export an OBJ mesh for 3D rendering
            3. Run PCA projection + severity + similarity
            4. Run VectorAI multi-resolution search (condition attribution)

        Args:
            nii_path:   Path to a *_seg.nii.gz segmentation file
            output_dir: Directory to write the generated OBJ file

        Returns:
            Same payload shape as simulate_heart(), plus obj_path and scan_path.
        """
        from mesh_rendering import save_segmented_obj, LABEL_INFO

        # 1. Feature extraction
        features  = extract_features_from_nii(nii_path)

        # 2. Mesh export
        os.makedirs(output_dir, exist_ok=True)
        base_name = os.path.basename(nii_path).replace(".nii.gz", "").replace(".nii", "")
        obj_path  = os.path.join(output_dir, f"{base_name}.obj")

        seg = nib.load(nii_path)
        save_segmented_obj(obj_path, seg.get_fdata(), seg.header.get_zooms(), LABEL_INFO)

        # 3 + 4. Analysis
        return self._analyze_features(features, {
            "source":    "scan",
            "scan_path": nii_path,
            "obj_path":  obj_path,
        })
