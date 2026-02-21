import os
from typing import List, Dict, Any

# --- Condition system ---
from condition_analysis import (
    run_pipeline,
    estimate_features,
    scaling_factors_for_viewer,
)

# --- Morphology intelligence ---
from morphology_engine import MorphologyEngine


class HeartBackend:
    """
    Full backend orchestration layer.
    Connects:
        condition estimation
        PCA projection
        similarity search
        severity scoring
        mesh scaling
    """

    def __init__(self, csv_path: str):
        # 1️⃣ Load dataset + compute condition effects
        (
            self.df,
            self.reference,
            self.condition_effects,
            self.condition_counts,
            self.condition_multipliers
        ) = run_pipeline(csv_path)

        # 2️⃣ Load PCA + scaler + dataset embeddings
        self.morph_engine = MorphologyEngine()

    # ---------------------------------------------------------
    # Core user-facing function
    # ---------------------------------------------------------

    def simulate_heart(
        self,
        selected_conditions: List[str]
    ) -> Dict[str, Any]:
        """
        Main function called by frontend.

        Input:
            selected_conditions: list of condition names

        Output:
            dict with:
                - estimated_features
                - mesh_scales
                - pca_location
                - similarity
                - severity_score
        """

        # Step 1: Estimate volumetric feature vector
        estimated = estimate_features(
            selected_conditions,
            self.reference,
            self.condition_multipliers,
        )

        # Step 2: PCA + similarity + severity
        analysis = self.morph_engine.analyze(estimated)

        # Step 3: Mesh scaling factors
        scales = scaling_factors_for_viewer(
            estimated,
            self.reference,
        )

        # Step 4: Unified payload
        return {
            "conditions": selected_conditions,
            "estimated_features": estimated,
            "mesh_scales": scales,
            "pca": {
                "pc1": analysis["pc1"],
                "pc2": analysis["pc2"]
            },
            "similarity": analysis["nearest"],
            "severity": analysis["severity"]
        }