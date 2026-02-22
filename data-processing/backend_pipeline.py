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

    def __init__(self, csv_path: str):
        _, self.reference, _, _, self.condition_multipliers = run_pipeline(csv_path)
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