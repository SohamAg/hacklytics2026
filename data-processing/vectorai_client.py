"""
Actian VectorAI DB client — multi-resolution anatomical search.

All query vectors are z-score standardized with the scaler fitted during ingestion
(saved at ../models/vectorai_scaler.json) so they land in the same space as the
stored vectors.  Euclidean distance is used, so lower score == closer match.

Distances are converted to a [0, 1] similarity score via:
    similarity = 1 / (1 + euclidean_distance)
so 1.0 = identical, ~0.5 = one z-score apart, ~0.0 = very far away.
"""

import json
import os
from typing import Optional

import numpy as np

try:
    from cortex import CortexClient, DistanceMetric
    from cortex.filters import Filter, Condition, ConditionType
    CORTEX_AVAILABLE = True
except ImportError:
    CORTEX_AVAILABLE = False

FULL_FEATURES    = [f"Label_{i}_vol_ml" for i in range(1, 9)] + ["Total_heart_vol"]
CHAMBER_FEATURES = ["Label_1_vol_ml", "Label_2_vol_ml", "Label_3_vol_ml", "Label_4_vol_ml"]
VESSEL_FEATURES  = ["Label_5_vol_ml", "Label_6_vol_ml", "Label_7_vol_ml", "Label_8_vol_ml"]

SCALER_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "vectorai_scaler.json")


# ── helpers ─────────────────────────────────────────────────────────────────────

def _load_scaler():
    if not os.path.isfile(SCALER_PATH):
        raise FileNotFoundError(
            f"Scaler not found at {SCALER_PATH}. Run vectorai_setup.py first."
        )
    with open(SCALER_PATH) as f:
        return json.load(f)


def _zscore(feature_dict: dict, scaler: dict) -> list:
    features = scaler["features"]
    means    = scaler["mean"]
    stds     = scaler["std"]
    return [
        (float(feature_dict.get(f, 0) or 0) - m) / s
        for f, m, s in zip(features, means, stds)
    ]


def _dist_to_similarity(distance: float) -> float:
    """Convert Euclidean distance → [0, 1] similarity. 1 = identical."""
    return 1.0 / (1.0 + distance)


def _anomaly_label(top_distance: Optional[float]) -> str:
    if top_distance is None:
        return "unknown"
    # Euclidean distance in z-score space:
    #   0–1 → within 1 std-dev range across all features  → normal
    #   1–2 → moderately unusual
    #   2+  → substantially different from any known case
    if top_distance < 1.0:
        return "within normal range"
    if top_distance < 2.0:
        return "mild deviation"
    if top_distance < 3.0:
        return "moderate anomaly"
    return "significant anomaly"


def _patient_from_result(r, distance: float) -> dict:
    payload = r.payload or {}
    return {
        "patient_id": payload.get("patient_id", r.id),
        "category":   payload.get("category", "Unknown"),
        "distance":   round(distance, 4),
        "similarity": round(_dist_to_similarity(distance), 4),
    }


def _archetype_from_result(r, distance: float) -> dict:
    payload = r.payload or {}
    return {
        "condition":  payload.get("condition", f"cond_{r.id}"),
        "n_patients": payload.get("n_patients", 0),
        "distance":   round(distance, 4),
        "similarity": round(_dist_to_similarity(distance), 4),
    }


# ── client ───────────────────────────────────────────────────────────────────────

class VectorAIClient:
    def __init__(self, host: str = "localhost:50051"):
        if not CORTEX_AVAILABLE:
            raise ImportError("cortex package not installed.")
        self.host    = host
        self._scaler = _load_scaler()

    def is_available(self) -> bool:
        try:
            with CortexClient(self.host) as c:
                c.health_check()
            return True
        except Exception:
            return False

    # ── per-resolution patient search ──────────────────────────────────────────

    def _search(self, client, collection: str, vec: list, k: int,
                filter=None) -> list:
        return client.search(collection, query=vec, top_k=k,
                             filter=filter, with_payload=True)

    def _min_patients_filter(self, min_patients: int) -> "Filter":
        return Filter(_must=[Condition("n_patients", ConditionType.GTE, value=min_patients)])

    def nearest_patients(self, feature_dict: dict, k: int = 5) -> list:
        vec = _zscore(feature_dict, self._scaler["full"])
        with CortexClient(self.host) as c:
            results = self._search(c, "heart_patients", vec, k)
        return [_patient_from_result(r, r.score) for r in results]

    def nearest_chamber_patients(self, feature_dict: dict, k: int = 5) -> list:
        vec = _zscore(feature_dict, self._scaler["chambers"])
        with CortexClient(self.host) as c:
            results = self._search(c, "heart_chambers", vec, k)
        return [_patient_from_result(r, r.score) for r in results]

    def nearest_vessel_patients(self, feature_dict: dict, k: int = 5) -> list:
        vec = _zscore(feature_dict, self._scaler["vessels"])
        with CortexClient(self.host) as c:
            results = self._search(c, "heart_vessels", vec, k)
        return [_patient_from_result(r, r.score) for r in results]

    # ── per-resolution archetype search ────────────────────────────────────────

    def nearest_condition_archetypes(
        self, feature_dict: dict, k: int = 5, min_patients: int = 2
    ) -> list:
        vec = _zscore(feature_dict, self._scaler["full"])
        flt = self._min_patients_filter(min_patients)
        with CortexClient(self.host) as c:
            results = self._search(c, "condition_archetypes", vec, k, filter=flt)
        return [_archetype_from_result(r, r.score) for r in results]

    def nearest_chamber_archetypes(
        self, feature_dict: dict, k: int = 5, min_patients: int = 2
    ) -> list:
        vec = _zscore(feature_dict, self._scaler["chambers"])
        flt = self._min_patients_filter(min_patients)
        with CortexClient(self.host) as c:
            results = self._search(c, "chamber_archetypes", vec, k, filter=flt)
        return [_archetype_from_result(r, r.score) for r in results]

    def nearest_vessel_archetypes(
        self, feature_dict: dict, k: int = 5, min_patients: int = 2
    ) -> list:
        vec = _zscore(feature_dict, self._scaler["vessels"])
        flt = self._min_patients_filter(min_patients)
        with CortexClient(self.host) as c:
            results = self._search(c, "vessel_archetypes", vec, k, filter=flt)
        return [_archetype_from_result(r, r.score) for r in results]

    # ── orchestrated multi-resolution query ────────────────────────────────────

    def multi_resolution_query(self, feature_dict: dict) -> dict:
        """
        Run full, chamber, and vessel searches simultaneously.
        Returns a structured dict with patients, archetypes, anomaly score,
        and soft condition attribution across all resolutions.
        """
        full_patients  = self.nearest_patients(feature_dict, k=5)
        cham_patients  = self.nearest_chamber_patients(feature_dict, k=5)
        ves_patients   = self.nearest_vessel_patients(feature_dict, k=5)

        full_archetypes = self.nearest_condition_archetypes(feature_dict, k=5)
        cham_archetypes = self.nearest_chamber_archetypes(feature_dict, k=5)
        ves_archetypes  = self.nearest_vessel_archetypes(feature_dict, k=5)

        # Anomaly: distance from the closest real patient in full-heart space
        top_distance = full_patients[0]["distance"] if full_patients else None
        anomaly_score = round(_dist_to_similarity(top_distance), 4) if top_distance is not None else None

        # Soft condition attribution from full archetype space
        soft_attribution = [
            {
                "condition":  a["condition"],
                "n_patients": a["n_patients"],
                "similarity": a["similarity"],
                "distance":   a["distance"],
            }
            for a in full_archetypes
        ]

        return {
            "anomaly_score":      anomaly_score,
            "anomaly_label":      _anomaly_label(top_distance),
            "nearest_patients":   full_patients,
            "soft_attribution":   soft_attribution,
            "multi_resolution": {
                "full":    {"patients": full_patients,  "archetypes": full_archetypes},
                "chambers":{"patients": cham_patients,  "archetypes": cham_archetypes},
                "vessels": {"patients": ves_patients,   "archetypes": ves_archetypes},
            },
        }
