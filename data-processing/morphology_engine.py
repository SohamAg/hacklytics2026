import numpy as np
import pandas as pd
import joblib

from condition_analysis import FEATURE_ORDER

class MorphologyEngine:
    def __init__(self):
        self.pca = joblib.load("../models/pca_model.pkl")
        self.scaler = joblib.load("../models/scaler_model.pkl")
        self.dataset = pd.read_csv("../models/pca_dataset.csv")

        self.dataset_pc = self.dataset[["PC1", "PC2"]].values
        self.patient_ids = self.dataset["Patient"].values
        self.categories = self.dataset["Category"].values

        self.min_pc1 = self.dataset_pc[:,0].min()
        self.max_pc1 = self.dataset_pc[:,0].max()

    def project(self, feature_dict):
        vec = np.array([feature_dict[f] for f in FEATURE_ORDER]).reshape(1, -1)
        vec_scaled = self.scaler.transform(vec)
        vec_pc = self.pca.transform(vec_scaled)
        return vec_pc

    def similarity_search(self, vec_pc, k=3):
        distances = np.linalg.norm(self.dataset_pc - vec_pc, axis=1)
        idx = np.argsort(distances)[:k]

        results = []
        for i in idx:
            results.append({
                "patient_id": int(self.patient_ids[i]),
                "category": self.categories[i],
                "distance": float(distances[i])
            })

        return results

    def severity_score(self, vec_pc):
        raw = vec_pc[0,0]
        norm = (raw - self.min_pc1) / (self.max_pc1 - self.min_pc1)
        return float(np.clip(norm, 0, 1))

    def analyze(self, estimated_features):
        vec_pc = self.project(estimated_features)
        neighbors = self.similarity_search(vec_pc)
        severity = self.severity_score(vec_pc)

        return {
            "pc1": float(vec_pc[0,0]),
            "pc2": float(vec_pc[0,1]),
            "severity": severity,
            "nearest": neighbors
        }