"""
One-time ingestion script: populates Actian VectorAI DB with patient feature vectors
and condition archetype vectors at three anatomical resolutions.

Uses z-score standardization + Euclidean distance so that magnitude differences
(e.g. RV twice the normal size) are captured correctly — unlike cosine similarity
which only measures direction and collapses all hearts to ~0.999 similarity.

Run once after starting the Docker container:
    docker compose up -d
    python vectorai_setup.py

Six collections:
    heart_patients         — 9D whole-heart vector per patient
    condition_archetypes   — 9D mean anatomy per condition
    heart_chambers         — 4D chamber-only (LV, RV, LA, RA)
    heart_vessels          — 4D vessel-only (Aorta, PA, SVC, IVC)
    chamber_archetypes     — 4D chamber archetype per condition
    vessel_archetypes      — 4D vessel archetype per condition

Scaler parameters saved to ../models/vectorai_scaler.json
so the client can apply the same standardization to query vectors.
"""

import json
import os
import sys

import numpy as np

from cortex import CortexClient, DistanceMetric

from condition_analysis import run_pipeline

# Feature sets for each resolution
FULL_FEATURES    = [f"Label_{i}_vol_ml" for i in range(1, 9)] + ["Total_heart_vol"]
CHAMBER_FEATURES = ["Label_1_vol_ml", "Label_2_vol_ml", "Label_3_vol_ml", "Label_4_vol_ml"]
VESSEL_FEATURES  = ["Label_5_vol_ml", "Label_6_vol_ml", "Label_7_vol_ml", "Label_8_vol_ml"]

PATIENTS_COLLECTION           = "heart_patients"
ARCHETYPES_COLLECTION         = "condition_archetypes"
CHAMBERS_COLLECTION           = "heart_chambers"
VESSELS_COLLECTION            = "heart_vessels"
CHAMBER_ARCHETYPES_COLLECTION = "chamber_archetypes"
VESSEL_ARCHETYPES_COLLECTION  = "vessel_archetypes"

HOST             = "localhost:50051"
SCALER_PATH      = os.path.join(os.path.dirname(__file__), "..", "models", "vectorai_scaler.json")

# These patients are held out from VectorAI so they can be used as unseen demo uploads
HOLDOUT_PATIENTS = [1, 14, 16, 28, 57]


# ── standardization ────────────────────────────────────────────────────────────

def fit_scaler(df, features: list) -> dict:
    """Compute mean and std per feature from patient data."""
    means, stds = [], []
    for f in features:
        vals = df[f].replace([np.inf, -np.inf], np.nan).dropna().values.astype(float)
        means.append(float(np.mean(vals)))
        std = float(np.std(vals))
        stds.append(std if std > 0 else 1.0)   # avoid division by zero
    return {"features": features, "mean": means, "std": stds}


def zscore(raw: list, scaler: dict) -> list:
    """Apply z-score standardization using a fitted scaler."""
    means = scaler["mean"]
    stds  = scaler["std"]
    return [(float(v) - m) / s for v, m, s in zip(raw, means, stds)]


def is_valid(vec: list) -> bool:
    return not any(np.isnan(v) or np.isinf(v) for v in vec)


# ── vector builders ─────────────────────────────────────────────────────────────

def build_patient_vectors(df, features: list, scaler: dict):
    ids, vectors, payloads = [], [], []
    for _, row in df.iterrows():
        raw = [float(row.get(f, 0) or 0) for f in features]
        if not is_valid(raw):
            continue
        ids.append(int(row["Patient"]))
        vectors.append(zscore(raw, scaler))
        payloads.append({
            "patient_id": int(row["Patient"]),
            "category":   str(row.get("Category", "Unknown")),
        })
    return ids, vectors, payloads


def build_archetype_vectors(reference, condition_effects, condition_counts,
                             features: list, scaler: dict):
    ids, vectors, payloads = [], [], []
    for i, (cond, feats) in enumerate(condition_effects.items()):
        raw = [float(feats.get(f, reference.get(f, 0)) or 0) for f in features]
        if not is_valid(raw):
            print(f"  Skipping '{cond}' — NaN/Inf in features")
            continue
        ids.append(i)
        vectors.append(zscore(raw, scaler))
        payloads.append({
            "condition":  cond,
            "n_patients": int(condition_counts.get(cond, 0)),
        })
    return ids, vectors, payloads


# ── ingestion ───────────────────────────────────────────────────────────────────

def ingest_collection(client, name: str, dimension: int, ids, vectors, payloads):
    print(f"\nCreating '{name}' (dim={dimension}, EUCLIDEAN)...")
    client.recreate_collection(
        name=name,
        dimension=dimension,
        distance_metric=DistanceMetric.EUCLIDEAN,
    )
    client.batch_upsert(name, ids=ids, vectors=vectors, payloads=payloads)
    count = client.count(name)
    print(f"  Inserted {count} vectors")


def ingest(csv_path: str, host: str = HOST):
    print(f"Loading data from {csv_path}...")
    df, reference, condition_effects, condition_counts, _ = run_pipeline(csv_path)
    print(f"  {len(df)} patients, {len(condition_effects)} conditions loaded")

    # Remove holdout patients — they stay unseen for demo uploads
    df = df[~df["Patient"].isin(HOLDOUT_PATIENTS)].reset_index(drop=True)
    print(f"  Excluding holdout patients {HOLDOUT_PATIENTS} -> {len(df)} patients for ingestion")

    # Fit scalers on patient data
    print("\nFitting z-score scalers...")
    scaler_full    = fit_scaler(df, FULL_FEATURES)
    scaler_chambers = fit_scaler(df, CHAMBER_FEATURES)
    scaler_vessels  = fit_scaler(df, VESSEL_FEATURES)

    # Save scalers for client use
    scaler_data = {
        "full":     scaler_full,
        "chambers": scaler_chambers,
        "vessels":  scaler_vessels,
    }
    os.makedirs(os.path.dirname(SCALER_PATH), exist_ok=True)
    with open(SCALER_PATH, "w") as f:
        json.dump(scaler_data, f, indent=2)
    print(f"  Scaler saved to {SCALER_PATH}")

    # Build all vector sets
    p_full_ids,  p_full_vecs,  p_full_pay  = build_patient_vectors(df, FULL_FEATURES,    scaler_full)
    p_cham_ids,  p_cham_vecs,  p_cham_pay  = build_patient_vectors(df, CHAMBER_FEATURES, scaler_chambers)
    p_ves_ids,   p_ves_vecs,   p_ves_pay   = build_patient_vectors(df, VESSEL_FEATURES,  scaler_vessels)

    a_full_ids,  a_full_vecs,  a_full_pay  = build_archetype_vectors(reference, condition_effects, condition_counts, FULL_FEATURES,    scaler_full)
    a_cham_ids,  a_cham_vecs,  a_cham_pay  = build_archetype_vectors(reference, condition_effects, condition_counts, CHAMBER_FEATURES, scaler_chambers)
    a_ves_ids,   a_ves_vecs,   a_ves_pay   = build_archetype_vectors(reference, condition_effects, condition_counts, VESSEL_FEATURES,  scaler_vessels)

    print(f"\nConnecting to VectorAI DB at {host}...")
    with CortexClient(host) as client:
        version, uptime = client.health_check()
        print(f"  Connected — {version}, uptime {uptime}s")

        ingest_collection(client, PATIENTS_COLLECTION,           len(FULL_FEATURES),    p_full_ids, p_full_vecs, p_full_pay)
        ingest_collection(client, ARCHETYPES_COLLECTION,         len(FULL_FEATURES),    a_full_ids, a_full_vecs, a_full_pay)
        ingest_collection(client, CHAMBERS_COLLECTION,           len(CHAMBER_FEATURES), p_cham_ids, p_cham_vecs, p_cham_pay)
        ingest_collection(client, VESSELS_COLLECTION,            len(VESSEL_FEATURES),  p_ves_ids,  p_ves_vecs,  p_ves_pay)
        ingest_collection(client, CHAMBER_ARCHETYPES_COLLECTION, len(CHAMBER_FEATURES), a_cham_ids, a_cham_vecs, a_cham_pay)
        ingest_collection(client, VESSEL_ARCHETYPES_COLLECTION,  len(VESSEL_FEATURES),  a_ves_ids,  a_ves_vecs,  a_ves_pay)

    print("\nIngestion complete. VectorAI DB is ready.")


if __name__ == "__main__":
    base = os.path.dirname(__file__)
    csv_path = os.path.join(base, "heart_features.csv")
    if not os.path.isfile(csv_path):
        print(f"Error: heart_features.csv not found at {csv_path}", file=sys.stderr)
        sys.exit(1)
    ingest(csv_path)
