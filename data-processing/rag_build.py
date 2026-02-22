#!/usr/bin/env python3
"""Build RAG index from heart_features.csv. Run once to create the vector store."""

import pickle
import numpy as np
import pandas as pd
from pathlib import Path

# Column name -> human-readable cardiac structure (HVSMR convention)
LABEL_NAMES = {
    "Label_1_vol_ml": "LV (left ventricle)",
    "Label_2_vol_ml": "RV (right ventricle)",
    "Label_3_vol_ml": "LA (left atrium)",
    "Label_4_vol_ml": "RA (right atrium)",
    "Label_5_vol_ml": "Aorta",
    "Label_6_vol_ml": "Pulmonary artery",
    "Label_7_vol_ml": "SVC",
    "Label_8_vol_ml": "IVC",
}

# Condition columns (binary X = present)
CONDITION_COLS = [
    "Normal", "MildModerateDilation", "VSD", "ASD", "DORV", "DLoopTGA",
    "ArterialSwitch", "BilateralSVC", "SevereDilation", "TortuousVessels",
    "Dextrocardia", "Mesocardia", "InvertedVentricles", "InvertedAtria",
    "LeftCentralIVC", "LeftCentralSVC", "LLoopTGA", "AtrialSwitch", "Rastelli",
    "SingleVentricle", "DILV", "DIDORV", "CommonAtrium", "Glenn", "Fontan",
    "Heterotaxy", "SuperoinferiorVentricles", "PAAtresiaOrMPAStump", "PABanding",
    "AOPAAnastamosis", "Marfan", "CMRArtifactAO", "CMRArtifactPA",
]


def row_to_document(row: pd.Series) -> str:
    """Convert a patient row to a rich text document for embedding."""
    parts = []

    # Demographics
    pat_id = row.get("Patient", row.get("Pat", "?"))
    age = row.get("Age", "?")
    category = row.get("Category", "?")
    parts.append(f"Patient {pat_id}, age {age}, severity category: {category}.")

    # Volumetric data
    vol_parts = []
    for col, name in LABEL_NAMES.items():
        if col in row and pd.notna(row[col]):
            try:
                vol = float(row[col])
                if vol > 0:
                    vol_parts.append(f"{name} {vol:.1f} ml")
            except (ValueError, TypeError):
                pass
    if vol_parts:
        parts.append("Chamber volumes: " + "; ".join(vol_parts) + ".")

    # Total and ratios
    if "Total_heart_vol" in row and pd.notna(row["Total_heart_vol"]):
        parts.append(f"Total heart volume {float(row['Total_heart_vol']):.1f} ml.")
    if "LV_RV_ratio" in row and pd.notna(row["LV_RV_ratio"]) and str(row["LV_RV_ratio"]) != "inf":
        try:
            parts.append(f"LV/RV ratio {float(row['LV_RV_ratio']):.2f}.")
        except (ValueError, TypeError):
            pass
    if "LA_RA_ratio" in row and pd.notna(row["LA_RA_ratio"]) and str(row["LA_RA_ratio"]) != "inf":
        try:
            parts.append(f"LA/RA ratio {float(row['LA_RA_ratio']):.2f}.")
        except (ValueError, TypeError):
            pass

    # Conditions (abbreviations expanded for searchability)
    conditions = []
    for col in CONDITION_COLS:
        if col in row and pd.notna(row[col]) and str(row[col]).strip().upper() == "X":
            conditions.append(col)
    if conditions:
        parts.append("Conditions: " + ", ".join(conditions) + ".")

    return " ".join(parts)


def build_index(csv_path: str, index_path: str = "heart_rag_index.pkl"):
    """Load CSV, create documents, embed, and persist to pickle + npz."""
    import os
    from sentence_transformers import SentenceTransformer

    csv_path = Path(csv_path)
    # Use project-local cache to avoid permission issues
    cache_dir = Path(__file__).parent / ".hf_cache"
    os.environ.setdefault("HF_HOME", str(cache_dir))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(cache_dir))
    index_path = Path(index_path)

    df = pd.read_csv(csv_path)
    documents = []
    metadatas = []

    for idx, row in df.iterrows():
        doc = row_to_document(row)
        documents.append(doc)
        metadatas.append({
            "patient": str(row.get("Patient", row.get("Pat", idx))),
            "age": str(row.get("Age", "")),
            "category": str(row.get("Category", "")),
        })

    print(f"Created {len(documents)} documents from {csv_path.name}")

    # Embed with sentence-transformers (runs locally, no API key)
    model = SentenceTransformer("all-MiniLM-L6-v2", cache_folder=str(cache_dir))
    embeddings = model.encode(documents, show_progress_bar=True)

    # Normalize for cosine similarity
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1
    embeddings_norm = embeddings / norms

    # Save index
    index_data = {
        "documents": documents,
        "metadatas": metadatas,
        "model_name": "all-MiniLM-L6-v2",
    }
    with open(index_path, "wb") as f:
        pickle.dump(index_data, f)
    np.savez(index_path.with_suffix(".npz"), embeddings=embeddings_norm)

    print(f"Index saved to {index_path}")
    return index_path


if __name__ == "__main__":
    script_dir = Path(__file__).parent
    csv_path = script_dir / "heart_features.csv"
    index_path = script_dir / "heart_rag_index.pkl"
    build_index(str(csv_path), str(index_path))
