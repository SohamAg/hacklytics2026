"""
Condition–feature analysis: discover how congenital conditions relate to heart
volumes/ratios, and estimate heart features for a chosen condition (or mix).

Uses heart_features.csv: chamber/vessel volumes (Label_1–8 = LV, RV, LA, RA, AO, PA, …)
and clinical flags (VSD, ASD, DORV, Normal, etc.). Learns a "normal" reference and
per-condition effects; estimates features for arbitrary condition sets so the 3D
viewer can show "what would this heart look like with VSD+ASD?"
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Feature columns used for trends and estimation (volumes + ratios)
VOL_COLS = [f"Label_{i}_vol_ml" for i in range(1, 9)]
RATIO_COLS = ["LV_RV_ratio", "LA_RA_ratio", "AO_fraction", "LV_fraction"]
FEATURE_COLS = VOL_COLS + ["Total_heart_vol"] + RATIO_COLS

# Subset used by PCA — must match the columns the scaler/PCA were trained on
FEATURE_ORDER = [
    "Label_1_vol_ml",
    "Label_2_vol_ml",
    "Label_4_vol_ml",
    "Label_6_vol_ml",
    "Total_heart_vol",
]

# Condition columns in heart_features (X = present)
CONDITION_COLS = [
    "Normal", "MildModerateDilation", "VSD", "ASD", "DORV", "DLoopTGA",
    "ArterialSwitch", "BilateralSVC", "SevereDilation", "TortuousVessels",
    "Dextrocardia", "Mesocardia", "InvertedVentricles", "InvertedAtria",
    "LeftCentralIVC", "LeftCentralSVC", "LLoopTGA", "AtrialSwitch", "Rastelli",
    "SingleVentricle", "DILV", "DIDORV", "CommonAtrium", "Glenn", "Fontan",
    "Heterotaxy", "SuperoinferiorVentricles", "PAAtresiaOrMPAStump", "PABanding",
    "AOPAAnastamosis", "Marfan", "CMRArtifactAO", "CMRArtifactPA",
]

# Names for 3D viewer — must match OBJ group names written by mesh_rendering.py
LABEL_NAMES = {
    1: "LV", 2: "RV", 3: "LA", 4: "RA", 5: "Aorta", 6: "PA", 7: "SVC", 8: "IVC",
}


def load_and_prepare(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    # Binarize condition columns: X or 'X' -> 1
    for col in CONDITION_COLS:
        if col in df.columns:
            df[col] = (df[col].fillna("").astype(str).str.strip().str.upper() == "X").astype(int)

    # Fix inf in ratios (e.g. LA_RA when RA=0)
    for col in RATIO_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df.loc[~np.isfinite(df[col]), col] = np.nan

    return df


def get_normal_mask(df: pd.DataFrame) -> pd.Series:
    """Normal = explicitly flagged Normal, and no other defect (optional)."""
    if "Normal" not in df.columns:
        return pd.Series(False, index=df.index)
    # Normal means Normal==1; optionally require no other conditions
    other = [c for c in CONDITION_COLS if c != "Normal" and c in df.columns]
    no_other = (df[other].sum(axis=1) == 0) if other else pd.Series(True, index=df.index)
    return (df["Normal"] == 1) & no_other


def safe_mean(series: pd.Series) -> float:
    s = series.replace([np.inf, -np.inf], np.nan).dropna()
    return float(s.mean()) if len(s) > 0 else np.nan


def safe_std(series: pd.Series) -> float:
    s = series.replace([np.inf, -np.inf], np.nan).dropna()
    return float(s.std()) if len(s) > 1 else 0.0


def compute_reference_and_effects(df: pd.DataFrame) -> Tuple[Dict[str, float], Dict[str, Dict[str, float]], Dict[str, int]]:
    """
    Returns:
      reference: mean feature vector for "normal" cohort
      condition_effects: for each condition, mean feature vector among patients with that condition
      condition_counts: n patients per condition (for reporting)
    """
    normal_mask = get_normal_mask(df)
    ref = {}
    for col in FEATURE_COLS:
        if col in df.columns:
            ref[col] = safe_mean(df.loc[normal_mask, col])
    # If no normals, use global mean
    if not normal_mask.any():
        for col in FEATURE_COLS:
            if col in df.columns:
                ref[col] = safe_mean(df[col])

    condition_effects = {}
    condition_counts = {}
    for cond in CONDITION_COLS:
        if cond not in df.columns:
            continue
        mask = df[cond] == 1
        n = mask.sum()
        condition_counts[cond] = int(n)
        if n < 2:
            condition_effects[cond] = {k: ref.get(k, np.nan) for k in FEATURE_COLS if k in df.columns}
            continue
        effects = {}
        for col in FEATURE_COLS:
            if col not in df.columns:
                continue
            effects[col] = safe_mean(df.loc[mask, col])
        condition_effects[cond] = effects

    return ref, condition_effects, condition_counts


def effect_as_multiplier(ref_val: float, cond_val: float, min_ratio: float = 0.3, max_ratio: float = 3.0) -> float:
    """Volume multiplier: cond_val/ref_val, clamped. For ratios we use same idea."""
    if ref_val is None or np.isnan(ref_val) or ref_val <= 0:
        return 1.0
    if cond_val is None or np.isnan(cond_val):
        return 1.0
    r = cond_val / ref_val
    return float(np.clip(r, min_ratio, max_ratio))


def build_condition_multipliers(
    reference: dict[str, float],
    condition_effects: dict[str, dict[str, float]],
) -> dict[str, dict[str, float]]:
    """For each condition, store multiplicative effect per feature (vs normal)."""
    mults = {}
    for cond, feats in condition_effects.items():
        mults[cond] = {}
        for col, cond_val in feats.items():
            ref_val = reference.get(col)
            mults[cond][col] = effect_as_multiplier(ref_val, cond_val)
    return mults


def estimate_features(
    conditions: list[str],
    reference: dict[str, float],
    condition_multipliers: dict[str, dict[str, float]],
    method: str = "multiplicative",
) -> dict[str, float]:
    """
    Estimate heart feature vector for a set of conditions.

    method:
      - "multiplicative": predicted = reference * (geometric mean of each condition's multiplier).
      - "nearest": not implemented here; use mean of patients with that exact mix if needed.
    """
    out = dict(reference)
    if not conditions:
        return out

    valid = [c for c in conditions if c in condition_multipliers]
    if not valid:
        return out

    if method == "multiplicative":
        # Geometric mean of multipliers across selected conditions (so multiple conditions blend)
        for col in reference:
            if col not in out:
                continue
            factors = [
                condition_multipliers[c].get(col, 1.0)
                for c in valid
            ]
            factors = [f for f in factors if f is not None and np.isfinite(f)]
            if not factors:
                continue
            geom = np.exp(np.mean(np.log(np.array(factors))))
            out[col] = reference[col] * geom
    return out


def scaling_factors_for_viewer(
    estimated: dict[str, float],
    reference: dict[str, float],
) -> dict[str, float]:
    """
    Return per-chamber (and vessel) scaling factors for 3D: (estimated_vol / reference_vol)^(1/3)
    so the viewer can scale each mesh part by this factor.
    """
    scale = {}
    for i in range(1, 9):
        col = f"Label_{i}_vol_ml"
        name = LABEL_NAMES.get(i, col)
        ref_v = reference.get(col)
        est_v = estimated.get(col)
        if ref_v is None or est_v is None or ref_v <= 0:
            scale[name] = 1.0
            continue
        ratio = est_v / ref_v
        ratio = np.clip(ratio, 0.2, 5.0)
        scale[name] = float(ratio ** (1 / 3))
    return scale


def discover_trends(
    reference: dict[str, float],
    condition_effects: dict[str, dict[str, float]],
    condition_counts: dict[str, int],
    min_count: int = 3,
) -> List[Dict[str, Any]]:
    """Produce a list of trend summaries: condition -> notable feature deltas vs normal."""
    trends = []
    for cond, feats in condition_effects.items():
        n = condition_counts.get(cond, 0)
        if n < min_count:
            continue
        diffs = []
        for col in FEATURE_COLS:
            ref_v = reference.get(col)
            c_v = feats.get(col)
            if ref_v is None or c_v is None or np.isnan(ref_v) or ref_v == 0:
                continue
            pct = (c_v - ref_v) / ref_v * 100
            if abs(pct) >= 5:  # at least 5% change
                diffs.append({"feature": col, "ref": ref_v, "with_condition": c_v, "pct_change": pct})
        if diffs:
            trends.append({"condition": cond, "n": n, "deltas": diffs})
    return trends


def run_pipeline(csv_path: str) -> Tuple[pd.DataFrame, Dict, Dict, Dict, Dict]:
    """Load data, compute reference + effects + multipliers. Return all for export/CLI."""
    df = load_and_prepare(csv_path)
    reference, condition_effects, condition_counts = compute_reference_and_effects(df)
    condition_multipliers = build_condition_multipliers(reference, condition_effects)
    return df, reference, condition_effects, condition_counts, condition_multipliers


def export_for_viewer(
    reference: dict[str, float],
    condition_multipliers: dict[str, dict[str, float]],
    condition_counts: dict[str, int],
    out_path: str,
) -> None:
    """Write JSON for the frontend: reference volumes and per-condition multipliers + counts."""
    # Keep only volumes and key ratios for payload size; viewer mainly needs scales
    vol_keys = VOL_COLS + ["Total_heart_vol"]
    ref_export = {k: reference[k] for k in vol_keys if k in reference}
    mult_export = {}
    for cond, mults in condition_multipliers.items():
        mult_export[cond] = {k: mults[k] for k in vol_keys if k in mults}
    payload = {
        "reference": ref_export,
        "condition_multipliers": mult_export,
        "condition_counts": condition_counts,
        "label_names": LABEL_NAMES,
    }
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)


