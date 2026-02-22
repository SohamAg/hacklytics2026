"""
Offline script: extracts per-label volumetric features from NIfTI segmentations
and merges with clinical labels to produce heart_features.csv.

Run once to regenerate the dataset:
    python feature_extraction.py
"""

import os

import nibabel as nib
import numpy as np
import pandas as pd


SEG_PATH = "cropped/cropped/"


if __name__ == "__main__":
    results = []

    for file in os.listdir(SEG_PATH):
        if not file.endswith("_cropped_seg.nii.gz"):
            continue

        label = nib.load(os.path.join(SEG_PATH, file))
        data = label.get_fdata()
        spacing = label.header.get_zooms()
        voxel_volume = spacing[0] * spacing[1] * spacing[2]

        patient_id = int(file.split("_")[0].replace("pat", ""))
        patient_features = {"Patient": patient_id}

        for label_id in range(1, 9):
            voxel_count = np.sum(data == label_id)
            volume_ml = (voxel_count * voxel_volume) / 1000  # mm³ → mL
            patient_features[f"Label_{label_id}_vol_ml"] = volume_ml

        results.append(patient_features)

    df = pd.DataFrame(results)
    df["Total_heart_vol"] = df[[col for col in df.columns if "_vol_ml" in col]].sum(axis=1)
    df["LV_RV_ratio"] = df["Label_1_vol_ml"] / df["Label_2_vol_ml"]
    df["LA_RA_ratio"] = df["Label_3_vol_ml"] / df["Label_4_vol_ml"]
    df["AO_fraction"] = df["Label_5_vol_ml"] / df["Total_heart_vol"]
    df["LV_fraction"] = df["Label_1_vol_ml"] / df["Total_heart_vol"]

    clinical = pd.read_csv("hvsmr_clinical.csv")
    df = df.merge(clinical, left_on="Patient", right_on="Pat")
    df.to_csv("heart_features.csv", index=False)
    print(f"Extracted features for {len(df)} patients -> heart_features.csv")
