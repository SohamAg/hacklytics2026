import os
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

path = "cropped/cropped/"
label = nib.load("cropped/cropped/pat1_cropped_seg.nii.gz")
data = label.get_fdata()


# checking the files

# files = [f for f in os.listdir(path) if f.endswith((".nii", ".nii.gz"))]
# files = sorted(files)[:3]
# print("Found files:", files)



# Finding the shapes and checking the valididty

# for file in files:
#     full_path = os.path.join(path, file)

#     label = nib.load(full_path)
#     data = label.get_fdata()

#     print("\nFile:", file)
#     print("Shape:", data.shape)
#     print("Unique labels:", np.unique(data))
    
    

# Showing the depth map

# plt.imshow(data[:, :, 80])
# plt.colorbar()
# plt.show()


# voxel understanding and segregation

# voxel_spacing = label.header.get_zooms()
# print("Voxel spacing:", voxel_spacing)
# for label_id in range(1, 9):
#     count = np.sum(data == label_id)
#     print(f"Label {label_id}: {count} voxels")


# Main loop for volume
results = []
for file in os.listdir(path):
    if file.endswith("_cropped_seg.nii.gz"):
        
        full_path = os.path.join(path, file)
        label = nib.load(full_path)
        data = label.get_fdata()
        
        # Get voxel volume
        spacing = label.header.get_zooms()
        voxel_volume = spacing[0] * spacing[1] * spacing[2]
        
        # Extract patient number
        patient_id = int(file.split("_")[0].replace("pat", ""))
        
        patient_features = {"Patient": patient_id}
        
        # Loop through labels 1–8
        for label_id in range(1, 9):
            voxel_count = np.sum(data == label_id)
            volume_ml = (voxel_count * voxel_volume) / 1000  # convert mm³ → mL
            patient_features[f"Label_{label_id}_vol_ml"] = volume_ml
        
        results.append(patient_features)

# Converting to dataframe, adding other values and columns to top the figured values
df = pd.DataFrame(results)
df["Total_heart_vol"] = df[[col for col in df.columns if "_vol_ml" in col]].sum(axis=1)
df["LV_RV_ratio"] = df["Label_1_vol_ml"] / df["Label_2_vol_ml"]
df["LA_RA_ratio"] = df["Label_3_vol_ml"] / df["Label_4_vol_ml"]
df["AO_fraction"] = df["Label_5_vol_ml"] / df["Total_heart_vol"]
df["LV_fraction"] = df["Label_1_vol_ml"] / df["Total_heart_vol"]

#Combining with the hvsmr clinical data of the diseases
clinical = pd.read_csv("hvsmr_clinical.csv")
df = df.merge(clinical, left_on="Patient", right_on="Pat")
df.to_csv("heart_features.csv", index=False)
# print(df)