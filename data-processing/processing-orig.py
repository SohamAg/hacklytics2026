import os
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt

path = "orig/orig/"
label = nib.load("orig/orig/pat1_orig_seg.nii.gz")
data = label.get_fdata()



# files = [f for f in os.listdir(path) if f.endswith((".nii", ".nii.gz"))]
# files = sorted(files)[:3]

# print("Found files:", files)

# for file in files:
#     full_path = os.path.join(path, file)

#     label = nib.load(full_path)
#     data = label.get_fdata()

#     print("\nFile:", file)
#     print("Shape:", data.shape)
#     print("Unique labels:", np.unique(data))
    
    
# plt.imshow(data[:, :, 80])
# plt.colorbar()
# plt.show()

voxel_spacing = label.header.get_zooms()
print("Voxel spacing:", voxel_spacing)

# label = nib.load("orig/orig/pat0_orig_seg.nii.gz")
# data = label.get_fdata()

# for label_id in range(1, 9):
#     count = np.sum(data == label_id)
#     print(f"Label {label_id}: {count} voxels")