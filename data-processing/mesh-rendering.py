import nibabel as nib
import numpy as np
import pyvista as pv
from skimage import measure


def render_label_mesh(nifti_path, label_value, color="red"):

    # Load segmentation file
    seg = nib.load(nifti_path)
    seg_data = seg.get_fdata()

    # Get voxel spacing (critical for correct scaling)
    spacing = seg.header.get_zooms()

    print("Voxel spacing:", spacing)

    # Extract chosen structure
    binary_mask = (seg_data == label_value).astype(np.uint8)

    if np.sum(binary_mask) == 0:
        print("No voxels found for this label.")
        return

    # Marching cubes surface extraction
    verts, faces, normals, values = measure.marching_cubes(
        binary_mask,
        level=0.5,
        spacing=spacing
    )

    # Convert faces for PyVista format
    faces = np.hstack(
        [np.full((faces.shape[0], 1), 3), faces]
    ).astype(np.int64)

    mesh = pv.PolyData(verts, faces)

    # Render
    plotter = pv.Plotter()
    plotter.add_mesh(mesh, color=color, opacity=0.9)
    plotter.add_axes()
    plotter.show()

def render_full_heart(nifti_path):

    seg = nib.load(nifti_path)
    seg_data = seg.get_fdata()
    spacing = seg.header.get_zooms()

    print("Voxel spacing:", spacing)

    
    label_colors = {
        1: "red",        #LV
        2: "blue",       # RV
        3: "pink",       #LA
        4: "purple",     # RA
        5: "yellow",     # Aorta
        6: "green",      # Pulmonary Artery
        7: "orange",     #SVC
        8: "brown"       # IVC
    }

    plotter = pv.Plotter()

    for label_value, color in label_colors.items():

        binary_mask = (seg_data == label_value).astype(np.uint8)

        if np.sum(binary_mask) == 0:
            continue

        verts, faces, normals, values = measure.marching_cubes(
            binary_mask,
            level=0.5,
            spacing=spacing
        )

        faces = np.hstack(
            [np.full((faces.shape[0], 1), 3), faces]
        ).astype(np.int64)

        mesh = pv.PolyData(verts, faces)

        plotter.add_mesh(mesh, color=color, opacity=0.9)

    plotter.add_axes()
    plotter.show()
    
# render_label_mesh(
#     "cropped/cropped/pat0_cropped_seg.nii.gz",
#     label_value=8,  # 1 = LV
#     color="red"
# )
render_full_heart("cropped/cropped/pat0_cropped_seg.nii.gz")

