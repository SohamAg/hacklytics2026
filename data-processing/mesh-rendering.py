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

#polished version of the render heart
def render_polished_heart(nifti_path):

    seg = nib.load(nifti_path)
    seg_data = seg.get_fdata()
    spacing = seg.header.get_zooms()

    label_info = {
        1: ("LV", "red"),
        2: ("RV", "blue"),
        3: ("LA", "pink"),
        4: ("RA", "purple"),
        5: ("Aorta", "yellow"),
        6: ("PA", "green"),
        7: ("SVC", "orange"),
        8: ("IVC", "brown")
    }

    plotter = pv.Plotter()
    actors = {}

    for label_value, (name, color) in label_info.items():

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

       #smoothening
        mesh = mesh.smooth(n_iter=30, relaxation_factor=0.1)

        actor = plotter.add_mesh(
            mesh,
            color=color,
            opacity=0.6,
            specular=0.5,
            specular_power=15,
            smooth_shading=True,
            name=name
        )

        actors[name] = actor

        #add definition to the edges
        plotter.add_mesh(mesh, style="wireframe", color="black", opacity=0.1)

    plotter.add_axes()
    plotter.enable_lightkit()

    #Hovering interaction (currently works on right click)
    def on_pick(mesh):
        if mesh is None:
            return

        picked_name = mesh.actor.GetName()

        for name, actor in actors.items():
            if name == picked_name:
                actor.prop.opacity = 1.0
            else:
                actor.prop.opacity = 0.2

        plotter.render()

    plotter.enable_mesh_picking(callback=on_pick, use_actor=True)

    plotter.show()
    
# render_full_heart("cropped/cropped/pat0_cropped_seg.nii.gz")
render_polished_heart("cropped/cropped/pat0_cropped_seg.nii.gz")

