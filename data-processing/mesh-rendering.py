import nibabel as nib
import numpy as np
import pyvista as pv
from skimage import measure
import os


def save_segmented_obj(filename, seg_data, spacing, label_info):
    """Save segmentation meshes as separate groups in a single OBJ file with proper centering."""
    
    # First, create all meshes and find bounds for centering
    segment_meshes = {}
    all_verts = []
    
    for label_value, (name, color) in label_info.items():
        binary_mask = (seg_data == label_value).astype(np.uint8)
        
        if np.sum(binary_mask) == 0:
            continue
        
        verts, faces, normals, values = measure.marching_cubes(
            binary_mask,
            level=0.5,
            spacing=spacing
        )
        
        # Store vertices for center calculation
        all_verts.append(verts)
        
        faces_pv = np.hstack(
            [np.full((faces.shape[0], 1), 3), faces]
        ).astype(np.int64)
        
        mesh = pv.PolyData(verts, faces_pv)
        mesh = mesh.smooth(n_iter=30, relaxation_factor=0.1)
        
        # Store both smoothed mesh and original face indices
        segment_meshes[name] = (mesh, faces)
    
    # Calculate center of all vertices
    all_verts_array = np.vstack(all_verts)
    center = np.mean(all_verts_array, axis=0)
    
    # Calculate scale to fit in reasonable size
    bounds = all_verts_array
    max_extent = np.max(np.ptp(bounds, axis=0))
    scale = 10.0 / max_extent if max_extent > 0 else 1.0
    
    # Write OBJ file with groups
    with open(filename, 'w') as f:
        f.write("# Heart segmentation model\n")
        f.write("# Exported from mesh-rendering.py\n\n")
        
        vertex_offset = 0  # Track total vertices written (0-based, will be +1 when used)
        
        for segment_name, (mesh, original_faces) in segment_meshes.items():
            f.write(f"g {segment_name}\n")
            
            # Get mesh data from smoothed mesh
            verts = mesh.points
            
            # Center and scale vertices, flip Z axis for correct orientation
            centered_verts = (verts - center) * scale
            
            # Write vertices for this segment
            for vert in centered_verts:
                # Flip Z coordinate to correct orientation
                f.write(f"v {vert[0]:.6f} {vert[1]:.6f} {-vert[2]:.6f}\n")
            
            # Write faces for this segment using original face indices
            for face in original_faces:
                v1 = face[0] + vertex_offset + 1  # +1 because OBJ is 1-indexed
                v2 = face[1] + vertex_offset + 1
                v3 = face[2] + vertex_offset + 1
                f.write(f"f {v1} {v2} {v3}\n")
            
            vertex_offset += len(verts)
    
    print(f"Mesh saved to {filename} with separate groups and proper centering")


def render_label_mesh(nifti_path, label_value, color="red"):
    """Render a single labeled structure from the segmentation."""
    
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

    # Save mesh with separate groups for each segment
    save_segmented_obj("heart_model_4.obj", seg_data, seg.header.get_zooms(), label_info)
    
    plotter.show()
    
# render_full_heart("cropped/cropped/pat0_cropped_seg.nii.gz")

# Batch process patients 0-58
def batch_process_patients(start_patient=0, end_patient=58, output_folder="heart_models"):
    """Generate OBJ files for multiple patients."""
    
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
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
    
    for patient_id in range(start_patient, end_patient + 1):
        nifti_path = f"cropped/cropped/pat{patient_id}_cropped_seg.nii.gz"
        output_path = os.path.join(output_folder, f"heart_model_pat{patient_id}.obj")
        
        # Check if file exists
        if not os.path.exists(nifti_path):
            print(f"⚠️  Skipping patient {patient_id}: {nifti_path} not found")
            continue
        
        try:
            print(f"Processing patient {patient_id}...")
            
            # Load segmentation file
            seg = nib.load(nifti_path)
            seg_data = seg.get_fdata()
            spacing = seg.header.get_zooms()
            
            # Save OBJ file
            save_segmented_obj(output_path, seg_data, spacing, label_info)
            print(f"✓ Patient {patient_id} saved to {output_path}")
            
        except Exception as e:
            print(f"✗ Error processing patient {patient_id}: {str(e)}")
    
    print(f"\n✓ Batch processing complete! {end_patient - start_patient + 1} patients processed.")
    print(f"✓ OBJ files saved to '{output_folder}' folder")


# Run batch processing
batch_process_patients()

