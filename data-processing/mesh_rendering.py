"""
Offline script: exports segmented heart meshes as OBJ files from NIfTI segmentations.

Run once to generate mesh assets:
    python mesh_rendering.py
"""

import os
from pathlib import Path

import nibabel as nib
import numpy as np
import pyvista as pv
from skimage import measure

PROJECT_MARKERS = [
    Path('heart_models'),
    Path('data-processing'),
    Path('data-processing'),
]

def find_project_root(start: Path) -> Path:
    cur = start.resolve()
    for parent in [cur] + list(cur.parents):
        if all((parent / m).exists() for m in PROJECT_MARKERS):
            return parent
    raise FileNotFoundError('Could not locate project root from ' + str(start))

PROJECT_ROOT = find_project_root(Path.cwd())


LABEL_INFO = {
    1: ("LV", "red"),
    2: ("RV", "blue"),
    3: ("LA", "pink"),
    4: ("RA", "purple"),
    5: ("Aorta", "yellow"),
    6: ("PA", "green"),
    7: ("SVC", "orange"),
    8: ("IVC", "brown"),
}


def save_segmented_obj(filename, seg_data, spacing, label_info):
    """Save segmentation meshes as separate groups in a single OBJ file with proper centering."""

    segment_meshes = {}
    all_verts = []

    for label_value, (name, color) in label_info.items():
        binary_mask = (seg_data == label_value).astype(np.uint8)

        if np.sum(binary_mask) == 0:
            continue

        verts, faces, normals, values = measure.marching_cubes(
            binary_mask,
            level=0.5,
            spacing=spacing,
        )

        all_verts.append(verts)

        faces_pv = np.hstack(
            [np.full((faces.shape[0], 1), 3), faces]
        ).astype(np.int64)

        mesh = pv.PolyData(verts, faces_pv)
        mesh = mesh.smooth(n_iter=30, relaxation_factor=0.1)

        segment_meshes[name] = (mesh, faces)

    all_verts_array = np.vstack(all_verts)
    center = np.mean(all_verts_array, axis=0)

    bounds = all_verts_array
    max_extent = np.max(np.ptp(bounds, axis=0))
    scale = 10.0 / max_extent if max_extent > 0 else 1.0

    with open(filename, "w") as f:
        f.write("# Heart segmentation model\n")
        f.write("# Exported from mesh_rendering.py\n\n")

        vertex_offset = 0

        for segment_name, (mesh, original_faces) in segment_meshes.items():
            f.write(f"g {segment_name}\n")

            verts = mesh.points
            centered_verts = (verts - center) * scale

            for vert in centered_verts:
                f.write(f"v {vert[0]:.6f} {vert[1]:.6f} {vert[2]:.6f}\n")

            for face in original_faces:
                v1 = face[0] + vertex_offset + 1
                v2 = face[1] + vertex_offset + 1
                v3 = face[2] + vertex_offset + 1
                f.write(f"f {v1} {v2} {v3}\n")

            vertex_offset += len(verts)

    print(f"Mesh saved to {filename}")


def render_label_mesh(nifti_path, label_value, color="red"):
    """Render a single labeled structure from the segmentation."""

    seg = nib.load(nifti_path)
    seg_data = seg.get_fdata()
    spacing = seg.header.get_zooms()

    print("Voxel spacing:", spacing)

    binary_mask = (seg_data == label_value).astype(np.uint8)

    if np.sum(binary_mask) == 0:
        print("No voxels found for this label.")
        return

    verts, faces, normals, values = measure.marching_cubes(
        binary_mask,
        level=0.5,
        spacing=spacing,
    )

    faces = np.hstack(
        [np.full((faces.shape[0], 1), 3), faces]
    ).astype(np.int64)

    mesh = pv.PolyData(verts, faces)

    plotter = pv.Plotter()
    plotter.add_mesh(mesh, color=color, opacity=0.9)
    plotter.add_axes()
    plotter.show()


def render_polished_heart(nifti_path):
    """Interactive polished render with per-segment picking."""

    seg = nib.load(nifti_path)
    seg_data = seg.get_fdata()
    spacing = seg.header.get_zooms()

    plotter = pv.Plotter()
    actors = {}

    for label_value, (name, color) in LABEL_INFO.items():
        binary_mask = (seg_data == label_value).astype(np.uint8)

        if np.sum(binary_mask) == 0:
            continue

        verts, faces, normals, values = measure.marching_cubes(
            binary_mask,
            level=0.5,
            spacing=spacing,
        )

        faces = np.hstack(
            [np.full((faces.shape[0], 1), 3), faces]
        ).astype(np.int64)

        mesh = pv.PolyData(verts, faces)
        mesh = mesh.smooth(n_iter=30, relaxation_factor=0.1)

        actor = plotter.add_mesh(
            mesh,
            color=color,
            opacity=0.6,
            specular=0.5,
            specular_power=15,
            smooth_shading=True,
            name=name,
        )

        actors[name] = actor
        plotter.add_mesh(mesh, style="wireframe", color="black", opacity=0.1)

    plotter.add_axes()
    plotter.enable_lightkit()

    def on_pick(mesh):
        if mesh is None:
            return
        picked_name = mesh.actor.GetName()
        for name, actor in actors.items():
            actor.prop.opacity = 1.0 if name == picked_name else 0.2
        plotter.render()

    plotter.enable_mesh_picking(callback=on_pick, use_actor=True)
    output_path = PROJECT_ROOT / 'heart_models' / 'heart_model.obj'
    save_segmented_obj(str(output_path), seg_data, seg.header.get_zooms(), LABEL_INFO)
    plotter.show()


def batch_process_patients(start_patient=0, end_patient=58, output_folder=None):
    """Generate OBJ files for multiple patients."""

    if output_folder is None:
        output_folder = PROJECT_ROOT / 'heart_models' / 'patient_models'
    else:
        output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    for patient_id in range(start_patient, end_patient + 1):
        nifti_path = PROJECT_ROOT / 'data-processing' / 'cropped_data' / 'cropped' / f'pat{patient_id}_cropped_seg.nii.gz'
        output_path = output_folder / f'heart_model_pat{patient_id}.obj'

        if not nifti_path.exists():
            print(f"Skipping patient {patient_id}: {nifti_path} not found")
            continue

        try:
            print(f"Processing patient {patient_id}...")
            seg = nib.load(nifti_path)
            save_segmented_obj(str(output_path), seg.get_fdata(), seg.header.get_zooms(), LABEL_INFO)
            print(f"Patient {patient_id} saved to {output_path}")
        except Exception as e:
            print(f"Error processing patient {patient_id}: {e}")

    print(f"\nBatch complete. OBJ files saved to '{output_folder}'")


if __name__ == "__main__":
    batch_process_patients()
