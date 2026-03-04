import os
import pyvista as pv
import nibabel as nib
import numpy as np

MESH_DIR = "backend/storage/meshes"
os.makedirs(MESH_DIR, exist_ok=True)

LABELS = {
    1: "NET",
    2: "Edema",
    3: "ET"
}

def generate_meshes(nifti_path, case_id):

    nii = nib.load(nifti_path)
    data = nii.get_fdata()
    spacing = nii.header.get_zooms()[:3]

    for label_value, label_name in LABELS.items():

        mask = (data == label_value).astype(np.uint8)

        if np.sum(mask) == 0:
            continue

        grid = pv.ImageData()
        grid.dimensions = mask.shape
        grid.spacing = spacing
        grid.origin = (0, 0, 0)

        grid.point_data["values"] = mask.flatten(order="F")

        # Extract surface
        surface = grid.contour(isosurfaces=[0.5])

        output_path = os.path.join(
            MESH_DIR,
            f"{case_id}_{label_name}.ply"
        )

        surface.save(output_path)
        print(f"[MESH] Saved {label_name} to {output_path}")