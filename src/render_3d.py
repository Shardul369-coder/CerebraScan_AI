import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import nibabel as nib
from matplotlib.colors import ListedColormap

# ======== MRI MODALITY SELECTION ========
MRI_MODALITY = "flair"  # options: flair, t1, t1ce, t2

# ======== COLOR MAP FOR MULTI-CLASS ========
BRATS_CMAP = ListedColormap([
    (0, 0, 0, 0),      # 0 = background (transparent)
    (0, 0, 1, 0.6),    # 1 = NET/NCR (blue)
    (1, 1, 0, 0.6),    # 2 = Edema (yellow)
    (1, 0, 0, 0.6),    # 3 = ET (red)
])

# ======== MRI FINDER =========
def load_mri_volume(patient_id):
    patient_dir = Path("raw_data") / patient_id

    files = list(patient_dir.iterdir())
    files_l = [f.name.lower() for f in files]

    if MRI_MODALITY == "flair":
        keys = ["t2f"]            # your dataset's FLAIR
    elif MRI_MODALITY == "t1":
        keys = ["t1n"]
    elif MRI_MODALITY == "t1ce":
        keys = ["t1c"]
    elif MRI_MODALITY == "t2":
        keys = ["t2w"]
    else:
        raise ValueError(f"Invalid modality {MRI_MODALITY}")

    # Match files correctly
    for f in files:
        fname = f.name.lower()
        if any(k in fname for k in keys) and fname.endswith(".nii.gz"):
            vol = nib.load(str(f)).get_fdata().astype(np.float32)
            vol = (vol - vol.mean()) / (vol.std() + 1e-8)
            return vol

    raise FileNotFoundError(f"No MRI modality {MRI_MODALITY} found for {patient_id}")

# ========= SLICE VISUALIZATION =========
def save_slice_visualizations(volume_3d, mri_3d, patient_id, output_dir):
    output_dir = Path(output_dir) / patient_id
    output_dir.mkdir(parents=True, exist_ok=True)

    depth = volume_3d.shape[2]
    step = max(1, depth // 10)
    slice_indices = list(range(0, depth, step))

    print(f"  Saving {len(slice_indices)} slices for {patient_id}")

    for i in slice_indices:
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))

        mri_slice = mri_3d[:, :, i]
        mask_slice = volume_3d[:, :, i]

        ax.imshow(mri_slice, cmap="gray")
        ax.imshow(mask_slice, cmap=BRATS_CMAP, vmin=0, vmax=3)
        ax.set_title(f"{patient_id} - Slice {i}/{depth}")
        ax.axis("off")

        plt.tight_layout()
        plt.savefig(output_dir / f"slice_{i:03d}.png", dpi=150, bbox_inches="tight")
        plt.close()

    return len(slice_indices)

# ========= 3D MIP PROJECTION =========
def create_3d_projection(volume_3d, patient_id, output_dir):
    output_dir = Path(output_dir) / patient_id
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    mip_axial = np.max(volume_3d, axis=2)
    mip_coronal = np.max(volume_3d, axis=1)
    mip_sagittal = np.max(volume_3d, axis=0)

    axes[0].imshow(mip_axial, cmap=BRATS_CMAP, vmin=0, vmax=3)
    axes[0].set_title("Axial MIP"); axes[0].axis("off")

    axes[1].imshow(mip_coronal.T, cmap=BRATS_CMAP, vmin=0, vmax=3)
    axes[1].set_title("Coronal MIP"); axes[1].axis("off")

    axes[2].imshow(mip_sagittal.T, cmap=BRATS_CMAP, vmin=0, vmax=3)
    axes[2].set_title("Sagittal MIP"); axes[2].axis("off")

    plt.tight_layout()
    plt.savefig(output_dir / "3d_projection.png", dpi=150, bbox_inches="tight")
    plt.close()

# ========= SUMMARY =========
def create_patient_summary(volume_3d, patient_id, output_dir):
    output_dir = Path(output_dir) / patient_id
    output_dir.mkdir(parents=True, exist_ok=True)

    tumor_mask = volume_3d > 0
    tumor_voxels = np.sum(tumor_mask)
    total_voxels = volume_3d.size
    tumor_percentage = (tumor_voxels / total_voxels) * 100

    fig = plt.figure(figsize=(12, 8))
    ax = plt.subplot(2, 2, 1)
    ax.imshow(np.max(volume_3d, axis=2), cmap=BRATS_CMAP, vmin=0, vmax=3)
    ax.set_title("Axial MIP"); ax.axis("off")

    mid = volume_3d.shape[2] // 2
    ax = plt.subplot(2, 2, 2)
    ax.imshow(volume_3d[:, :, mid], cmap=BRATS_CMAP, vmin=0, vmax=3)
    ax.set_title("Middle Slice"); ax.axis("off")

    ax = plt.subplot(2, 2, 3)
    ax.text(0.1, 0.8, f"Patient: {patient_id}", fontsize=12)
    ax.text(0.1, 0.6, f"Tumor Voxels: {tumor_voxels:,}", fontsize=12)
    ax.text(0.1, 0.4, f"Total Voxels: {total_voxels:,}", fontsize=12)
    ax.text(0.1, 0.2, f"Tumor %: {tumor_percentage:.2f}%", fontsize=12)
    ax.axis("off")

    plt.tight_layout()
    plt.savefig(output_dir / "summary.png", dpi=150, bbox_inches="tight")
    plt.close()

    return tumor_voxels, tumor_percentage

# ========= MAIN DRIVER =========
def render_all_patients(pred_dir="predictions_3d", output_dir="visualizations"):
    pred_dir = Path(pred_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    # Ignore probability volumes
    patients = sorted(pred_dir.glob("*_slice_3d.npy"))

    print(f"[INFO] Found {len(patients)} patients to visualize")

    for pred_file in patients:
        patient_id = pred_file.stem.replace("_slice_3d", "")

        print(f"[PROCESSING] {patient_id}")

        volume_3d = np.load(pred_file)
        mri_3d = load_mri_volume(patient_id)

        save_slice_visualizations(volume_3d, mri_3d, patient_id, output_dir)
        create_3d_projection(volume_3d, patient_id, output_dir)
        create_patient_summary(volume_3d, patient_id, output_dir)

if __name__ == "__main__":
    render_all_patients()
    print("✅ Visualization complete!")
