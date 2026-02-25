from pathlib import Path
import numpy as np
import nibabel as nib


def load_nifti(p):
    return nib.load(str(p)).get_fdata().astype(np.float32)


def zscore(vol):
    std = vol.std()
    if std < 1e-6:
        return np.zeros_like(vol)
    return (vol - vol.mean()) / (std + 1e-8)


def find_file(folder, keys):
    """
    Search recursively inside upload folder
    """
    for f in Path(folder).rglob("*.nii*"):
        name = f.name.lower()
        if any(k in name for k in keys):
            return f
    return None

def convert_case(upload_dir: str, output_dir: str):
    """
    Convert ONE uploaded case → npy slices for inference
    """

    upload_dir = Path(upload_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ======================
    # find modalities
    # ======================
    flair = find_file(upload_dir, ["flair", "t2f"])
    t1 = find_file(upload_dir, ["t1n", "-t1"])
    t1ce = find_file(upload_dir, ["t1c", "t1ce"])
    t2 = find_file(upload_dir, ["t2w", "-t2"])

    if not all([flair, t1, t1ce, t2]):
        raise RuntimeError("Missing modalities during conversion")

    print("[CONVERT] Loading modalities...")

    flair = zscore(load_nifti(flair))
    t1 = zscore(load_nifti(t1))
    t1ce = zscore(load_nifti(t1ce))
    t2 = zscore(load_nifti(t2))

    volume = np.stack([flair, t1, t1ce, t2], axis=-1)

    depth = volume.shape[2]

    print(f"[CONVERT] Saving {depth} slices")

    for i in range(depth):
        x = volume[:, :, i, :]
        np.save(output_dir / f"patient_slice_{i:03d}.npy", x)

    print("[CONVERT] DONE")