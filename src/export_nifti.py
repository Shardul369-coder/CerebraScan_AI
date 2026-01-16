import numpy as np
import nibabel as nib
from pathlib import Path

def export_nifti(volume_path, out_dir="nifti_outputs"):
    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True)

    volume = np.load(volume_path).astype(np.uint8)
    affine = np.load("affines/patient_001_affine.npy")
    nii = nib.Nifti1Image(volume, affine=affine)

    out_path = out_dir / (Path(volume_path).stem + ".nii.gz")
    nib.save(nii, out_path)
    print(f"[SAVED] {out_path}")
    return out_path

def batch_export(pred_dir="predictions_3d", out_dir="nifti_outputs"):
    pred_dir = Path(pred_dir)
    for p in pred_dir.glob("*.npy"):
        export_nifti(str(p), out_dir)

if __name__ == "__main__":
    batch_export()
