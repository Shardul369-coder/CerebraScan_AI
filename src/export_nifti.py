import numpy as np
import nibabel as nib
from pathlib import Path

def export_nifti(volume_path, out_dir="nifti_outputs"):
    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True)

    volume = np.load(volume_path).astype(np.uint8)
    nii = nib.Nifti1Image(volume, affine=np.eye(4))

    out_path = out_dir / (Path(volume_path).stem + ".nii.gz")
    nib.save(nii, out_path)
    print(f"[SAVED] {out_path}")
    return out_path

if __name__ == "__main__":
    export_nifti("predictions_3d/patient_001_3d.npy")
