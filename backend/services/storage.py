"""
Storage service — manages all filesystem paths used by the pipeline.

Directory layout inside backend/storage/:
  uploads/<upload_id>/          raw NIfTI files (flair, t1, t1ce, t2)
  results/<upload_id>/
      slices/                   preprocessed .npy slices
      predictions/              3D mask + prob .npy files
      nifti/                    exported .nii.gz segmentation
      visualizations/           overlay PNGs
"""

import shutil
import uuid
from pathlib import Path
from typing import List, Optional

# ── Root anchored at backend/storage ──────────────────────────────────────────
STORAGE_ROOT = Path(__file__).parent.parent / "storage"
UPLOAD_ROOT  = STORAGE_ROOT / "uploads"
RESULT_ROOT  = STORAGE_ROOT / "results"

MODALITY_KEYS = ["flair", "t1", "t1ce", "t2"]   # expected upload field names


def new_upload_id() -> str:
    return uuid.uuid4().hex


# ── Path helpers ───────────────────────────────────────────────────────────────

def upload_dir(upload_id: str) -> Path:
    p = UPLOAD_ROOT / upload_id
    p.mkdir(parents=True, exist_ok=True)
    return p


def result_dir(upload_id: str, sub: str = "") -> Path:
    p = RESULT_ROOT / upload_id / sub if sub else RESULT_ROOT / upload_id
    p.mkdir(parents=True, exist_ok=True)
    return p


def slices_dir(upload_id: str) -> Path:
    return result_dir(upload_id, "slices")


def predictions_dir(upload_id: str) -> Path:
    return result_dir(upload_id, "predictions")


def nifti_dir(upload_id: str) -> Path:
    return result_dir(upload_id, "nifti")


def viz_dir(upload_id: str) -> Path:
    return result_dir(upload_id, "visualizations")


# ── File helpers ───────────────────────────────────────────────────────────────

def save_upload(upload_id: str, modality: str, tmp_path: Path) -> Path:
    """Move an uploaded temp file to the upload directory."""
    dest = upload_dir(upload_id) / f"{modality}.nii.gz"
    shutil.move(str(tmp_path), str(dest))
    return dest


def get_modality_path(upload_id: str, modality: str) -> Optional[Path]:
    """Return path to a stored modality file, or None if missing."""
    p = upload_dir(upload_id) / f"{modality}.nii.gz"
    return p if p.exists() else None


def list_uploaded_modalities(upload_id: str) -> List[str]:
    d = upload_dir(upload_id)
    return [f.stem.replace(".nii", "") for f in d.glob("*.nii.gz")]


def list_result_slices(upload_id: str) -> List[Path]:
    return sorted(slices_dir(upload_id).glob("*.npy"))


def prediction_mask_path(upload_id: str, patient_id: str) -> Path:
    return predictions_dir(upload_id) / f"{patient_id}_3d.npy"


def prediction_prob_path(upload_id: str, patient_id: str) -> Path:
    return predictions_dir(upload_id) / f"{patient_id}_probs_3d.npy"


def nifti_output_path(upload_id: str, patient_id: str) -> Path:
    return nifti_dir(upload_id) / f"{patient_id}_3d.nii.gz"


def cleanup_upload(upload_id: str) -> None:
    """Delete all stored files for a given upload (use with caution)."""
    for root in [upload_dir(upload_id), result_dir(upload_id)]:
        if root.exists():
            shutil.rmtree(root)