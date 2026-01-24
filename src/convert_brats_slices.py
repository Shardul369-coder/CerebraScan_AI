import os
import sys
import logging
import random
from pathlib import Path

import numpy as np
import nibabel as nib

# =====================================
# Config
# =====================================
RAW_DATA_DIR = Path("raw_data")
OUTPUT_DIR = Path("processed_data")

TRAIN_COUNT = 600
VAL_COUNT = 300
TEST_COUNT = 100

MIN_TUMOR_PIXELS = 10
NORMALIZE = True

# Figshare modality mapping
MODALITY_MAP = {
    "flair": ["fl.", "t2f"],  # FLAIR / T2-FLAIR
    "t1": ["t1n"],            # T1 native
    "t1ce": ["t1c"],          # T1 post-contrast
    "t2": ["t2w"],            # T2 weighted
}

MASK_KEY = "seg"  # segmentation mask keyword

# =====================================
# Utils
# =====================================
def safe_path(p: Path):
    p = p.resolve()
    if not p.exists():
        print(f"Path does not exist: {p}")
        raise FileNotFoundError(f"Path does not exist: {p}")
    return p

def load_nifti(path: Path) -> np.ndarray:
    try:
        img = nib.load(str(path))
        data = img.get_fdata(dtype=np.float32)
        if data.ndim != 3:
            raise ValueError(f"Expected 3D, got {data.shape}")
        return data
    except Exception as e:
        print(f"Failed to load NIfTI: {path}")
        raise e

def zscore_normalize(vol: np.ndarray) -> np.ndarray:
    mean = vol.mean()
    std = vol.std()
    if std < 1e-6:
        return np.zeros_like(vol)
    return (vol - mean) / (std + 1e-8)

def find_modality(patient_dir: Path, keys: list):
    """Find NIfTI matching any key in list."""
    for k in keys:
        file = next((f for f in patient_dir.iterdir()
                     if k in f.name.lower() and f.suffix in [".nii", ".nii.gz", ".gz"]), None)
        if file:
            return load_nifti(file).astype(np.float32)
    raise FileNotFoundError(f"Modality {keys} not found in {patient_dir}")

def find_mask(patient_dir: Path):
    mask = next((f for f in patient_dir.iterdir()
                 if MASK_KEY in f.name.lower() and f.suffix in [".nii", ".nii.gz", ".gz"]), None)
    if not mask:
        raise FileNotFoundError(f"No mask found in {patient_dir}")
    return load_nifti(mask).astype(np.uint8)

def already_done(pid: str, img_dir: Path):
    exists = any(img_dir.glob(f"{pid}_slice_*.npy"))
    if exists:
        print(f"[SKIP] {pid} already processed")
        return exists

# =====================================
# Core Processing
# =====================================
def process_patient(patient_dir: Path, img_dir: Path, mask_dir: Path):
    pid = patient_dir.name

    # load modalities
    flair = find_modality(patient_dir, MODALITY_MAP["flair"])
    t1 = find_modality(patient_dir, MODALITY_MAP["t1"])
    t1ce = find_modality(patient_dir, MODALITY_MAP["t1ce"])
    t2 = find_modality(patient_dir, MODALITY_MAP["t2"])

    # load mask
    mask = find_mask(patient_dir)

    # normalize
    if NORMALIZE:
        flair = zscore_normalize(flair)
        t1 = zscore_normalize(t1)
        t1ce = zscore_normalize(t1ce)
        t2 = zscore_normalize(t2)

    # stack: (H, W, D, 4)
    volume = np.stack([flair, t1, t1ce, t2], axis=-1)

    depth = volume.shape[2]
    saved = 0

    for i in range(depth):
        m = mask[:, :, i].astype(np.int32)
        if np.count_nonzero(m) < MIN_TUMOR_PIXELS:
            continue

        # Remap BRATS labels: 4 → 3 (ET)
        m[m == 4] = 3
        x = volume[:, :, i, :].astype(np.float32)
        y = m.copy()

        np.save(img_dir / f"{pid}_slice_{i:03d}.npy", x)
        np.save(mask_dir / f"{pid}_slice_{i:03d}.npy", y)
        saved += 1

    print(f"[OK] {pid}: saved {saved} slices")

# =====================================
# Directory Setup
# =====================================
def setup_dirs():
    dirs = {
        "train": OUTPUT_DIR / "Train_data",
        "val": OUTPUT_DIR / "Validation_data",
        "test": OUTPUT_DIR / "Test_data"
    }
    for split, base in dirs.items():
        (base / "Images").mkdir(parents=True, exist_ok=True)
        (base / "Masks").mkdir(parents=True, exist_ok=True)
    return dirs

# =====================================
# Main
# =====================================
def main():
    raw = safe_path(RAW_DATA_DIR)
    patients = sorted([d for d in raw.iterdir() if d.is_dir()])
    total = len(patients)
    print(f"Found {total} patients")

    if total < TRAIN_COUNT + VAL_COUNT + TEST_COUNT:
        raise RuntimeError("Not enough patients")

    random.shuffle(patients)

    train = patients[:TRAIN_COUNT]
    val = patients[TRAIN_COUNT:TRAIN_COUNT+VAL_COUNT]
    test = patients[TRAIN_COUNT+VAL_COUNT:TRAIN_COUNT+VAL_COUNT+TEST_COUNT]

    dirs = setup_dirs()

    for split, plist in [("train", train), ("val", val), ("test", test)]:
        img_dir = dirs[split] / "Images"
        mask_dir = dirs[split] / "Masks"
        print(f"Processing {split.upper()} ({len(plist)} patients)")
        for p in plist:
            if not already_done(p.name, img_dir):
                process_patient(p, img_dir, mask_dir)

    print("SUCCESS: Preprocessing complete!")

# =====================================
# Entry
# =====================================
if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"FATAL ERROR: {exc}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
