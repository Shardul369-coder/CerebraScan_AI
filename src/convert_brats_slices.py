import os
import sys
import logging
from pathlib import Path
import numpy as np
import nibabel as nib
import random

# -------------------------
# Logging Setup
# -------------------------
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

logger = logging.getLogger('data_preprocessing')
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

log_file_path = os.path.join(log_dir, 'data_preprocessing.log')
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel(logging.DEBUG)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

# -------------------------
# MRI Pipeline Config
# -------------------------
RAW_DATA_DIR = Path("raw_data")
OUTPUT_DIR = Path("processed_data")

TRAIN_COUNT = 600
VAL_COUNT = 300
TEST_COUNT = 100

MIN_TUMOR_PIXELS = 10
NORMALIZE = True

MRI_MODALITY_KEYWORD = "t2f"
MASK_KEYWORD = "seg"

# -------------------------
# Utility Functions
# -------------------------
def safe_path(p: Path) -> Path:
    resolved = p.resolve()
    if not resolved.exists():
        logger.error(f"Path does not exist: {resolved}")
        raise FileNotFoundError(f"Path does not exist: {resolved}")
    return resolved

def load_nifti(path: Path) -> np.ndarray:
    try:
        img = nib.load(str(path))
        data = img.get_fdata(dtype=np.float32)
        if data.ndim != 3:
            logger.error(f"Expected 3D volume, got shape {data.shape} for file {path}")
            raise ValueError(f"Expected 3D volume, got {data.shape}")
        return data
    except Exception as e:
        logger.exception(f"Failed to load NIfTI file: {path}")
        raise

def normalize_volume(vol: np.ndarray) -> np.ndarray:
    mean, std = np.mean(vol), np.std(vol)
    if std < 1e-6:
        return np.zeros_like(vol)
    return (vol - mean) / std

def get_25d_slice(volume: np.ndarray, idx: int) -> np.ndarray:
    d = volume.shape[2]
    if idx <= 0:
        slices = (volume[:, :, 0], volume[:, :, 0], volume[:, :, 1])
    elif idx >= d - 1:
        slices = (volume[:, :, d - 2], volume[:, :, d - 1], volume[:, :, d - 1])
    else:
        slices = (volume[:, :, idx - 1], volume[:, :, idx], volume[:, :, idx + 1])
    return np.stack(slices, axis=-1)

def patient_already_processed(patient_id: str, out_img_dir: Path) -> bool:
    exists = any(out_img_dir.glob(f"{patient_id}_slice_*.npy"))
    if exists:
        logger.info(f"[SKIP] Already processed patient {patient_id}")
    return exists

def process_patient(patient_dir: Path, out_img_dir: Path, out_mask_dir: Path):
    nii_files = [p for p in patient_dir.iterdir() if ".nii" in p.name.lower()]

    def find_file(keyword: str) -> Path:
        matches = [p for p in nii_files if keyword in p.name.lower()]
        if len(matches) != 1:
            logger.error(
                f"Expected 1 file for keyword '{keyword}', found {len(matches)} in {patient_dir}"
            )
            raise RuntimeError(
                f"Expected 1 file for '{keyword}', found {len(matches)}"
            )
        return safe_path(matches[0])

    mri_path = find_file(MRI_MODALITY_KEYWORD)
    mask_path = find_file(MASK_KEYWORD)

    volume = load_nifti(mri_path)
    mask = load_nifti(mask_path)

    if NORMALIZE:
        volume = normalize_volume(volume)

    pid = patient_dir.name
    depth = volume.shape[2]

    saved_slices = 0
    for idx in range(depth):
        mask_slice = mask[:, :, idx]
        if np.count_nonzero(mask_slice) < MIN_TUMOR_PIXELS:
            continue

        x = get_25d_slice(volume, idx)
        y = mask_slice.astype(np.uint8)

        np.save(out_img_dir / f"{pid}_slice_{idx:03d}.npy", x)
        np.save(out_mask_dir / f"{pid}_slice_{idx:03d}.npy", y)
        saved_slices += 1

    logger.info(f"Processed patient {pid}: saved {saved_slices} slices")

def prepare_dirs():
    dirs = {
        "train": OUTPUT_DIR / "Train_data",
        "val": OUTPUT_DIR / "Validation_data",
        "test": OUTPUT_DIR / "Test_data",
    }

    for subset, base in dirs.items():
        for sub in ["Images", "Masks"]:
            path = base / sub
            path.mkdir(parents=True, exist_ok=True)

    logger.debug("Output directories prepared successfully.")
    return dirs

# -------------------------
# MAIN PIPELINE
# -------------------------
def main():
    raw_dir = safe_path(RAW_DATA_DIR)

    patient_dirs = sorted(d for d in raw_dir.iterdir() if d.is_dir())
    total = len(patient_dirs)
    logger.info(f"Found {total} patients in raw directory.")

    if total < (TRAIN_COUNT + VAL_COUNT + TEST_COUNT):
        logger.error("Not enough patients to create train/val/test splits!")
        raise RuntimeError("Not enough patients to split!")

    random.shuffle(patient_dirs)

    train_patients = patient_dirs[:TRAIN_COUNT]
    val_patients = patient_dirs[TRAIN_COUNT:TRAIN_COUNT + VAL_COUNT]
    test_patients = patient_dirs[TRAIN_COUNT + VAL_COUNT:TRAIN_COUNT + VAL_COUNT + TEST_COUNT]

    dirs = prepare_dirs()

    subsets = [
        ("train", train_patients),
        ("val", val_patients),
        ("test", test_patients),
    ]

    for subset_name, subset_patients in subsets:
        img_dir = dirs[subset_name] / "Images"
        mask_dir = dirs[subset_name] / "Masks"

        logger.info(f"Processing {subset_name.upper()} set ({len(subset_patients)} patients)")

        for patient in subset_patients:
            pid = patient.name
            if patient_already_processed(pid, img_dir):
                continue

            logger.info(f"Processing patient: {pid}")
            process_patient(patient, img_dir, mask_dir)

    logger.info("SUCCESS: All splits processed safely.")

if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        logger.exception(f"FATAL ERROR: {exc}")
        sys.exit(1)
