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

logger = logging.getLogger('data_preprocessing_3d')
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

log_file_path = os.path.join(log_dir, 'data_preprocessing_3d.log')
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel(logging.DEBUG)

formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

# -------------------------
# Config
# -------------------------
RAW_DATA_DIR = Path("raw_data")
OUTPUT_DIR = Path("processed_data")

TRAIN_COUNT = 600
VAL_COUNT = 300
TEST_COUNT = 100

MODALITY_KEYS = {
    "flair": "flair",
    "t1": "t1",
    "t1ce": "t1ce",
    "t2": "t2"
}
MASK_KEYWORD = "seg"

NORMALIZE = True

# -------------------------
# Utils
# -------------------------
def safe_path(p: Path) -> Path:
    resolved = p.resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Path does not exist: {resolved}")
    return resolved

def load_nii(path: Path):
    img = nib.load(str(path))
    data = img.get_fdata(dtype=np.float32)
    return data, img

def zscore(vol: np.ndarray) -> np.ndarray:
    mean, std = vol.mean(), vol.std()
    if std < 1e-8:
        return np.zeros_like(vol)
    return (vol - mean) / std

def prepare_dirs():
    subsets = ["Train_data", "Validation_data", "Test_data"]
    subdirs = ["Images", "Masks"]
    for subset in subsets:
        for sub in subdirs:
            path = OUTPUT_DIR / subset / sub
            path.mkdir(parents=True, exist_ok=True)
    logger.info("Output directory structure ready.")

# -------------------------
# Core Processing
# -------------------------
def process_patient(patient_dir: Path, img_out_dir: Path, mask_out_dir: Path):

    pid = patient_dir.name
    img_out_path = img_out_dir / f"{pid}.npz"
    mask_out_path = mask_out_dir / f"{pid}_seg.npz"

    # ---- SKIP IF ALREADY DONE ----
    if img_out_path.exists() and mask_out_path.exists():
        logger.info(f"[SKIP] {pid} already processed.")
        return

    nii_files = list(patient_dir.glob("*.nii.gz"))

    def find_file(key: str) -> Path:
        for f in nii_files:
            if key in f.name.lower():
                return safe_path(f)
        raise RuntimeError(f"Missing modality '{key}' in {patient_dir}")

    # Load modalities
    vols = []
    affine = None
    header = None

    for k, v in MODALITY_KEYS.items():
        p = find_file(v)
        vol, img = load_nii(p)
        if NORMALIZE:
            vol = zscore(vol)
        vols.append(vol)
        affine = img.affine
        header = img.header

    # Stack (H, W, D, 4)
    volume = np.stack(vols, axis=-1)

    # Load mask
    mask_path = find_file(MASK_KEYWORD)
    mask, _ = load_nii(mask_path)
    mask = mask.astype(np.uint8)

    # Save outputs
    np.savez_compressed(img_out_path,
                        volume=volume,
                        affine=affine,
                        header=np.array(header.structarr, dtype=object))

    np.savez_compressed(mask_out_path,
                        mask=mask)

    logger.info(f"[OK] Processed {pid}")

# -------------------------
# MAIN
# -------------------------
def main():
    raw_dir = safe_path(RAW_DATA_DIR)
    patient_dirs = sorted(d for d in raw_dir.iterdir() if d.is_dir())
    total = len(patient_dirs)
    logger.info(f"Found {total} patients in raw_data")

    if total < (TRAIN_COUNT + VAL_COUNT + TEST_COUNT):
        raise RuntimeError("Not enough patients to split!")

    random.shuffle(patient_dirs)

    train_patients = patient_dirs[:TRAIN_COUNT]
    val_patients = patient_dirs[TRAIN_COUNT:TRAIN_COUNT + VAL_COUNT]
    test_patients = patient_dirs[TRAIN_COUNT + VAL_COUNT:TRAIN_COUNT + VAL_COUNT + TEST_COUNT]

    prepare_dirs()

    config = [
        ("Train_data", train_patients),
        ("Validation_data", val_patients),
        ("Test_data", test_patients),
    ]

    for subset_name, patients in config:
        img_dir = OUTPUT_DIR / subset_name / "Images"
        mask_dir = OUTPUT_DIR / subset_name / "Masks"

        logger.info(f"Processing {subset_name} ({len(patients)} patients)")
        for p in patients:
            process_patient(p, img_dir, mask_dir)

    logger.info("SUCCESS: All data processed for 3D U-Net.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.exception(f"FATAL ERROR: {e}")
        sys.exit(1)
