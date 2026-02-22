import shutil
from pathlib import Path

DEST_DIR = Path("processed_data/Test_data/Images")


def prepare_for_inference(case_id, upload_dir):

    DEST_DIR.mkdir(parents=True, exist_ok=True)

    copied = []

    # copy all .npy slices
    for f in Path(upload_dir).glob("*.npy"):
        target = DEST_DIR / f.name
        shutil.copy(f, target)
        copied.append(str(target))

    return {
        "status": "ready",
        "total_files": len(copied),
        "destination": str(DEST_DIR)
    }