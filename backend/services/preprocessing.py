from pathlib import Path
import shutil

from src.convert_backend import convert_case

BASE_DIR = Path("backend/storage/images")


def prepare_for_inference(case_id, upload_dir):

    case_dest = BASE_DIR / case_id

    if case_dest.exists():
        shutil.rmtree(case_dest)

    case_dest.mkdir(parents=True, exist_ok=True)

    convert_case(upload_dir, case_dest)

    return {
        "status": "ready",
        "destination": str(case_dest),
    }