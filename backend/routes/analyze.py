from fastapi import APIRouter
from pathlib import Path
from backend.services.pipeline import run_full_pipeline

router = APIRouter()

UPLOAD_DIR = Path("backend/storage/uploads")


@router.post("/analyze/{case_id}")
def analyze(case_id: str):

    case_root = UPLOAD_DIR / case_id

    if not case_root.exists():
        return {"error": "case not found"}

    # 🔥 find patient folder automatically
    subfolders = [f for f in case_root.iterdir() if f.is_dir()]

    if len(subfolders) == 0:
        return {"error": "no patient folder found"}

    patient_folder = subfolders[0]

    result = run_full_pipeline(case_id, str(patient_folder))

    return result