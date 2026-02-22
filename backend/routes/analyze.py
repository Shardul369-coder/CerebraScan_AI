from fastapi import APIRouter
from pathlib import Path
from backend.services.pipeline import run_full_pipeline

router = APIRouter()

UPLOAD_DIR = Path("backend/storage/uploads")


@router.post("/analyze/{case_id}")
def analyze(case_id: str):

    input_dir = UPLOAD_DIR / case_id

    if not input_dir.exists():
        return {"error": "case not found"}

    result = run_full_pipeline(case_id, str(input_dir))

    return result