from fastapi import APIRouter
from pathlib import Path
import json

router = APIRouter()

RESULT_DIR = Path("backend/storage/results")


@router.get("/results/{case_id}")
def get_results(case_id: str):

    result_file = RESULT_DIR / f"{case_id}.json"

    if not result_file.exists():
        return {"error": "Result not found"}

    with open(result_file, "r") as f:
        data = json.load(f)

    return data