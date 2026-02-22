from fastapi import APIRouter, UploadFile, File
from pathlib import Path
import uuid
import zipfile

router = APIRouter()

UPLOAD_DIR = Path("backend/storage/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


@router.post("/upload")
async def upload_scan(file: UploadFile = File(...)):

    case_id = str(uuid.uuid4())
    case_folder = UPLOAD_DIR / case_id
    case_folder.mkdir(parents=True, exist_ok=True)

    zip_path = case_folder / file.filename

    # save zip
    with open(zip_path, "wb") as f:
        f.write(await file.read())

    # extract zip
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(case_folder)

    return {
        "case_id": case_id,
        "message": "Upload and extraction successful"
    }