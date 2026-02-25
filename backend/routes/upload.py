from fastapi import APIRouter, UploadFile, File
from pathlib import Path
import uuid
import zipfile
import shutil

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

    # --------------------------------------------------
    # FIND ALL NIFTI FILES (including nested folders)
    # --------------------------------------------------
    all_nifti = list(case_folder.rglob("*.nii")) + \
                list(case_folder.rglob("*.nii.gz"))

    # --------------------------------------------------
    # BRA TS MODALITY MAPPING
    # --------------------------------------------------
    for f in all_nifti:

        name = f.name.lower()

        # ⭐ ORDER MATTERS (specific first)

        if "t2f" in name or "flair" in name:
            shutil.copy(f, case_folder / "flair.nii.gz")

        elif "t1c" in name or "t1ce" in name:
            shutil.copy(f, case_folder / "t1ce.nii.gz")

        elif "t1n" in name or "-t1" in name:
            shutil.copy(f, case_folder / "t1.nii.gz")

        elif "t2w" in name:
            shutil.copy(f, case_folder / "t2.nii.gz")

    return {
        "case_id": case_id,
        "message": "Upload and modality mapping successful"
    }