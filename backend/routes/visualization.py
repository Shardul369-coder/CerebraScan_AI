from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pathlib import Path

router = APIRouter(prefix="/visualization", tags=["Visualization"])

VIS_DIR = Path("backend/storage/visualizations")


# --------------------------------------------------
# LIST ALL VISUALIZATION IMAGES
# --------------------------------------------------
@router.get("/{case_id}")
def list_visualizations(case_id: str):

    case_dir = VIS_DIR / case_id

    if not case_dir.exists():
        raise HTTPException(
            status_code=404,
            detail=f"No visualizations found for {case_id}"
        )

    images = sorted([f.name for f in case_dir.glob("*.png")])

    return {
        "case_id": case_id,
        "total_images": len(images),
        "images": images
    }


# --------------------------------------------------
# GET SINGLE IMAGE
# --------------------------------------------------
@router.get("/{case_id}/{image_name}")
def get_visualization_image(case_id: str, image_name: str):

    img_path = VIS_DIR / case_id / image_name

    if not img_path.exists():
        raise HTTPException(status_code=404, detail="Image not found")

    return FileResponse(img_path)