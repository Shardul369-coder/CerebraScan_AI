from fastapi import APIRouter
from pathlib import Path
import json

# visualization service
from src.visualize_service import run_visualization_service

router = APIRouter()

# ============================================
# PATHS (match your backend structure)
# ============================================
PRED_DIR = Path("predictions_3d")
VIS_DIR = Path("visualizations")


# ============================================
# VISUALIZATION ENDPOINT
# ============================================
@router.post("/visualize/{case_id}")
def visualize_case(case_id: str):
    """
    Generate slice visualizations for a prediction case.

    Example:
        POST /visualize/case_001
    """

    # locate prediction file
    pred_file_nii = PRED_DIR / f"{case_id}.nii"
    pred_file_niigz = PRED_DIR / f"{case_id}.nii.gz"

    if pred_file_nii.exists():
        pred_path = pred_file_nii
    elif pred_file_niigz.exists():
        pred_path = pred_file_niigz
    else:
        return {"error": "Prediction file not found"}

    # run visualization service
    output_path = run_visualization_service(
        prediction_path=str(pred_path),
        output_dir=str(VIS_DIR)
    )

    # collect generated images
    images = sorted(Path(output_path).glob("*.png"))

    return {
        "case_id": case_id,
        "status": "success",
        "visualization_dir": str(output_path),
        "images": [str(img) for img in images]
    }


# ============================================
# GET VISUALIZATION RESULTS
# ============================================
@router.get("/visualize/{case_id}")
def get_visualization(case_id: str):

    case_dir = VIS_DIR / case_id

    if not case_dir.exists():
        return {"error": "Visualization not found"}

    images = sorted(case_dir.glob("*.png"))

    return {
        "case_id": case_id,
        "images": [str(img) for img in images]
    }