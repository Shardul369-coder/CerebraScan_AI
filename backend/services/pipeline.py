from pathlib import Path
import json
import threading

# services
from backend.services.preprocessing import prepare_for_inference
from backend.services.segmentation import run_segmentation
from src.export_nifti import export_nifti
from backend.services.volumetric import run_volumetric
from backend.services.visualize_service import run_visualization

# ====================================================
# STORAGE
# ====================================================
RESULT_DIR = Path("backend/storage/results")
RESULT_DIR.mkdir(parents=True, exist_ok=True)

# in-memory job tracker
JOBS = {}


# ====================================================
# STATUS HELPERS
# ====================================================
def update_status(upload_id, status, progress=0, step=None, error=None):
    if upload_id not in JOBS:
        JOBS[upload_id] = {}

    JOBS[upload_id].update({
        "status": status,
        "progress": progress,
        "current_step": step,
        "error": error,
    })


def get_job_status(upload_id):
    return JOBS.get(upload_id)


# ====================================================
# MAIN PIPELINE
# ====================================================
def run_full_pipeline(upload_id: str, patient_id: str):

    try:
        update_status(upload_id, "processing", 5, "preprocessing")

        upload_dir = f"backend/storage/uploads/{upload_id}"

        # STEP 1 — PREPROCESS
        prep_result = prepare_for_inference(upload_id, upload_dir)
        test_img_dir = prep_result["destination"]

        update_status(upload_id, "processing", 30, "segmentation")

        # STEP 2 — SEGMENTATION
        run_segmentation(test_img_dir, patient_id)
        update_status(upload_id, "processing", 90, "visualization")
        run_visualization(patient_id)

        mask_files = list(
            Path(f"backend/storage/masks/{patient_id}").glob("*_3d.npy")
        )

        if not mask_files:
            raise RuntimeError("Segmentation failed — no masks generated")

        mask_path = mask_files[0]

        update_status(upload_id, "processing", 60, "export_nifti")

        # STEP 3 — EXPORT NIFTI
        nifti_path = export_nifti(str(mask_path))

        update_status(upload_id, "processing", 80, "volumetrics")

        # STEP 4 — VOLUMETRIC
        volumes = run_volumetric(str(nifti_path))

        result = {
            "upload_id": upload_id,
            "patient_id": patient_id,
            "status": "completed",
            "segmentation": {
                "patient_id": patient_id,
                "volume_shape": [],
                "classes_detected": [],
                "distribution": [],
                "mask_path": str(mask_path),
                "prob_path": "",
            },
            "volumetrics": volumes,
        }

        with open(RESULT_DIR / f"{upload_id}.json", "w") as f:
            json.dump(result, f, indent=4)

        result.update({
            "progress": 100,
            "current_step": "done"
        })

        JOBS[upload_id] = result

    except Exception as e:
        update_status(upload_id, "failed", 100, error=str(e))
        raise


# ====================================================
# BACKGROUND LAUNCHER
# ====================================================
def launch_pipeline(upload_id: str, patient_id: str):
    """
    Starts pipeline in background thread
    """
    thread = threading.Thread(
        target=run_full_pipeline,
        args=(upload_id, patient_id),
        daemon=True
    )
    thread.start()