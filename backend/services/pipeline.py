from pathlib import Path
import json
import threading
import sys

# services
from backend.services.preprocessing import prepare_for_inference
from backend.services.segmentation import run_segmentation
from src.export_nifti import export_nifti
from backend.services.volumetric import run_volumetric
from backend.services.visualize_service import run_visualization
from backend.services.mesh_generator import generate_meshes

# ====================================================
# STORAGE
# ====================================================
RESULT_DIR = Path("backend/storage/results")
RESULT_DIR.mkdir(parents=True, exist_ok=True)

JOBS = {}
LOG_STREAMS = {}
LOG_LOCK = threading.Lock()


# ====================================================
# LOGGING SYSTEM
# ====================================================
def append_log(upload_id: str, message: str):
    with LOG_LOCK:
        if upload_id not in LOG_STREAMS:
            LOG_STREAMS[upload_id] = []
        LOG_STREAMS[upload_id].append(message)


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

    class StreamToBuffer:
        def __init__(self, upload_id):
            self.upload_id = upload_id
            self._buffer = ""

        def write(self, message):
            if not message:
                return

            # Preserve carriage returns and split properly
            self._buffer += message

            while "\n" in self._buffer:
                line, self._buffer = self._buffer.split("\n", 1)

                # Remove carriage return formatting from tqdm
                line = line.replace("\r", "").strip()

                if line:
                    append_log(self.upload_id, line)

        def flush(self):
            if self._buffer.strip():
                append_log(self.upload_id, self._buffer.strip())
            self._buffer = ""
            
    original_stdout = sys.stdout
    sys.stdout = StreamToBuffer(upload_id)

    try:
        print(f"[PIPELINE] Starting pipeline for {upload_id}")

        update_status(upload_id, "processing", 5, "preprocessing")

        upload_dir = f"backend/storage/uploads/{upload_id}"

        prep_result = prepare_for_inference(upload_id, upload_dir)
        test_img_dir = prep_result["destination"]

        update_status(upload_id, "processing", 30, "segmentation")

        run_segmentation(test_img_dir, patient_id)

        update_status(upload_id, "processing", 60, "export_nifti")

        mask_files = list(
            Path(f"backend/storage/masks/{patient_id}").glob("*_3d.npy")
        )

        if not mask_files:
            raise RuntimeError("Segmentation failed — no masks generated")

        mask_path = mask_files[0]

        nifti_path = export_nifti(str(mask_path))

        update_status(upload_id, "processing", 80, "volumetrics")

        volumes = run_volumetric(str(nifti_path))

        update_status(upload_id, "processing", 90, "visualization")

        run_visualization(patient_id)

        # Generates 3D meshes
        generate_meshes(str(nifti_path), upload_id)

        result = {
            "upload_id": upload_id,
            "patient_id": patient_id,
            "status": "completed",
            "volumetrics": volumes,
            "progress": 100,
            "current_step": "done"
        }

        with open(RESULT_DIR / f"{upload_id}.json", "w") as f:
            json.dump(result, f, indent=4)

        JOBS[upload_id] = result

        print("[PIPELINE] Completed successfully.")

    except Exception as e:
        print(f"[ERROR] {str(e)}")
        update_status(upload_id, "failed", 100, error=str(e))

    finally:
        sys.stdout = original_stdout


# ====================================================
# BACKGROUND LAUNCHER
# ====================================================
def launch_pipeline(upload_id: str, patient_id: str):
    thread = threading.Thread(
        target=run_full_pipeline,
        args=(upload_id, patient_id),
        daemon=True
    )
    thread.start()