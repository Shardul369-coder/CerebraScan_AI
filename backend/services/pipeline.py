from pathlib import Path
import json

# services
from backend.services.preprocessing import prepare_for_inference
from backend.services.segmentation import run_segmentation
from src.export_nifti import export_nifti
from backend.services.volumetric import run_volumetric


# storage locations
RESULT_DIR = Path("backend/storage/results")
RESULT_DIR.mkdir(parents=True, exist_ok=True)


def run_full_pipeline(case_id: str, upload_dir: str):
    """
    MASTER PIPELINE
    Runs complete CerebraScan workflow
    """

    print(f"\n[PIPELINE] Starting analysis for {case_id}")

    # --------------------------------------------------
    # STEP 1 — PREPARE INPUT
    # --------------------------------------------------
    print("[STEP 1] Preparing data...")
    prep_result = prepare_for_inference(case_id, upload_dir)

    test_img_dir = prep_result["destination"]

    # --------------------------------------------------
    # STEP 2 — SEGMENTATION
    # --------------------------------------------------
    print("[STEP 2] Running segmentation...")
    seg_result = run_segmentation(test_img_dir, case_id)

    # seg_result = output folder from run_inference
    # get first predicted mask
    mask_files = list(Path(f"backend/storage/masks/{case_id}").glob("*_3d.npy"))

    if not mask_files:
        raise RuntimeError("Segmentation failed — no masks generated")

    mask_path = mask_files[0]

    # --------------------------------------------------
    # STEP 3 — EXPORT TO NIFTI
    # --------------------------------------------------
    print("[STEP 3] Exporting NIfTI...")
    nifti_path = export_nifti(str(mask_path))

    # --------------------------------------------------
    # STEP 4 — VOLUMETRIC ANALYSIS
    # --------------------------------------------------
    print("[STEP 4] Computing volumes...")
    volumes = run_volumetric(str(nifti_path))

    # --------------------------------------------------
    # STEP 5 — SAVE RESULT JSON
    # --------------------------------------------------
    result = {
        "case_id": case_id,
        "status": "completed",
        "mask_path": str(mask_path),
        "nifti_path": str(nifti_path),
        "volumetric_analysis": volumes
    }

    result_file = RESULT_DIR / f"{case_id}.json"

    with open(result_file, "w") as f:
        json.dump(result, f, indent=4)

    print("[PIPELINE] Finished successfully")

    return result