from pathlib import Path
from src.render_3d import visualize_patient


def run_visualization(case_id: str):
    """
    Generate PNG visualizations from segmentation masks.

    INPUT:
        backend/storage/masks/<case_id>/*.npy

    OUTPUT:
        backend/storage/visualizations/<case_id>/
    """

    masks_dir = Path(f"backend/storage/masks/{case_id}")
    output_dir = Path(f"backend/storage/visualizations/{case_id}")
    raw_data_dir = Path(f"backend/storage/uploads/{case_id}")

    output_dir.mkdir(parents=True, exist_ok=True)

    # only segmentation masks (skip probs)
    pred_files = [
        f for f in masks_dir.glob("*_3d.npy")
        if "_probs_" not in f.name
    ]

    if not pred_files:
        print("[VIS] No mask files found")
        return

    print(f"[VIS] Found {len(pred_files)} masks")

    for pf in pred_files:

        patient_id = pf.stem.replace("_3d", "")

        visualize_patient(
            patient_id=patient_id,
            pred_file=pf,
            output_dir=output_dir,
            raw_data_dir=raw_data_dir,
            save_comparison=True,
            save_all_slices=False,
        )

    print(f"[VIS] Visualization completed → {output_dir}")