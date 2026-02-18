import os
import glob
import nibabel as nib
import numpy as np
import pandas as pd


# ==============================
# CONFIG
# ==============================
SEGMENTATION_FOLDER = "nifti_outputs"
CSV_OUTPUT = "volumetric_results.csv"
XLS_OUTPUT = "volumetric_results.xlsx"

# Confirmed label mapping
LABELS = {
    1: "NET",      # Blue
    2: "Edema",    # Green
    3: "ET"        # Yellow
}


# ==============================
# Compute Volume for One Patient
# ==============================
def compute_volumes(seg_path):

    nii = nib.load(seg_path)
    data = nii.get_fdata()

    # voxel spacing (mm)
    dx, dy, dz = nii.header.get_zooms()[:3]
    voxel_volume = dx * dy * dz  # mm³ per voxel

    results = {}
    total_voxels = 0

    for label_value, label_name in LABELS.items():

        voxel_count = np.sum(data == label_value)

        volume_mm3 = voxel_count * voxel_volume
        volume_cm3 = volume_mm3 / 1000.0

        results[f"{label_name}_voxels"] = int(voxel_count)
        results[f"{label_name}_mm3"] = round(volume_mm3, 2)
        results[f"{label_name}_cm3"] = round(volume_cm3, 2)

        total_voxels += voxel_count

    # Whole Tumor (WT) = NET + Edema + ET
    total_mm3 = total_voxels * voxel_volume
    total_cm3 = total_mm3 / 1000.0

    results["Whole_Tumor_mm3"] = round(total_mm3, 2)
    results["Whole_Tumor_cm3"] = round(total_cm3, 2)

    # Tumor Core (TC) = NET + ET
    tc_voxels = results["NET_voxels"] + results["ET_voxels"]
    tc_mm3 = tc_voxels * voxel_volume
    tc_cm3 = tc_mm3 / 1000.0

    results["Tumor_Core_mm3"] = round(tc_mm3, 2)
    results["Tumor_Core_cm3"] = round(tc_cm3, 2)

    # Percentage breakdown (relative to Whole Tumor)
    if total_voxels > 0:
        for label_name in ["NET", "Edema", "ET"]:
            percent = (results[f"{label_name}_voxels"] / total_voxels) * 100
            results[f"{label_name}_percent"] = round(percent, 2)
    else:
        for label_name in ["NET", "Edema", "ET"]:
            results[f"{label_name}_percent"] = 0

    return results


# ==============================
# Process All Patients
# ==============================
def process_all_patients():

    all_results = []

    seg_files = glob.glob(
        os.path.join(SEGMENTATION_FOLDER, "*_slice_3d.nii.gz")
    )

    if not seg_files:
        print("No segmentation files found!")
        return

    for seg_file in seg_files:

        patient_id = os.path.basename(seg_file).replace("_slice_3d.nii.gz", "")
        print(f"Processing {patient_id}...")

        volumes = compute_volumes(seg_file)
        volumes["Patient_ID"] = patient_id

        all_results.append(volumes)

    df = pd.DataFrame(all_results)

    # Put Patient_ID first column
    cols = ["Patient_ID"] + [col for col in df.columns if col != "Patient_ID"]
    df = df[cols]

    # Save
    df.to_csv(CSV_OUTPUT, index=False)
    df.to_excel(XLS_OUTPUT, index=False)

    print("\nVolumetric analysis completed successfully.")
    print("Saved:", CSV_OUTPUT)
    print("Saved:", XLS_OUTPUT)


# ==============================
# MAIN
# ==============================
if __name__ == "__main__":
    process_all_patients()
