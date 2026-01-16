import numpy as np
from pathlib import Path
from tqdm import tqdm
from keras.models import load_model
from src.data import load_npy
from src.losses import dice_coef_multiclass

def predict_patient(model, slice_paths):
    volume = []

    for p in tqdm(sorted(slice_paths), desc="Predicting slices"):
        img, _ = load_npy(str(p), str(p))
        pred = model.predict(np.expand_dims(img, 0))[0]
        mask = np.argmax(pred, axis=-1)
        volume.append(mask)

    return np.stack(volume, axis=-1)  # (H, W, D)

def run_inference(model_path, test_dir, output_dir="predictions_3d"):
    model = load_model(model_path, compile=False,
                       custom_objects={"dice_coef_multiclass": dice_coef_multiclass})

    test_dir = Path(test_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    patients = {}
    for p in test_dir.glob("*.npy"):
        pid = "_".join(p.stem.split("_")[:-2])
        patients.setdefault(pid, []).append(p)

    results = {}
    for pid, slices in patients.items():
        print(f"[INFO] Processing {pid} ({len(slices)} slices)")
        volume = predict_patient(model, slices)
        np.save(output_dir / f"{pid}_3d.npy", volume)
        results[pid] = str(output_dir / f"{pid}_3d.npy")

    return results

if __name__ == "__main__":
    run_inference("checkpoints/seg_best.h5", "processed_data/Test_data/Images")
