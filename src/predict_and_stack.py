import os
import numpy as np
from pathlib import Path
from tqdm import tqdm
from keras.models import load_model
from src.losses import dice_coef_multiclass
import tensorflow as tf
from src.config import IMG_SIZE, INPUT_CHANNELS

# In predict_and_stack.py, update the load_image_only function:
def load_image_only(img_path):
    img = np.load(img_path)
    img = tf.convert_to_tensor(img, tf.float32)
    img = tf.image.resize(img, IMG_SIZE)
    return img.numpy()
    
def predict_patient(model, slice_paths):
    volume = []

    for p in tqdm(sorted(slice_paths), desc="Predicting slices"):
        img = load_image_only(str(p))
        img = np.expand_dims(img, axis=0)  # (1, H, W, C)
        pred = model.predict(img, verbose=0)[0]
        mask = np.argmax(pred, axis=-1)  # (H, W)
        volume.append(mask)

    return np.stack(volume, axis=-1)  # (H, W, D)

def run_inference(model_path, test_img_dir, output_dir="predictions_3d"):
    model = load_model(
        model_path,
        compile=False,
        custom_objects={"dice_coef_multiclass": dice_coef_multiclass}
    )

    test_img_dir = Path(test_img_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    patients = {}
    for p in test_img_dir.glob("*.npy"):
        parts = p.stem.split("_")
        pid = "_".join(parts[:-2])  # e.g. BraTS19_001 -> 001
        patients.setdefault(pid, []).append(p)

    print(f"[INFO] Found {len(patients)} patients")

    results = {}
    for pid, slices in patients.items():
        print(f"[INFO] Processing {pid} ({len(slices)} slices)")
        volume = predict_patient(model, slices)
        np.save(output_dir / f"{pid}_3d.npy", volume)
        results[pid] = str(output_dir / f"{pid}_3d.npy")

    return results

if __name__ == "__main__":
    run_inference("checkpoints/seg_best.h5", "processed_data/Test_data/Images")
