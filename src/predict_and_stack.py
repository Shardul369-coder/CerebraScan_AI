import os
import numpy as np
from pathlib import Path
from tqdm import tqdm
from keras.models import load_model
from src.losses import dice_coef_multiclass_no_bg,dice_coef_multiclass
import tensorflow as tf
from src.config import IMG_SIZE, INPUT_CHANNELS

def load_image_only(img_path):
    """Load and preprocess single image"""
    img = np.load(img_path)
    img = tf.convert_to_tensor(img, tf.float32)
    img = tf.image.resize(img, IMG_SIZE)
    return img.numpy()

def predict_patient_multiclass(model, slice_paths):
    """Predict 3D volume with multi-class output"""
    all_masks = []
    all_probs = []
    
    for p in tqdm(sorted(slice_paths), desc="Predicting slices"):
        # Load and preprocess
        img = load_image_only(str(p))
        img = np.expand_dims(img, axis=0)  # (1, H, W, C)
        
        # Get model prediction
        pred = model.predict(img, verbose=0)[0]  # (H, W, num_classes)
        
        # Store probabilities (optional, for analysis)
        all_probs.append(pred)
        
        # Get segmentation mask
        mask = np.argmax(pred, axis=-1)  # (H, W)
        all_masks.append(mask)
    
    # Stack into 3D volumes
    volume_mask = np.stack(all_masks, axis=-1)  # (H, W, D)
    volume_probs = np.stack(all_probs, axis=2)  # (H, W, D, num_classes)
    
    return volume_mask, volume_probs

def run_inference(model_path, test_img_dir, output_dir="predictions_3d"):
    """Main inference function with multi-class support"""
    
    # Load model
    print(f"[INFO] Loading model from {model_path}")
    
    model = load_model(
        model_path,
        compile=False,
        custom_objects={
            "dice_coef_multiclass": dice_coef_multiclass,
            "dice_coef_multiclass_no_bg": dice_coef_multiclass_no_bg
        }
    )
    
    # Verify model is multi-class
    print(f"[INFO] Model input shape: {model.input_shape}")
    print(f"[INFO] Model output shape: {model.output_shape}")
    
    num_classes = model.output_shape[-1]
    print(f"[INFO] Model has {num_classes} output channels")
    
    if num_classes <= 1:
        print("[WARNING] Model appears to be binary! Expecting at least 2 classes for multi-class.")
    
    test_img_dir = Path(test_img_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Group slices by patient
    patients = {}
    for p in test_img_dir.glob("*.npy"):
        # Extract patient ID from filename
        # Assuming format: BraTS-GLI-00035-000_001.npy
        stem = p.stem  # "BraTS-GLI-00035-000_001"
        
        # Remove slice number to get patient ID
        if "_" in stem:
            pid = stem.rsplit("_", 1)[0]  # "BraTS-GLI-00035-000"
        else:
            pid = stem
        
        patients.setdefault(pid, []).append(p)
    
    print(f"[INFO] Found {len(patients)} patients")
    
    results = {}
    for pid, slices in patients.items():
        print(f"\n[INFO] Processing {pid} ({len(slices)} slices)")
        
        # Predict
        volume_mask, volume_probs = predict_patient_multiclass(model, slices)
        
        # Save predictions
        mask_path = output_dir / f"{pid}_3d.npy"
        np.save(mask_path, volume_mask)
        
        # Also save probabilities for analysis (optional)
        prob_path = output_dir / f"{pid}_probs_3d.npy"
        np.save(prob_path, volume_probs)
        
        # Analyze results
        unique_vals = np.unique(volume_mask)
        print(f"  Prediction shape: {volume_mask.shape}")
        print(f"  Unique classes found: {sorted(unique_vals.tolist())}")
        
        if len(unique_vals) > 2:
            print(f"  🎉 MULTI-CLASS DETECTED! Classes: {list(unique_vals)}")
        
        # Class distribution
        print("  Class distribution:")
        for class_id in sorted(unique_vals):
            count = np.sum(volume_mask == class_id)
            percentage = (count / volume_mask.size) * 100
            print(f"    Class {class_id}: {count} voxels ({percentage:.2f}%)")
        
        results[pid] = {
            'mask_path': str(mask_path),
            'prob_path': str(prob_path),
            'shape': volume_mask.shape,
            'classes': unique_vals.tolist()
        }
    
    print(f"\n[INFO] All predictions saved to {output_dir}")
    return results

# Quick verification function
def verify_predictions(pred_dir="predictions_3d"):
    """Check if predictions contain multiple classes"""
    pred_dir = Path(pred_dir)
    
    for pred_file in pred_dir.glob("*_3d.npy"):
        if "_probs_" in str(pred_file):
            continue
            
        pred = np.load(pred_file)
        unique = np.unique(pred)
        
        print(f"\n{pred_file.name}:")
        print(f"  Shape: {pred.shape}")
        print(f"  Unique values: {sorted(unique.tolist())}")
        
        if len(unique) > 2:
            print(f"  ✅ MULTI-CLASS!")
        elif len(unique) == 2 and 0 in unique and 1 in unique:
            print(f"  ❌ Binary only (0 and 1)")
        else:
            print(f"  ⚠️ Unexpected classes: {list(unique)}")

if __name__ == "__main__":
    # Run inference
    print("=" * 60)
    print("RUNNING MULTI-CLASS INFERENCE")
    print("=" * 60)
    
    results = run_inference(
        model_path="checkpoints/seg_best.h5",
        test_img_dir="processed_data/Test_data/Images",
        output_dir="predictions_3d"
    )
    
    print("\n" + "=" * 60)
    print("VERIFICATION")
    print("=" * 60)
    verify_predictions()