import json
import numpy as np
from keras.models import load_model
from src.data import get_dataset
from src.losses import dice_coef_multiclass, dice_coef_multiclass_no_bg

def compute_iou(y_true, y_pred, num_classes):
    """
    Computes mean IoU across classes.
    y_true, y_pred are numpy arrays with shape (B,H,W,C).
    """
    y_true = np.argmax(y_true, axis=-1)
    y_pred = np.argmax(y_pred, axis=-1)

    ious = []
    for cls in range(num_classes):
        intersection = np.logical_and(y_true == cls, y_pred == cls).sum()
        union = np.logical_or(y_true == cls, y_pred == cls).sum()
        if union == 0:
            ious.append(np.nan)
        else:
            ious.append(intersection / union)

    return float(np.nanmean(ious))

# Around line 25, update function:
def evaluate_model(model_path="checkpoints/seg_best.h5", output_path="test_metrics.json"):
    # Load model
    model = load_model(
        model_path, 
        compile=False, 
        custom_objects={
            "dice_coef_multiclass": dice_coef_multiclass,
            "dice_coef_multiclass_no_bg": dice_coef_multiclass_no_bg
        }
    )

    test_ds, test_steps = get_dataset("Test")

    dice_scores = []
    dice_tumor_scores = []  # NEW: Track tumor-only dice
    iou_scores = []

    print("[INFO] Running evaluation on Test set...")

    for images, masks in test_ds:
        preds = model.predict(images, verbose=0)
        
        # Convert masks to numpy if needed
        if hasattr(masks, 'numpy'):
            masks_np = masks.numpy()
        else:
            masks_np = masks
        
        # Overall Dice (includes background)
        dice = dice_coef_multiclass(masks_np, preds).numpy()
        dice_scores.append(float(dice))
        
        # Tumor-only Dice (KEY METRIC!)
        dice_tumor = dice_coef_multiclass_no_bg(masks_np, preds).numpy()
        dice_tumor_scores.append(float(dice_tumor))

        # IoU
        iou = compute_iou(masks_np, preds, preds.shape[-1])
        iou_scores.append(iou)

    # Aggregate final metrics
    results = {
        "test_mean_dice": float(np.mean(dice_scores)),
        "test_mean_dice_tumor_only": float(np.mean(dice_tumor_scores)),  # NEW!
        "test_mean_iou": float(np.mean(iou_scores))
    }

    with open(output_path, "w") as f:
        json.dump(results, f, indent=4)

    print("===== TEST METRICS =====")
    print(f"Mean Dice (all):   {results['test_mean_dice']:.4f}")
    print(f"Mean Dice (tumor): {results['test_mean_dice_tumor_only']:.4f}")  # ADD THIS
    print(f"Mean IoU:          {results['test_mean_iou']:.4f}")
    print("========================")

    return results

if __name__ == "__main__":
    evaluate_model()