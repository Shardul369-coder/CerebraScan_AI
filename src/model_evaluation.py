import json
import numpy as np
from keras.models import load_model
from src.data import get_dataset
from src.losses import dice_coef_multiclass

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


def evaluate_model(model_path="checkpoints/seg_best.h5", output_path="test_metrics.json"):
    # Load model (compile=False to avoid missing custom objects)
    model = load_model(model_path, compile=False, custom_objects={"dice_coef_multiclass": dice_coef_multiclass})

    # Load test dataset
    test_ds = get_dataset("Test")

    dice_scores = []
    iou_scores = []

    print("[INFO] Running evaluation on Test set...")

    for images, masks in test_ds:
        preds = model.predict(images)
        
        # Dice (uses your custom metric)
        dice = dice_coef_multiclass(masks, preds).numpy()
        dice_scores.append(float(dice))

        # IoU (uses numpy function above)
        iou = compute_iou(masks.numpy(), preds.numpy(), preds.shape[-1])
        iou_scores.append(iou)

    # Aggregate final metrics
    results = {
        "test_mean_dice": float(np.mean(dice_scores)),
        "test_mean_iou": float(np.mean(iou_scores))
    }

    # Save to json
    with open(output_path, "w") as f:
        json.dump(results, f, indent=4)

    print("===== TEST METRICS =====")
    print(f"Mean Dice: {results['test_mean_dice']:.4f}")
    print(f"Mean IoU:  {results['test_mean_iou']:.4f}")
    print("========================")

    return results


if __name__ == "__main__":
    evaluate_model()
