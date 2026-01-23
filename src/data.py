import tensorflow as tf
import numpy as np
from pathlib import Path
from src.config import PROCESSED_DATA_DIR, NUM_CLASSES, IMG_SIZE, INPUT_CHANNELS, BATCH_SIZE


def load_npy(img_path, mask_path):
    img_path = img_path.decode("utf-8")
    mask_path = mask_path.decode("utf-8")

    # Load data with numpy
    img = np.load(img_path).astype(np.float32)  # (H, W, 4)
    mask = np.load(mask_path)  # (H, W)
    
    # Handle mask - ensure it's 2D
    mask = np.squeeze(mask)
    if mask.ndim != 2:
        # Handle any dimension issues
        if mask.ndim == 3:
            mask = mask[:, :, 0]
        elif mask.ndim == 1:
            mask = mask.reshape(int(np.sqrt(len(mask))), -1)
    
    # Convert to binary (0 or 1)
    mask = (mask > 0).astype(np.int32)
    
    # Get target size
    H_target, W_target = IMG_SIZE
    H_orig, W_orig = mask.shape
    
    # Simple resize using numpy
    # Create coordinate grids for resizing
    y_coords = np.linspace(0, H_orig-1, H_target).astype(np.int32)
    x_coords = np.linspace(0, W_orig-1, W_target).astype(np.int32)
    
    # Resize mask (nearest neighbor)
    mask_resized = mask[y_coords[:, None], x_coords[None, :]]
    
    # Resize image (simple indexing - nearest neighbor)
    img_resized = img[y_coords[:, None], x_coords[None, :], :]
    
    # Normalize image
    img_normalized = (img_resized - np.mean(img_resized)) / (np.std(img_resized) + 1e-8)
    
    # One-hot encode mask
    mask_onehot = np.zeros((H_target, W_target, NUM_CLASSES), dtype=np.float32)
    for c in range(NUM_CLASSES):
        mask_onehot[:, :, c] = (mask_resized == c).astype(np.float32)
    
    return img_normalized.astype(np.float32), mask_onehot.astype(np.float32)


def get_dataset(split="Train", return_len=False):
    """
    Create TensorFlow dataset for training/validation/testing
    
    Args:
        split: One of "Train", "Val", or "Test"
        return_len: If True, return (total_samples, steps_per_epoch) instead of dataset
    
    Returns:
        If return_len=False: (dataset, steps_per_epoch)
        If return_len=True: (total_samples, steps_per_epoch)
    """
    split_map = {
        "Train": "Train_data",
        "Val": "Validation_data",
        "Test": "Test_data"
    }

    folder = split_map[split]
    img_dir = Path(PROCESSED_DATA_DIR) / folder / "Images"
    mask_dir = Path(PROCESSED_DATA_DIR) / folder / "Masks"

    img_files = sorted(img_dir.glob("*.npy"))
    mask_files = sorted(mask_dir.glob("*.npy"))

    assert len(img_files) == len(mask_files), f"Image/Mask count mismatch! Images: {len(img_files)}, Masks: {len(mask_files)}"

    total = len(img_files)
    steps = total // BATCH_SIZE

    if return_len:
        return total, steps

    # Create dataset from file paths
    ds = tf.data.Dataset.from_tensor_slices(
        (list(map(str, img_files)), list(map(str, mask_files)))
    )

    def _load(img_path, mask_path):
        """Wrapper function for tf.numpy_function"""
        img, mask = tf.numpy_function(
            load_npy,
            [img_path, mask_path],
            Tout=[tf.float32, tf.float32]
        )

        # Set shapes explicitly
        img.set_shape([IMG_SIZE[0], IMG_SIZE[1], INPUT_CHANNELS])
        mask.set_shape([IMG_SIZE[0], IMG_SIZE[1], NUM_CLASSES])

        return img, mask

    # Map loading function
    ds = ds.map(_load, num_parallel_calls=tf.data.AUTOTUNE)
    
    # Shuffle only for training
    if split == "Train":
        ds = ds.shuffle(buffer_size=32)
    
    # Batch and prefetch
    ds = ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    return ds, steps