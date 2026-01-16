import tensorflow as tf
import numpy as np
from pathlib import Path
from src.config import PROCESSED_DATA_DIR, NUM_CLASSES, IMG_SIZE, INPUT_CHANNELS


def load_npy(img_path, mask_path):
    img = np.load(img_path)
    mask = np.load(mask_path)

    # Resize
    img = tf.image.resize(img, IMG_SIZE)
    mask = tf.image.resize(tf.expand_dims(mask, -1), IMG_SIZE, method="nearest")

    # One-hot encode mask
    mask = tf.squeeze(mask)
    mask = tf.one_hot(tf.cast(mask, tf.int32), NUM_CLASSES)

    return img, mask


def get_dataset(split="Train"):
    split_map = {
        "Train": "Train_data",
        "Val": "Validation_data",
        "Test": "Test_data"
    }
    folder = split_map[split]
    img_dir = PROCESSED_DATA_DIR / folder / "Images"
    mask_dir = PROCESSED_DATA_DIR / folder / "Masks"

    img_files = sorted([str(p) for p in img_dir.glob("*.npy")])
    mask_files = sorted([str(p) for p in mask_dir.glob("*.npy")])

    ds = tf.data.Dataset.from_tensor_slices((img_files, mask_files))

    def _load(img_path, mask_path):
        img, mask = tf.numpy_function(
            func=load_npy,
            inp=[img_path, mask_path],
            Tout=[tf.float32, tf.float32]
        )
        img.set_shape((*IMG_SIZE, INPUT_CHANNELS))
        mask.set_shape((*IMG_SIZE, NUM_CLASSES))
        return img, mask

    ds = ds.map(_load, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(4).prefetch(tf.data.AUTOTUNE)
    return ds
