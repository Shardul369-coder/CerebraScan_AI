import tensorflow as tf
import numpy as np
from pathlib import Path
from src.config import PROCESSED_DATA_DIR, NUM_CLASSES, IMG_SIZE, INPUT_CHANNELS

def load_npy(img_path, mask_path):
    img = np.load(img_path)
    mask = np.load(mask_path)

    # resize image
    img = tf.image.resize(img, IMG_SIZE)

    # resize mask (nearest keeps labels intact)
    mask = tf.image.resize(tf.expand_dims(mask, -1), IMG_SIZE, method="nearest")

    # convert to int labels
    mask = tf.cast(mask, tf.int32)

    # remove that last 1-dim so shape becomes (H, W)
    mask = tf.squeeze(mask, axis=-1)

    # one-hot to (H, W, C)
    mask = tf.one_hot(mask, NUM_CLASSES)

    return img, mask


def get_dataset(split="Train", return_len=False):
    split_map = {
        "Train": "Train_data",
        "Val": "Validation_data",
        "Test": "Test_data"
    }
    folder = split_map[split]

    img_dir = Path(PROCESSED_DATA_DIR) / folder / "Images"
    mask_dir = Path(PROCESSED_DATA_DIR) / folder / "Masks"

    img_files = sorted(list(img_dir.glob("*.npy")))
    mask_files = sorted(list(mask_dir.glob("*.npy")))

    assert len(img_files) == len(mask_files), "Image/Mask count mismatch!"

    batch_size = 4

    if return_len:
        total = len(img_files)
        steps = total // batch_size
        return total, steps

    ds = tf.data.Dataset.from_tensor_slices(
        (list(map(str, img_files)), list(map(str, mask_files)))
    )

    def _load(img_path, mask_path):
        img, mask = tf.numpy_function(load_npy, [img_path, mask_path],
                                      Tout=[tf.float32, tf.float32])
        img.set_shape((*IMG_SIZE, INPUT_CHANNELS))
        mask.set_shape((*IMG_SIZE, NUM_CLASSES))
        return img, mask

    ds = ds.map(_load, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    steps = len(img_files) // batch_size

    return ds, steps
