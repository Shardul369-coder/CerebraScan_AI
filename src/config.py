from pathlib import Path

RAW_DATA_DIR = Path("data/raw")
PROCESSED_DATA_DIR = Path("data/processed")

NUM_CLASSES = 4   # 0..3 BRATS labels
INPUT_CHANNELS = 4  # 2.5D slices
IMG_SIZE = (256, 256)
BATCH_SIZE = 4
EPOCHS = 50
LEARNING_RATE = 1e-4
