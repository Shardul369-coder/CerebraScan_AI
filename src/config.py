from pathlib import Path
from pathlib import Path

PROCESSED_DATA_DIR = Path("processed_data")
RAW_DATA_DIR = Path("data/raw")
NUM_CLASSES = 4   # 0..3 BRATS labels
INPUT_CHANNELS = 4  # 2.5D slices
IMG_SIZE = (240, 240)
BATCH_SIZE = 4
EPOCHS = 50
LEARNING_RATE = 1e-4
