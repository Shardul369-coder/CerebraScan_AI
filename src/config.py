from ruamel.yaml import YAML
from pathlib import Path

yaml = YAML()
params = yaml.load(open("params.yaml"))["model_seg"]

PROCESSED_DATA_DIR = Path("processed_data")
RAW_DATA_DIR = Path("data/raw")

IMG_SIZE = tuple(params["IMG_SIZE"])        # <--- FIX HERE
INPUT_CHANNELS = params["INPUT_CHANNELS"]
NUM_CLASSES = params["NUM_CLASSES"]
BATCH_SIZE = params["BATCH_SIZE"]
EPOCHS = params["EPOCHS"]
LEARNING_RATE = params["LEARNING_RATE"]
