import os
import logging
import yaml
from src.data import get_dataset
import tensorflow 
from tensorflow import keras
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Concatenate, Input
from keras.models import Model
from keras.optimizers import Adam
from src.losses import hybrid_loss, dice_coef_multiclass

log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

logger = logging.getLogger('Model_seg')
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

log_file = os.path.join(log_dir, 'Model_seg.log')
file_handler = logging.FileHandler(log_file)
file_handler.setLevel(logging.DEBUG)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def load_params(param_path: str) -> dict:
    """Load parameters from a YAML file."""
    try:
        with open(param_path, 'r') as file:
            params = yaml.safe_load(file)
        logger.debug(f"Parameters loaded from {param_path}")
        return params
    except FileNotFoundError:
        logger.error(f"File not found: {param_path}")
        raise
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML file: {e}")
        raise

def conv_block(x, filters):
    x = Conv2D(filters, 3, padding="same", activation="relu")(x)
    x = Conv2D(filters, 3, padding="same", activation="relu")(x)
    return x

def unet2d(input_shape, num_classes):
    inputs = Input(shape=input_shape)

    c1 = conv_block(inputs, 64)
    p1 = MaxPooling2D()(c1)

    c2 = conv_block(p1, 128)
    p2 = MaxPooling2D()(c2)

    c3 = conv_block(p2, 256)
    p3 = MaxPooling2D()(c3)

    c4 = conv_block(p3, 512)
    p4 = MaxPooling2D()(c4)

    c5 = conv_block(p4, 1024)

    u6 = UpSampling2D()(c5)
    u6 = Concatenate()([u6, c4])
    c6 = conv_block(u6, 512)

    u7 = UpSampling2D()(c6)
    u7 = Concatenate()([u7, c3])
    c7 = conv_block(u7, 256)

    u8 = UpSampling2D()(c7)
    u8 = Concatenate()([u8, c2])
    c8 = conv_block(u8, 128)

    u9 = UpSampling2D()(c8)
    u9 = Concatenate()([u9, c1])
    c9 = conv_block(u9, 64)

    outputs = Conv2D(num_classes, 1, activation="softmax")(c9)

    return Model(inputs, outputs)

def train_segmentation(train_ds, val_ds, params):

    try:
        input_shape = (params['IMG_SIZE'][0],params['IMG_SIZE'][1],params['INPUT_CHANNELS'])
        model = unet2d(input_shape=input_shape, num_classes=params['NUM_CLASSES'])
        model.summary(print_fn=lambda x: logger.debug(x))
        model.compile(optimizer = Adam(learning_rate=params['LEARNING_RATE']),loss=hybrid_loss,metrics=[dice_coef_multiclass])

        train_ds = train_ds.prefetch(tensorflow.data.AUTOTUNE)
        val_ds = val_ds.prefetch(tensorflow.data.AUTOTUNE)

        os.makedirs("checkpoints", exist_ok=True)

        cb = [
            keras.callbacks.ModelCheckpoint(filepath="checkpoints/seg_best.h5",save_best_only=True,monitor="val_dice_coef_multiclass",mode="max"),
            keras.callbacks.ReduceLROnPlateau(monitor="val_loss",patience=3),
            keras.callbacks.EarlyStopping(monitor="val_dice_coef_multiclass",mode="max",patience=7,restore_best_weights=True)
             ]

        logger.debug('Model training started')
        logger.debug(f"Training for {params['EPOCHS']} epochs...")
        model.fit(train_ds,validation_data=val_ds,epochs=params['EPOCHS'],callbacks=cb)
        logger.debug('Model training completed')
    
        return model

    except ValueError as e:
        logger.error(f"ValueError during model training: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error during model training: {e}")
        raise

def save_model(model, model_path):
    try:
        os.makedirs("checkpoints", exist_ok=True)
        model.save(model_path)
        logger.debug(f"Model saved to {model_path}")
    except Exception as e:
        logger.error(f"Error saving model: {e}")
        raise

if __name__ == "__main__":
    train_segmentation(get_dataset("Train"), get_dataset("Val"), load_params("params.yaml")['model_seg'])
