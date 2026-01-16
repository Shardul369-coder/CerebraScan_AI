import tensorflow as tf
from tensorflow import keras
from keras import backend as K

def dice_coef_multiclass(y_true, y_pred, smooth=1e-6):
    num_classes = y_pred.shape[-1]
    dice = 0.0
    for i in range(num_classes):
        y_true_f = K.flatten(y_true[..., i])
        y_pred_f = K.flatten(y_pred[..., i])
        intersection = K.sum(y_true_f * y_pred_f)
        denom = K.sum(y_true_f) + K.sum(y_pred_f)
        dice += (2.0 * intersection + smooth) / (denom + smooth)
    return dice / num_classes

def dice_loss_multiclass(y_true, y_pred, smooth=1e-6):
    return 1.0 - dice_coef_multiclass(y_true, y_pred, smooth)

def hybrid_loss(y_true, y_pred):
    ce = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
    dice = dice_loss_multiclass(y_true, y_pred)
    return ce + dice
