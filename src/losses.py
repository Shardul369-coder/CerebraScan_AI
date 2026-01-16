import tensorflow as tf
from tensorflow.keras import backend as K

# -------------------------------
# Multiclass Dice Coefficient
# -------------------------------
def dice_coef_multiclass(y_true, y_pred, smooth=1e-6):
    num_classes = y_pred.shape[-1]
    dice = 0.0

    for i in range(num_classes):
        # Flatten each class map manually
        y_true_f = tf.reshape(y_true[..., i], [-1])
        y_pred_f = tf.reshape(y_pred[..., i], [-1])

        intersection = tf.reduce_sum(y_true_f * y_pred_f)
        denom = tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f)

        dice += (2.0 * intersection + smooth) / (denom + smooth)

    return dice / tf.cast(num_classes, tf.float32)


# -------------------------------
# Multiclass Dice Loss
# -------------------------------
def dice_loss_multiclass(y_true, y_pred, smooth=1e-6):
    return 1.0 - dice_coef_multiclass(y_true, y_pred, smooth)


# -------------------------------
# Hybrid Loss = CCE + Dice Loss
# -------------------------------
def hybrid_loss(y_true, y_pred):
    cce = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
    dice = dice_loss_multiclass(y_true, y_pred)
    return cce + dice
