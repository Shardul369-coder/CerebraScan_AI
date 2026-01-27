import tensorflow as tf
from tensorflow.keras import backend as K

# -------------------------------
# Multiclass Dice Coefficient (Vectorized)
# -------------------------------
def dice_coef_multiclass(y_true, y_pred, smooth=1e-6):
    """
    Calculate Dice coefficient for multi-class segmentation.
    Vectorized implementation - no loops needed.
    
    Args:
        y_true: Ground truth one-hot encoded tensor (batch, H, W, num_classes)
        y_pred: Predicted probabilities tensor (batch, H, W, num_classes)
        smooth: Smoothing factor to avoid division by zero
    
    Returns:
        Mean Dice coefficient across all classes
    """
    # Flatten spatial dimensions (H, W) for each class
    # Shape: (batch * H * W, num_classes)
    y_true_f = tf.reshape(y_true, [-1, tf.shape(y_true)[-1]])
    y_pred_f = tf.reshape(y_pred, [-1, tf.shape(y_pred)[-1]])
    
    # Calculate intersection and union for all classes at once
    # Shape: (num_classes,)
    intersection = tf.reduce_sum(y_true_f * y_pred_f, axis=0)
    union = tf.reduce_sum(y_true_f, axis=0) + tf.reduce_sum(y_pred_f, axis=0)
    
    # Dice coefficient for each class
    # Shape: (num_classes,)
    dice_per_class = (2.0 * intersection + smooth) / (union + smooth)
    
    # Return mean dice across all classes
    return tf.reduce_mean(dice_per_class)


# -------------------------------
# Multiclass Dice Loss
# -------------------------------
def dice_loss_multiclass(y_true, y_pred, smooth=1e-6):
    """Dice loss = 1 - Dice coefficient"""
    return 1.0 - dice_coef_multiclass(y_true, y_pred, smooth)


# -------------------------------
# Hybrid Loss = CCE + Dice Loss
# -------------------------------
def hybrid_loss(y_true, y_pred):
    """
    Combined Categorical Cross-Entropy and Dice Loss.
    Both y_true and y_pred should be one-hot encoded.
    """
    # Categorical cross-entropy
    cce = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
    cce = tf.reduce_mean(cce)  # Average across batch
    
    # Dice loss
    dice = dice_loss_multiclass(y_true, y_pred)
    
    return cce + dice


# -------------------------------
# Alternative: Weighted Dice Coefficient
# -------------------------------
def dice_coef_multiclass_weighted(y_true, y_pred, class_weights=None, smooth=1e-6):
    """
    Calculate weighted Dice coefficient for multi-class segmentation.
    Useful when classes are imbalanced.
    """
    # Flatten spatial dimensions
    y_true_f = tf.reshape(y_true, [-1, tf.shape(y_true)[-1]])
    y_pred_f = tf.reshape(y_pred, [-1, tf.shape(y_pred)[-1]])
    
    # Calculate per-class metrics
    intersection = tf.reduce_sum(y_true_f * y_pred_f, axis=0)
    union = tf.reduce_sum(y_true_f, axis=0) + tf.reduce_sum(y_pred_f, axis=0)
    dice_per_class = (2.0 * intersection + smooth) / (union + smooth)
    
    # Apply class weights if provided
    if class_weights is not None:
        class_weights = tf.constant(class_weights, dtype=tf.float32)
        dice_per_class = dice_per_class * class_weights
        return tf.reduce_sum(dice_per_class) / tf.reduce_sum(class_weights)
    
    return tf.reduce_mean(dice_per_class)


# -------------------------------
# Focal Loss (for imbalanced classes)
# -------------------------------
def focal_loss(y_true, y_pred, alpha=0.25, gamma=2.0):
    """
    Focal loss to handle class imbalance.
    """
    # Clip predictions to prevent log(0)
    y_pred = tf.clip_by_value(y_pred, K.epsilon(), 1 - K.epsilon())
    
    # Calculate focal loss
    cross_entropy = -y_true * tf.math.log(y_pred)
    weight = alpha * y_true * tf.pow((1 - y_pred), gamma)
    
    focal = weight * cross_entropy
    return tf.reduce_mean(tf.reduce_sum(focal, axis=-1))


# -------------------------------
# Focal + Dice Loss
# -------------------------------
def focal_dice_loss(y_true, y_pred):
    """
    Combined Focal Loss and Dice Loss.
    Better for highly imbalanced segmentation tasks.
    """
    focal = focal_loss(y_true, y_pred)
    dice = dice_loss_multiclass(y_true, y_pred)
    
    return focal + dice


# -------------------------------
# Weighted Hybrid Loss
# -------------------------------
def weighted_hybrid_loss(class_weights=None):
    """
    Factory function to create a weighted hybrid loss.
    
    Usage:
        loss_fn = weighted_hybrid_loss(class_weights=[0.3, 0.7])
        model.compile(loss=loss_fn, ...)
    """
    def loss(y_true, y_pred):
        cce = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
        cce = tf.reduce_mean(cce)
        dice = 1.0 - dice_coef_multiclass_weighted(y_true, y_pred, class_weights)
        return cce + dice
    
    return loss

# ========================================
# NEW FUNCTIONS FOR SCENARIO 2
# ========================================

# -------------------------------
# Dice Coefficient Excluding Background
# -------------------------------
def dice_coef_multiclass_no_bg(y_true, y_pred, smooth=1e-6):
    """
    Calculate Dice coefficient EXCLUDING background (class 0).
    This measures tumor segmentation quality only.
    
    Args:
        y_true: Ground truth one-hot encoded (batch, H, W, num_classes)
        y_pred: Predicted probabilities (batch, H, W, num_classes)
        smooth: Smoothing factor
    
    Returns:
        Mean Dice coefficient for tumor classes only (classes 1, 2, 3)
    """
    # Flatten spatial dimensions
    y_true_f = tf.reshape(y_true, [-1, tf.shape(y_true)[-1]])
    y_pred_f = tf.reshape(y_pred, [-1, tf.shape(y_pred)[-1]])
    
    # Calculate for classes 1, 2, 3 only (skip background at index 0)
    intersection = tf.reduce_sum(y_true_f[:, 1:] * y_pred_f[:, 1:], axis=0)
    union = tf.reduce_sum(y_true_f[:, 1:], axis=0) + tf.reduce_sum(y_pred_f[:, 1:], axis=0)
    
    # Dice per class
    dice_per_class = (2.0 * intersection + smooth) / (union + smooth)
    
    # Return mean across tumor classes
    return tf.reduce_mean(dice_per_class)


def dice_loss_no_bg(y_true, y_pred, smooth=1e-6):
    """Dice loss excluding background"""
    return 1.0 - dice_coef_multiclass_no_bg(y_true, y_pred, smooth)


# -------------------------------
# Weighted Loss with Class Weights
# -------------------------------
def weighted_focal_dice_loss(class_weights=None):
    """
    Factory function for weighted loss optimized for imbalanced tumor segmentation.
    
    Usage:
        loss_fn = weighted_focal_dice_loss([0.1, 1.5, 1.5, 2.0])
        model.compile(loss=loss_fn, ...)
    
    Args:
        class_weights: List of weights for [Background, NET, Edema, ET]
                      Default: [0.1, 1.5, 1.5, 2.0]
                      - Background (0.1): Low weight, mistakes don't matter much
                      - NET/Edema (1.5): High weight, mistakes matter
                      - ET (2.0): Highest weight, most important class
    
    Returns:
        Loss function for model compilation
    """
    if class_weights is None:
        class_weights = [0.1, 1.5, 1.5, 2.0]
    
    def loss(y_true, y_pred):
        # Weighted Dice Loss
        dice = 1.0 - dice_coef_multiclass_weighted(y_true, y_pred, class_weights)
        
        # Focal Loss with high gamma (focuses on hard examples)
        y_pred_clipped = tf.clip_by_value(y_pred, K.epsilon(), 1 - K.epsilon())
        cross_entropy = -y_true * tf.math.log(y_pred_clipped)
        
        # Gamma = 4.0 makes model focus heavily on misclassified pixels
        weight = 0.25 * y_true * tf.pow((1 - y_pred_clipped), 4.0)
        focal = tf.reduce_mean(tf.reduce_sum(weight * cross_entropy, axis=-1))
        
        # Combine: prioritize dice for segmentation quality
        return focal + 3.0 * dice
    
    return loss