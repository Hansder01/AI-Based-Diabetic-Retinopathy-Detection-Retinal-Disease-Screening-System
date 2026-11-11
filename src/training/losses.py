"""
Loss functions for Ordinal Regression, Multi-Class Pathology, and Vessel Segmentation.
"""

import tensorflow as tf
from tensorflow.keras import losses


@tf.keras.utils.register_keras_serializable(package="diabetic_retinopathy")
class OrdinalBinaryCrossEntropy(losses.Loss):
    """
    CORAL-style Binary Cross-Entropy across cumulative ordinal thresholds.
    Loss is averaged over the 4 binary ordinal indicators:
    L = - 1/4 * sum_{k=0}^3 [ y_k * log(p_k) + (1 - y_k) * log(1 - p_k) ]
    """

    def __init__(self, name: str = "ordinal_bce", **kwargs):
        super().__init__(name=name, **kwargs)
        self.bce = losses.BinaryCrossentropy(from_logits=False)

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        # y_true shape: (batch_size, 4), y_pred shape: (batch_size, 4)
        return self.bce(y_true, y_pred)

    def get_config(self):
        config = super().get_config()
        return config


@tf.keras.utils.register_keras_serializable(package="diabetic_retinopathy")
class DiceLoss(losses.Loss):
    """
    Soft-Dice Loss for microvascular semantic segmentation.
    Combats extreme foreground (vessel) vs. background imbalance.
    """

    def __init__(self, smooth: float = 1.0, name: str = "dice_loss", **kwargs):
        super().__init__(name=name, **kwargs)
        self.smooth = smooth

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        y_true_f = tf.cast(tf.reshape(y_true, [-1]), tf.float32)
        y_pred_f = tf.cast(tf.reshape(y_pred, [-1]), tf.float32)

        intersection = tf.reduce_sum(y_true_f * y_pred_f)
        dice = (2.0 * intersection + self.smooth) / (
            tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + self.smooth
        )
        return 1.0 - dice

    def get_config(self):
        config = super().get_config()
        config.update({"smooth": self.smooth})
        return config


@tf.keras.utils.register_keras_serializable(package="diabetic_retinopathy")
class BCEDiceLoss(losses.Loss):
    """
    Composite loss combining Binary Cross-Entropy (for pixel-level sharpness)
    and Dice Loss (for global vessel structure continuity).
    """

    def __init__(self, bce_weight: float = 0.5, smooth: float = 1.0, name: str = "bce_dice_loss", **kwargs):
        super().__init__(name=name, **kwargs)
        self.bce_weight = bce_weight
        self.bce = losses.BinaryCrossentropy(from_logits=False)
        self.dice = DiceLoss(smooth=smooth)

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        bce_loss = self.bce(y_true, y_pred)
        dice_loss = self.dice(y_true, y_pred)
        return (self.bce_weight * bce_loss) + ((1.0 - self.bce_weight) * dice_loss)

    def get_config(self):
        config = super().get_config()
        config.update({"bce_weight": self.bce_weight})
        return config

