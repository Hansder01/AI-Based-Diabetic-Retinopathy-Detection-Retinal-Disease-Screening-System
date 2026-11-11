"""
EfficientNet-B0 4-Class Softmax Classifier for Secondary Retinal Pathologies.

Handles:
- AMD (Age-Related Macular Degeneration)
- Diabetic Retinopathy (DR fallback)
- Glaucoma
- Normal Retinal Eye
"""

from typing import Tuple, Dict, Any, Union, List, Optional
import os
from pathlib import Path
import numpy as np

# Ensure Keras points to the workspace .keras directory with pre-cached weights
workspace_keras = Path(__file__).resolve().parents[2] / ".keras"
if workspace_keras.exists():
    os.environ.setdefault("KERAS_HOME", str(workspace_keras))

import tensorflow as tf

try:
    from keras import layers, Model
    from keras.applications import EfficientNetB0
except ImportError:
    from tensorflow.keras import layers, Model  # type: ignore
    from tensorflow.keras.applications import EfficientNetB0  # type: ignore


DISEASE_LABELS = {
    0: "Age-Related Macular Degeneration (AMD)",
    1: "Diabetic Retinopathy",
    2: "Glaucoma",
    3: "Normal"
}

DISEASE_SHORT_CODES = {
    0: "AMD",
    1: "DR",
    2: "Glaucoma",
    3: "Normal"
}


def build_fives_classifier(
    input_shape: Tuple[int, int, int] = (224, 224, 3),
    num_classes: int = 4,
    dropout_rate: float = 0.3,
    trainable_backbone: bool = True
) -> Model:
    """
    Constructs an EfficientNet-B0 architecture with a 4-class Softmax classification head.

    Args:
        input_shape: Input image dimensions (H, W, C).
        num_classes: Total number of disease categories (AMD, DR, Glaucoma, Normal).
        dropout_rate: Dropout rate before classification head.
        trainable_backbone: Whether backbone layers are trainable.

    Returns:
        tf.keras.Model
    """
    base_model = EfficientNetB0(
        weights="imagenet",
        include_top=False,
        input_shape=input_shape
    )
    base_model.trainable = trainable_backbone

    inputs = layers.Input(shape=input_shape, name="fundus_input")
    # EfficientNetB0 has internal Rescaling(1/255.0). Scale [0, 1] inputs to [0, 255]
    # so EfficientNet receives standard pixel intensities as expected by ImageNet weights.
    x = base_model(inputs * 255.0, training=False)

    # 1x1x1280 Global Average Pooling
    x = layers.GlobalAveragePooling2D(name="gap_pooling")(x)

    if dropout_rate > 0.0:
        x = layers.Dropout(dropout_rate, name="head_dropout")(x)

    # 4-class Softmax Output
    outputs = layers.Dense(
        num_classes,
        activation="softmax",
        name="disease_softmax"
    )(x)

    model = Model(inputs=inputs, outputs=outputs, name="EfficientNetB0_FIVESClassifier")
    return model


class FIVESClassifierWrapper:
    """
    Inference wrapper for FIVES multi-pathology classification.
    Provides disease diagnosis, probability breakdown, and confidence evaluation
    (Slide 10 safety threshold >= 70%).
    """

    def __init__(self, model: Model, confidence_threshold: float = 0.70):
        self.model = model
        self.confidence_threshold = confidence_threshold

    @classmethod
    def load_from_checkpoint(cls, checkpoint_path: str, confidence_threshold: float = 0.70):
        model = tf.keras.models.load_model(checkpoint_path, compile=False)
        return cls(model, confidence_threshold=confidence_threshold)

    def predict_probabilities(self, image_tensor: np.ndarray) -> np.ndarray:
        if len(image_tensor.shape) == 3:
            image_tensor = np.expand_dims(image_tensor, axis=0)
        return self.model.predict(image_tensor, verbose=0)

    def predict_single(self, preprocessed_image: np.ndarray) -> Dict[str, Any]:
        """
        Runs multi-disease inference on a single 224x224x3 fundus image.

        Returns:
            Dictionary containing:
            - predicted_class: Integer index (0: AMD, 1: DR, 2: Glaucoma, 3: Normal)
            - disease_name: Full clinical disease name
            - disease_code: Short code ("AMD", "DR", "Glaucoma", "Normal")
            - probabilities: Dictionary of {disease_name: probability}
            - confidence: Top class probability [0.0 - 1.0]
            - is_high_confidence: True if top probability >= confidence_threshold
            - triage_status: 'High Confidence' or 'Uncertain / Manual Review Recommended'
            - is_pathological: True if not Normal
        """
        probs = self.predict_probabilities(preprocessed_image)[0]
        pred_idx = int(np.argmax(probs))
        confidence = float(probs[pred_idx])

        is_high_conf = bool(confidence >= self.confidence_threshold)
        triage_status = "High Confidence" if is_high_conf else "Uncertain / Manual Review Recommended"

        prob_breakdown = {
            DISEASE_SHORT_CODES[i]: round(float(probs[i]), 4)
            for i in range(len(probs))
        }

        return {
            "predicted_class": pred_idx,
            "disease_name": DISEASE_LABELS.get(pred_idx, f"Class {pred_idx}"),
            "disease_code": DISEASE_SHORT_CODES.get(pred_idx, "Unknown"),
            "probabilities": prob_breakdown,
            "confidence": round(confidence, 4),
            "is_high_confidence": is_high_conf,
            "triage_status": triage_status,
            "is_pathological": bool(pred_idx != 3)
        }
