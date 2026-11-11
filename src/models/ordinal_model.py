"""
EfficientNet-B0 Ordinal Regression Model for Diabetic Retinopathy Severity Grading.

Architecture:
- Backbone: Pretrained EfficientNet-B0 (ImageNet)
- Feature map: 7x7x1280 at 'top_activation'
- Global Average Pooling 2D -> 1280-dim feature vector
- Dropout (0.3)
- Ordinal Head: Dense(4, activation='sigmoid')
- Output: 4 cumulative probabilities [P(Y > 0), P(Y > 1), P(Y > 2), P(Y > 3)]
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

from src.data.dataset import ordinal_vector_to_grade, continuous_severity_score


SEVERITY_STAGES = {
    0: "No DR",
    1: "Mild",
    2: "Moderate",
    3: "Severe",
    4: "Proliferative"
}


def build_ordinal_model(
    input_shape: Tuple[int, int, int] = (224, 224, 3),
    num_classes: int = 5,
    dropout_rate: float = 0.3,
    trainable_backbone: bool = True
) -> Model:
    """
    Constructs an EfficientNet-B0 architecture with an Ordinal Regression output head.

    Args:
        input_shape: Input image dimensions (H, W, C).
        num_classes: Total number of severity grades (default 5: 0 to 4).
        dropout_rate: Dropout probability before the classification head.
        trainable_backbone: Whether backbone layers are trainable.

    Returns:
        Compiled or uncompiled tf.keras.Model.
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

    # 4 Sigmoids representing P(Y > 0), P(Y > 1), P(Y > 2), P(Y > 3)
    ordinal_output = layers.Dense(
        num_classes - 1,
        activation="sigmoid",
        name="ordinal_output"
    )(x)

    model = Model(inputs=inputs, outputs=ordinal_output, name="EfficientNetB0_OrdinalDR")
    return model


def ordinal_to_class_probabilities(probs: Union[List[float], np.ndarray]) -> np.ndarray:
    """
    Decomposes cumulative ordinal threshold probabilities P(Y > k)
    into a valid probability mass distribution across the 5 discrete grades [P(Y=0), ..., P(Y=4)].
    Enforces monotonic cumulative constraints: P(Y > 0) >= P(Y > 1) >= P(Y > 2) >= P(Y > 3).
    """
    p = np.clip(np.asarray(probs, dtype=np.float32), 0.0, 1.0)
    # Enforce non-increasing cumulative probabilities
    p_mono = np.maximum.accumulate(p[::-1])[::-1]

    p_classes = np.zeros(5, dtype=np.float32)
    p_classes[0] = 1.0 - p_mono[0]
    p_classes[1] = max(0.0, float(p_mono[0] - p_mono[1]))
    p_classes[2] = max(0.0, float(p_mono[1] - p_mono[2]))
    p_classes[3] = max(0.0, float(p_mono[2] - p_mono[3]))
    p_classes[4] = p_mono[3]

    total = float(np.sum(p_classes))
    if total > 0:
        p_classes /= total
    else:
        p_classes = np.ones(5, dtype=np.float32) / 5.0

    return p_classes


class OrdinalClassifierWrapper:
    """
    Inference wrapper for the Ordinal DR model. Provides decoding,
    confidence evaluation (Slide 10 safety threshold >= 70%),
    and continuous severity estimation.
    """

    def __init__(self, model: Model, confidence_threshold: float = 0.70):
        self.model = model
        self.confidence_threshold = confidence_threshold

    @classmethod
    def load_from_checkpoint(cls, checkpoint_path: str, confidence_threshold: float = 0.70):
        model = tf.keras.models.load_model(checkpoint_path, compile=False)
        return cls(model, confidence_threshold=confidence_threshold)

    def predict_probabilities(self, image_tensor: np.ndarray) -> np.ndarray:
        """
        Runs forward pass for batch or single image.
        """
        if len(image_tensor.shape) == 3:
            image_tensor = np.expand_dims(image_tensor, axis=0)
        return self.model.predict(image_tensor, verbose=0)

    def predict_single(self, preprocessed_image: np.ndarray) -> Dict[str, Any]:
        """
        Performs full ordinal inference and clinical triage assessment for a single image.

        Args:
            preprocessed_image: Float32 image array (224, 224, 3) normalized.

        Returns:
            Dictionary containing:
            - grade: Decoded integer grade (0-4)
            - stage_name: Clinical stage name (e.g. 'Moderate')
            - continuous_score: Continuous progression index [0.0 - 4.0]
            - ordinal_probs: [P(Y>0), P(Y>1), P(Y>2), P(Y>3)]
            - class_probabilities: [P(Y=0), P(Y=1), P(Y=2), P(Y=3), P(Y=4)]
            - confidence: Clinical confidence score in [0.0 - 1.0]
            - is_high_confidence: True if confidence >= confidence_threshold (default 0.70)
            - triage_status: 'High Confidence' or 'Uncertain / Manual Review Recommended'
        """
        probs = self.predict_probabilities(preprocessed_image)[0]
        grade = ordinal_vector_to_grade(probs, threshold=0.5)
        continuous_score = continuous_severity_score(probs)

        # Decompose cumulative probabilities into discrete class distribution
        class_probs = ordinal_to_class_probabilities(probs)
        # Clinical confidence is the probability mass of the selected grade
        confidence = float(class_probs[grade])
        # If the threshold-derived grade has lower mass than argmax, take the top mass
        max_prob = float(np.max(class_probs))
        if max_prob > confidence:
            confidence = max_prob

        is_high_conf = bool(confidence >= self.confidence_threshold)
        triage_status = "High Confidence" if is_high_conf else "Uncertain / Manual Review Recommended"

        return {
            "grade": grade,
            "stage_name": SEVERITY_STAGES.get(grade, f"Grade {grade}"),
            "continuous_score": round(continuous_score, 3),
            "ordinal_probs": [round(float(p), 4) for p in probs],
            "class_probabilities": [round(float(cp), 4) for cp in class_probs],
            "confidence": round(confidence, 3),
            "is_high_confidence": is_high_conf,
            "triage_status": triage_status,
            "is_dr_positive": bool(grade > 0)
        }
