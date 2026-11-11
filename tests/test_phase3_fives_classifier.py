"""
Unit Tests for Phase 3: FIVES Multi-Disease Pathology Classifier (AMD, DR, Glaucoma, Normal).
"""

import os
from pathlib import Path

# Ensure KERAS_HOME points to workspace cache
workspace_keras = Path(__file__).resolve().parents[1] / ".keras"
if workspace_keras.exists():
    os.environ["KERAS_HOME"] = str(workspace_keras)

import unittest
import numpy as np
import tensorflow as tf

from src.models.fives_classifier import (
    build_fives_classifier,
    FIVESClassifierWrapper,
    DISEASE_LABELS,
    DISEASE_SHORT_CODES
)


class TestPhase3FIVESClassifier(unittest.TestCase):
    """Test suite for FIVES secondary pathology classifier."""

    @classmethod
    def setUpClass(cls):
        cls.model = build_fives_classifier(
            input_shape=(224, 224, 3),
            num_classes=4,
            dropout_rate=0.2,
            trainable_backbone=False
        )

    def test_model_topology_and_softmax(self):
        self.assertEqual(self.model.input_shape, (None, 224, 224, 3))
        self.assertEqual(self.model.output_shape, (None, 4))

        # Test forward pass with dummy batch
        dummy_batch = np.random.uniform(0.0, 1.0, (3, 224, 224, 3)).astype(np.float32)
        preds = self.model(dummy_batch, training=False).numpy()

        self.assertEqual(preds.shape, (3, 4))
        # Softmax probabilities must sum to 1.0 per row
        for row in preds:
            self.assertAlmostEqual(float(np.sum(row)), 1.0, places=5)
            self.assertTrue(np.all(row >= 0.0))

    def test_categorical_cross_entropy_loss(self):
        loss_fn = tf.keras.losses.CategoricalCrossentropy()

        # True class is Glaucoma (index 2) -> [0, 0, 1, 0]
        y_true = tf.constant([[0.0, 0.0, 1.0, 0.0]])

        # Confident correct prediction
        pred_correct = tf.constant([[0.02, 0.03, 0.90, 0.05]])
        loss_correct = loss_fn(y_true, pred_correct).numpy()

        # Confident wrong prediction (predicted AMD index 0)
        pred_wrong = tf.constant([[0.90, 0.05, 0.02, 0.03]])
        loss_wrong = loss_fn(y_true, pred_wrong).numpy()

        self.assertLess(loss_correct, loss_wrong)
        self.assertGreater(loss_correct, 0.0)

    def test_inference_wrapper_and_safety_gate(self):
        wrapper = FIVESClassifierWrapper(self.model, confidence_threshold=0.70)
        dummy_img = np.random.uniform(0.0, 1.0, (224, 224, 3)).astype(np.float32)

        res = wrapper.predict_single(dummy_img)

        # Verify all clinical report fields are present
        required_fields = [
            "predicted_class",
            "disease_name",
            "disease_code",
            "probabilities",
            "confidence",
            "is_high_confidence",
            "triage_status",
            "is_pathological"
        ]
        for f in required_fields:
            self.assertIn(f, res)

        self.assertIn(res["predicted_class"], [0, 1, 2, 3])
        self.assertIn(res["disease_code"], ["AMD", "DR", "Glaucoma", "Normal"])
        self.assertIsInstance(res["probabilities"], dict)
        self.assertEqual(len(res["probabilities"]), 4)
        self.assertIsInstance(res["confidence"], float)
        self.assertIsInstance(res["is_high_confidence"], bool)

        # Check safety status logic
        if res["confidence"] >= 0.70:
            self.assertEqual(res["triage_status"], "High Confidence")
        else:
            self.assertEqual(res["triage_status"], "Uncertain / Manual Review Recommended")

    def test_gradient_backpropagation(self):
        loss_fn = tf.keras.losses.CategoricalCrossentropy()
        dummy_x = tf.random.uniform((2, 224, 224, 3))
        dummy_y = tf.constant([[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]])

        with tf.GradientTape() as tape:
            preds = self.model(dummy_x, training=True)
            loss = loss_fn(dummy_y, preds)

        grads = tape.gradient(loss, self.model.trainable_variables)
        self.assertGreater(len(grads), 0)
        has_non_zero = any(np.any(g.numpy() != 0) for g in grads if g is not None)
        self.assertTrue(has_non_zero)


if __name__ == "__main__":
    unittest.main()

