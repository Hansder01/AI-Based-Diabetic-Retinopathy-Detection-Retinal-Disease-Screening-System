"""
Unit Tests for Phase 2: EfficientNet-B0 Ordinal Regression Model & Metrics.
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

from src.models.ordinal_model import build_ordinal_model, OrdinalClassifierWrapper
from src.training.losses import OrdinalBinaryCrossEntropy
from src.training.metrics import compute_ordinal_metrics
from src.data.dataset import grade_to_ordinal_vector


class TestPhase2Model(unittest.TestCase):
    """Test suite for Phase 2 architecture, ordinal loss, and inference wrapper."""

    @classmethod
    def setUpClass(cls):
        # Build small model for fast testing
        cls.model = build_ordinal_model(
            input_shape=(224, 224, 3),
            num_classes=5,
            dropout_rate=0.2,
            trainable_backbone=False
        )

    def test_model_topology(self):
        self.assertEqual(self.model.input_shape, (None, 224, 224, 3))
        self.assertEqual(self.model.output_shape, (None, 4))

        # Check forward pass with dummy batch
        dummy_batch = np.random.uniform(0.0, 1.0, (2, 224, 224, 3)).astype(np.float32)
        preds = self.model(dummy_batch, training=False)
        self.assertEqual(preds.shape, (2, 4))
        # Sigmoid must be strictly bounded in (0, 1)
        self.assertTrue(np.all(preds.numpy() >= 0.0))
        self.assertTrue(np.all(preds.numpy() <= 1.0))

    def test_ordinal_loss_mathematics(self):
        """
        Verify Slide 8 ordinal property:
        A 1-grade error MUST yield significantly lower loss than a 3-grade error!
        """
        loss_fn = OrdinalBinaryCrossEntropy()

        # True grade = 1: target vector = [1, 0, 0, 0]
        y_true = tf.constant([[1.0, 0.0, 0.0, 0.0]])

        # Prediction A: 1-grade miss (predicted Grade 2: [0.95, 0.95, 0.05, 0.05])
        pred_grade_2 = tf.constant([[0.95, 0.95, 0.05, 0.05]])
        loss_1_grade_miss = loss_fn(y_true, pred_grade_2).numpy()

        # Prediction B: 3-grade miss (predicted Grade 4: [0.95, 0.95, 0.95, 0.95])
        pred_grade_4 = tf.constant([[0.95, 0.95, 0.95, 0.95]])
        loss_3_grade_miss = loss_fn(y_true, pred_grade_4).numpy()

        self.assertLess(
            loss_1_grade_miss,
            loss_3_grade_miss,
            f"Ordinal loss failed: 1-grade miss ({loss_1_grade_miss}) should be < 3-grade miss ({loss_3_grade_miss})"
        )

    def test_metrics_evaluation(self):
        # 5 samples with known grades
        y_true = [0, 1, 2, 3, 4]
        # Perfect predictions
        y_pred_probs = [
            [0.1, 0.05, 0.01, 0.0],
            [0.9, 0.1, 0.05, 0.0],
            [0.9, 0.9, 0.1, 0.05],
            [0.95, 0.9, 0.85, 0.1],
            [0.99, 0.95, 0.92, 0.88]
        ]
        metrics = compute_ordinal_metrics(y_true, y_pred_probs)
        self.assertEqual(metrics["accuracy"], 1.0)
        self.assertEqual(metrics["mean_absolute_error"], 0.0)
        self.assertEqual(metrics["quadratic_weighted_kappa"], 1.0)
        self.assertEqual(metrics["macro_f1"], 1.0)

    def test_inference_wrapper_and_safety_gate(self):
        wrapper = OrdinalClassifierWrapper(self.model, confidence_threshold=0.70)
        dummy_img = np.random.uniform(0.0, 1.0, (224, 224, 3)).astype(np.float32)

        res = wrapper.predict_single(dummy_img)
        self.assertIn("grade", res)
        self.assertIn("stage_name", res)
        self.assertIn("continuous_score", res)
        self.assertIn("ordinal_probs", res)
        self.assertIn("confidence", res)
        self.assertIn("is_high_confidence", res)
        self.assertIn("triage_status", res)
        self.assertIn("is_dr_positive", res)

        # Check types
        self.assertIsInstance(res["grade"], int)
        self.assertIn(res["grade"], [0, 1, 2, 3, 4])
        self.assertIsInstance(res["continuous_score"], float)
        self.assertIsInstance(res["confidence"], float)

    def test_gradient_backward_pass(self):
        """Verify model trains and receives non-zero gradients."""
        loss_fn = OrdinalBinaryCrossEntropy()
        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)

        dummy_x = tf.random.uniform((2, 224, 224, 3))
        dummy_y = tf.constant([[1.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 0.0]])

        with tf.GradientTape() as tape:
            preds = self.model(dummy_x, training=True)
            loss = loss_fn(dummy_y, preds)

        grads = tape.gradient(loss, self.model.trainable_variables)
        self.assertGreater(len(grads), 0)
        # Verify gradients are not None and not all zero
        has_non_zero = any(np.any(g.numpy() != 0) for g in grads if g is not None)
        self.assertTrue(has_non_zero)


if __name__ == "__main__":
    unittest.main()
