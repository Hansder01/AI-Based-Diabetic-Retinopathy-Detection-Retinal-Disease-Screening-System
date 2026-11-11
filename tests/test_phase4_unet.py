"""
Unit Tests for Phase 4: U-Net Retinal Vessel Semantic Segmentation.
"""

import unittest
import numpy as np
import tensorflow as tf

from src.models.unet import build_unet, UNetVesselSegmentationWrapper
from src.training.losses import DiceLoss, BCEDiceLoss
from src.training.train_unet_vessels import dice_coefficient, iou_score


class TestPhase4UNet(unittest.TestCase):
    """Test suite for U-Net architecture, losses, metrics, and wrapper."""

    @classmethod
    def setUpClass(cls):
        # Use initial_filters=16 for fast unit testing
        cls.model = build_unet(input_shape=(256, 256, 3), initial_filters=16)

    def test_unet_topology(self):
        self.assertEqual(self.model.input_shape, (None, 256, 256, 3))
        self.assertEqual(self.model.output_shape, (None, 256, 256, 1))

        # Forward pass with dummy batch
        dummy_batch = np.random.uniform(0.0, 1.0, (2, 256, 256, 3)).astype(np.float32)
        preds = self.model(dummy_batch, training=False).numpy()

        self.assertEqual(preds.shape, (2, 256, 256, 1))
        # Sigmoid bounded
        self.assertTrue(np.all(preds >= 0.0))
        self.assertTrue(np.all(preds <= 1.0))

    def test_dice_loss_and_metrics(self):
        dice_loss = DiceLoss(smooth=1.0)
        bce_dice = BCEDiceLoss(bce_weight=0.5, smooth=1.0)

        # Perfect prediction
        y_true = tf.constant([[[[1.0], [0.0]], [[0.0], [1.0]]]])
        y_pred_perfect = tf.constant([[[[1.0], [0.0]], [[0.0], [1.0]]]])
        y_pred_wrong = tf.constant([[[[0.0], [1.0]], [[1.0], [0.0]]]])

        loss_perfect = dice_loss(y_true, y_pred_perfect).numpy()
        loss_wrong = dice_loss(y_true, y_pred_wrong).numpy()

        self.assertAlmostEqual(loss_perfect, 0.0, places=4)
        self.assertGreater(loss_wrong, 0.5)

        # Composite BCE + Dice
        composite_loss = bce_dice(y_true, y_pred_perfect).numpy()
        self.assertLess(composite_loss, 0.1)

    def test_segmentation_wrapper_and_vasculature_analysis(self):
        wrapper = UNetVesselSegmentationWrapper(self.model, threshold=0.5)

        # Create synthetic fundus image (circular disc with synthetic vessel tree)
        synthetic_img = np.zeros((300, 300, 3), dtype=np.uint8)
        # Retinal circle
        cv2_circle = np.zeros((300, 300), dtype=np.uint8)
        import cv2
        cv2.circle(synthetic_img, (150, 150), 120, (180, 80, 20), -1)
        # Synthetic vessels
        cv2.line(synthetic_img, (150, 150), (100, 50), (120, 20, 10), 3)
        cv2.line(synthetic_img, (150, 150), (200, 250), (120, 20, 10), 3)

        analysis = wrapper.analyze_vasculature(synthetic_img, target_size=(256, 256))

        expected_keys = [
            "probability_mask",
            "binary_mask",
            "vessel_density_percent",
            "vessel_pixel_count",
            "retina_area_pixels",
            "overlay_image",
            "preprocessed_image"
        ]
        for k in expected_keys:
            self.assertIn(k, analysis)

        self.assertEqual(analysis["probability_mask"].shape, (256, 256))
        self.assertEqual(analysis["binary_mask"].shape, (256, 256))
        self.assertEqual(analysis["overlay_image"].shape, (256, 256, 3))
        self.assertGreaterEqual(analysis["vessel_density_percent"], 0.0)
        self.assertLessEqual(analysis["vessel_density_percent"], 100.0)

    def test_skip_connection_gradient_flow(self):
        """Verify gradients flow through both encoder contracting and decoder skip paths."""
        loss_fn = BCEDiceLoss()
        dummy_x = tf.random.uniform((1, 256, 256, 3))
        dummy_y = tf.ones((1, 256, 256, 1))

        with tf.GradientTape() as tape:
            preds = self.model(dummy_x, training=True)
            loss = loss_fn(dummy_y, preds)

        grads = tape.gradient(loss, self.model.trainable_variables)
        self.assertGreater(len(grads), 0)
        has_non_zero = any(np.any(g.numpy() != 0) for g in grads if g is not None)
        self.assertTrue(has_non_zero)


if __name__ == "__main__":
    unittest.main()

