"""
Unit Tests for Phase 5: Grad-CAM Explainable AI (XAI) & Lesion Localization.
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
import cv2

from src.models.ordinal_model import build_ordinal_model
from src.models.fives_classifier import build_fives_classifier
from src.explainability.gradcam import GradCAMVisualizer
from src.data.dataset import APTOSDataset


class TestPhase5GradCAM(unittest.TestCase):
    """Test suite for Grad-CAM explainability engine."""

    @classmethod
    def setUpClass(cls):
        cls.ordinal_model = build_ordinal_model(input_shape=(224, 224, 3), num_classes=5)
        cls.fives_model = build_fives_classifier(input_shape=(224, 224, 3), num_classes=4)

    def test_gradcam_initialization_and_layer_resolution(self):
        visualizer = GradCAMVisualizer(self.ordinal_model, target_conv_layer="top_activation")
        self.assertEqual(visualizer.target_conv_layer, "top_activation")
        self.assertIsNotNone(visualizer.feature_extractor)

    def test_compute_heatmap_ordinal(self):
        visualizer = GradCAMVisualizer(self.ordinal_model, target_conv_layer="top_activation")
        dummy_tensor = np.random.uniform(0.0, 1.0, (224, 224, 3)).astype(np.float32)

        heatmap = visualizer.compute_heatmap(dummy_tensor, target_index=1)
        self.assertEqual(heatmap.shape, (7, 7))
        self.assertGreaterEqual(float(np.min(heatmap)), 0.0)
        self.assertLessEqual(float(np.max(heatmap)), 1.0)

    def test_explain_end_to_end_and_overlay(self):
        visualizer = GradCAMVisualizer(self.ordinal_model, target_conv_layer="top_activation")

        # Synthetic fundus image with central lesion
        synthetic_img = np.full((300, 300, 3), 100, dtype=np.uint8)
        # Lesion in center
        synthetic_img[120:160, 120:160] = [255, 20, 20]

        res = visualizer.explain(synthetic_img, target_index=0, alpha=0.4)

        required_keys = ["heatmap_raw", "heatmap_224", "overlay_image", "preprocessed_image", "hotspot_fraction"]
        for k in required_keys:
            self.assertIn(k, res)

        self.assertEqual(res["heatmap_raw"].shape, (7, 7))
        self.assertEqual(res["heatmap_224"].shape, (224, 224))
        self.assertEqual(res["overlay_image"].shape, (224, 224, 3))
        self.assertEqual(res["overlay_image"].dtype, np.uint8)
        self.assertGreaterEqual(res["hotspot_fraction"], 0.0)
        self.assertLessEqual(res["hotspot_fraction"], 1.0)

    def test_gradcam_on_fives_model(self):
        visualizer = GradCAMVisualizer(self.fives_model, target_conv_layer="top_activation")
        dummy_tensor = np.random.uniform(0.0, 1.0, (224, 224, 3)).astype(np.float32)

        heatmap = visualizer.compute_heatmap(dummy_tensor, target_index=0)
        self.assertEqual(heatmap.shape, (7, 7))
        self.assertTrue(np.all(heatmap >= 0.0))

    def test_gradcam_on_real_fundus_scan(self):
        if not os.path.exists("APTOS_2019"):
            self.skipTest("APTOS_2019 not present")

        ds = APTOSDataset(base_dir="APTOS_2019", split="valid")
        img_path = ds.get_image_path(0)

        visualizer = GradCAMVisualizer(self.ordinal_model, target_conv_layer="top_activation")
        res = visualizer.explain(img_path, target_index=0)

        os.makedirs("exports/samples", exist_ok=True)
        cv2.imwrite("exports/samples/gradcam_sample.png", cv2.cvtColor(res["overlay_image"], cv2.COLOR_RGB2BGR))
        cv2.imwrite("exports/samples/gradcam_heatmap_sample.png", (res["heatmap_224"] * 255).astype(np.uint8))
        self.assertTrue(os.path.exists("exports/samples/gradcam_sample.png"))


if __name__ == "__main__":
    unittest.main()

