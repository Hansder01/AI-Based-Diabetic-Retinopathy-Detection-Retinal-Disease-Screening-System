"""
Integration Tests for Phase 6: Hierarchical Inference & Clinical Safety Triage Engine.
"""

import os
from pathlib import Path

# Ensure KERAS_HOME points to workspace cache
workspace_keras = Path(__file__).resolve().parents[1] / ".keras"
if workspace_keras.exists():
    os.environ["KERAS_HOME"] = str(workspace_keras)

import unittest
import numpy as np

from src.pipeline.hierarchical_triage import HierarchicalTriageEngine
from src.data.dataset import APTOSDataset, FIVESDataset


class TestPhase6HierarchicalTriage(unittest.TestCase):
    """Test suite for Slide 10 hierarchical decision pipeline."""

    @classmethod
    def setUpClass(cls):
        cls.engine = HierarchicalTriageEngine(
            ordinal_model_path="checkpoints/aptos_ordinal_best.keras",
            fives_model_path="checkpoints/fives_classifier_best.keras",
            unet_model_path="checkpoints/unet_vessel_best.keras",
            confidence_threshold=0.70
        )

    def test_engine_initialization(self):
        self.assertIsNotNone(self.engine.ordinal_wrapper)
        self.assertIsNotNone(self.engine.fives_wrapper)
        self.assertIsNotNone(self.engine.unet_wrapper)
        self.assertIsNotNone(self.engine.gradcam_dr)
        self.assertIsNotNone(self.engine.gradcam_fives)

    def test_triage_on_synthetic_scan(self):
        # Create synthetic fundus image
        synthetic = np.full((300, 300, 3), 120, dtype=np.uint8)
        report = self.engine.diagnose(synthetic, run_segmentation=True, run_explainability=True)

        self.assertIn("summary", report)
        self.assertIn("dr_ordinal_grading", report)
        self.assertIn("vasculature", report)
        self.assertIn("visual_artifacts", report)

        # Check summary structure
        s = report["summary"]
        self.assertIn("primary_diagnosis", s)
        self.assertIn("triage_badge", s)
        self.assertIn("confidence_percent", s)
        self.assertIn("pathway_taken", s)
        self.assertIn("clinical_recommendation", s)
        self.assertIn("requires_manual_review", s)

        # Safety gate verification
        if s["confidence_percent"] >= 70.0:
            self.assertEqual(s["triage_badge"], "High Confidence")
            self.assertFalse(s["requires_manual_review"])
        else:
            self.assertEqual(s["triage_badge"], "Uncertain / Manual Review Recommended")
            self.assertTrue(s["requires_manual_review"])

        # Check visual artifacts
        vis = report["visual_artifacts"]
        self.assertIsNotNone(vis["vessel_mask"])
        self.assertIsNotNone(vis["vessel_overlay"])
        self.assertIsNotNone(vis["gradcam_overlay"])
        self.assertEqual(vis["vessel_mask"].shape, (256, 256))
        self.assertEqual(vis["gradcam_overlay"].shape, (224, 224, 3))

    def test_triage_on_real_aptos_scan(self):
        if not os.path.exists("APTOS_2019"):
            self.skipTest("APTOS_2019 not available")

        ds = APTOSDataset(base_dir="APTOS_2019", split="valid")
        sample_path = ds.get_image_path(0)
        report = self.engine.diagnose(sample_path)

        print("\n--- APTOS Validation Sample 0 Triage Report ---")
        print("Primary Diagnosis:", report["summary"]["primary_diagnosis"])
        print("Pathway:", report["summary"]["pathway_taken"])
        print("Confidence:", f"{report['summary']['confidence_percent']}%")
        print("Triage Badge:", report["summary"]["triage_badge"])
        print("Vessel Density:", f"{report['vasculature']['vessel_density_percent']}%")
        print("Clinical Advice:", report["summary"]["clinical_recommendation"])

        self.assertGreater(report["vasculature"]["vessel_density_percent"], 0.0)

    def test_triage_on_fives_scan(self):
        if not os.path.exists("FIVES_Data"):
            self.skipTest("FIVES_Data not available")

        ds = FIVESDataset(base_dir="FIVES_Data", split="test")
        sample_path = ds.samples[0]["image_path"]
        report = self.engine.diagnose(sample_path)

        print("\n--- FIVES Test Sample 0 Triage Report ---")
        print("Primary Diagnosis:", report["summary"]["primary_diagnosis"])
        print("Pathway:", report["summary"]["pathway_taken"])
        print("Triage Badge:", report["summary"]["triage_badge"])

        self.assertIsNotNone(report["visual_artifacts"]["vessel_overlay"])


if __name__ == "__main__":
    unittest.main()

