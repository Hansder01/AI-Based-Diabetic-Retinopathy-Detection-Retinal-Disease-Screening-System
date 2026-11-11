"""
Unit & Integration Tests for Phase 1: Environment & Preprocessing Engine.
Validates:
1. Mathematical correctness of Ordinal Target Encoding.
2. Circular FOV / Black Border Cropping.
3. CLAHE enhancement.
4. APTOS 2019 dataset loader with nested path resolution.
5. FIVES multi-pathology & vessel segmentation loader.
"""

import os
import unittest
import numpy as np

from src.data.preprocessing import (
    crop_black_borders,
    apply_clahe,
    normalize_image,
    preprocess_fundus_image,
    preprocess_segmentation_pair
)
from src.data.dataset import (
    grade_to_ordinal_vector,
    ordinal_vector_to_grade,
    continuous_severity_score,
    APTOSDataset,
    FIVESDataset
)


class TestOrdinalEncoding(unittest.TestCase):
    """Test suite for monotonic ordinal regression math."""

    def test_grade_to_ordinal_vectors(self):
        expected = {
            0: np.array([0., 0., 0., 0.], dtype=np.float32),
            1: np.array([1., 0., 0., 0.], dtype=np.float32),
            2: np.array([1., 1., 0., 0.], dtype=np.float32),
            3: np.array([1., 1., 1., 0.], dtype=np.float32),
            4: np.array([1., 1., 1., 1.], dtype=np.float32),
        }
        for grade, vec in expected.items():
            result = grade_to_ordinal_vector(grade)
            np.testing.assert_array_equal(result, vec, err_msg=f"Mismatch for grade {grade}")

    def test_ordinal_vector_to_grade(self):
        cases = [
            ([0.1, 0.05, 0.01, 0.0], 0),
            ([0.85, 0.2, 0.05, 0.0], 1),
            ([0.95, 0.88, 0.3, 0.1], 2),
            ([0.99, 0.95, 0.91, 0.2], 3),
            ([0.98, 0.96, 0.94, 0.91], 4),
        ]
        for probs, expected_grade in cases:
            self.assertEqual(ordinal_vector_to_grade(probs), expected_grade)

    def test_continuous_severity_score(self):
        probs = [0.9, 0.8, 0.4, 0.1]
        score = continuous_severity_score(probs)
        self.assertAlmostEqual(score, 2.2, places=4)


class TestPreprocessing(unittest.TestCase):
    """Test suite for fundus image transformations."""

    def test_synthetic_black_border_crop(self):
        # Create 100x100 synthetic image with 20px black margin
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        img[20:80, 20:80] = 200  # Central bright region

        cropped = crop_black_borders(img, tol=15)
        self.assertEqual(cropped.shape[0], 60)
        self.assertEqual(cropped.shape[1], 60)

    def test_clahe_enhancement(self):
        # Verify output maintains shape and uint8 dtype
        img = np.random.randint(50, 200, (128, 128, 3), dtype=np.uint8)
        enhanced_lab = apply_clahe(img, channel_mode="lab")
        enhanced_green = apply_clahe(img, channel_mode="green")

        self.assertEqual(enhanced_lab.shape, (128, 128, 3))
        self.assertEqual(enhanced_lab.dtype, np.uint8)
        self.assertEqual(enhanced_green.shape, (128, 128, 3))

    def test_end_to_end_fundus_preprocessing(self):
        synthetic = np.random.randint(0, 255, (300, 400, 3), dtype=np.uint8)
        processed = preprocess_fundus_image(synthetic, target_size=(224, 224), normalization="scale")

        self.assertEqual(processed.shape, (224, 224, 3))
        self.assertEqual(processed.dtype, np.float32)
        self.assertLessEqual(processed.max(), 1.0)
        self.assertGreaterEqual(processed.min(), 0.0)


class TestRealDatasetIngestion(unittest.TestCase):
    """Integration test suite verifying actual workspace dataset integrity."""

    def test_aptos_dataset_loader(self):
        if not os.path.exists("APTOS_2019"):
            self.skipTest("APTOS_2019 directory not found")

        dataset = APTOSDataset(base_dir="APTOS_2019", split="valid", target_size=(224, 224))
        self.assertGreater(len(dataset), 0)

        image, ordinal_vec, grade = dataset.get_sample(0)
        self.assertEqual(image.shape, (224, 224, 3))
        self.assertEqual(ordinal_vec.shape, (4,))
        self.assertIn(grade, [0, 1, 2, 3, 4])

        dist = dataset.get_class_distribution()
        self.assertGreater(sum(dist.values()), 0)
        print("\nAPTOS Valid Class Distribution:", dist)

    def test_fives_dataset_loader(self):
        if not os.path.exists("FIVES_Data"):
            self.skipTest("FIVES_Data directory not found")

        dataset = FIVESDataset(base_dir="FIVES_Data", split="train")
        self.assertGreater(len(dataset), 0)

        # Classification sample test
        cls_img, class_idx, code = dataset.get_classification_sample(0)
        self.assertEqual(cls_img.shape, (224, 224, 3))
        self.assertIn(class_idx, [0, 1, 2, 3])
        self.assertIn(code, ["A", "D", "G", "N"])

        # Segmentation sample test
        seg_img, mask = dataset.get_segmentation_sample(0)
        self.assertEqual(seg_img.shape, (256, 256, 3))
        self.assertEqual(mask.shape, (256, 256, 1))
        # Mask must be binary
        unique_vals = np.unique(mask)
        for v in unique_vals:
            self.assertIn(v, [0.0, 1.0])

        dist = dataset.get_class_distribution()
        print("FIVES Train Class Distribution:", dist)


if __name__ == "__main__":
    unittest.main()

