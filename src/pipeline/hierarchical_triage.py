"""
Hierarchical Inference and Clinical Triage Engine (Slide 10).

Orchestrates:
1. Input Image Ingestion & Preprocessing (Crop black borders + CLAHE + Dual Resize).
2. Step 1 (Primary Gatekeeper): APTOS EfficientNet-B0 Ordinal Severity Classifier.
   - If Grade 1-4: Triggers DR Positive Pathway with progressive severity staging.
   - If Grade 0: Routes to Step 2.
3. Step 2 (FIVES Fallback): EfficientNet-B0 4-Class Softmax Classifier for AMD, Glaucoma, or Normal.
4. Step 3 (Structural Vessel Analysis): U-Net Retinal Vessel Segmentation & Vessel Density %.
5. Step 4 (Visual Explainability): Grad-CAM attention heatmap overlay.
6. Step 5 (Safety Mechanism): Confidence thresholding (>= 70% High Confidence, < 70% Manual Review Recommended).
"""

from typing import Tuple, Dict, Any, Union, Optional
import os
from pathlib import Path
import numpy as np
import cv2

# Ensure workspace .keras is respected
workspace_keras = Path(__file__).resolve().parents[2] / ".keras"
if workspace_keras.exists():
    os.environ.setdefault("KERAS_HOME", str(workspace_keras))

from src.data.preprocessing import (
    crop_black_borders,
    apply_clahe,
    load_image,
    preprocess_fundus_image
)
from src.models.ordinal_model import build_ordinal_model, OrdinalClassifierWrapper
from src.models.fives_classifier import build_fives_classifier, FIVESClassifierWrapper
from src.models.unet import build_unet, UNetVesselSegmentationWrapper
from src.explainability.gradcam import GradCAMVisualizer


# Clinical Recommendations Engine
CLINICAL_RECOMMENDATIONS = {
    "No DR - Normal": "Routine annual screening. No active retinal diabetic lesions or glaucomatous neuropathy detected.",
    "No DR - AMD": "Suspected Age-Related Macular Degeneration. Urgently refer to retina specialist for Optical Coherence Tomography (OCT) scan and macular evaluation.",
    "No DR - Glaucoma": "Suspected Glaucomatous changes. Refer for visual field perimetry test and intraocular pressure (IOP) measurement.",
    "Mild DR": "Mild Non-Proliferative Diabetic Retinopathy (NPDR). Optimize glycemic and blood pressure control. Follow-up retinal exam in 6-12 months.",
    "Moderate DR": "Moderate NPDR with evident microvascular anomalies. Refer to ophthalmologist for evaluation within 3-6 months.",
    "Severe DR": "Severe NPDR (high risk of proliferative progression). Expedited referral to vitreoretinal specialist within 2-4 weeks.",
    "Proliferative DR": "Proliferative Diabetic Retinopathy (PDR) with active neovascularization. Urgent vitreoretinal referral for Panretinal Photocoagulation (PRP) or anti-VEGF therapy.",
    "Uncertain": "Prediction falls below the 70% confidence safety threshold. Formal comprehensive examination by a board-certified ophthalmologist is mandated."
}


class HierarchicalTriageEngine:
    """
    Unified diagnostic and triage orchestrator matching Slide 10.
    """

    def __init__(
        self,
        ordinal_model_path: Optional[str] = None,
        fives_model_path: Optional[str] = None,
        unet_model_path: Optional[str] = None,
        confidence_threshold: float = 0.70
    ):
        self.confidence_threshold = confidence_threshold

        # Load or build models
        print("[Hierarchical Engine] Initializing diagnostic sub-systems...")

        # 1. Primary Gatekeeper (APTOS Ordinal DR)
        if ordinal_model_path and os.path.exists(ordinal_model_path):
            self.ordinal_wrapper = OrdinalClassifierWrapper.load_from_checkpoint(
                ordinal_model_path, confidence_threshold=confidence_threshold
            )
        else:
            m1 = build_ordinal_model(input_shape=(224, 224, 3), num_classes=5)
            self.ordinal_wrapper = OrdinalClassifierWrapper(m1, confidence_threshold=confidence_threshold)

        # 2. Secondary Fallback (FIVES Multi-Disease)
        if fives_model_path and os.path.exists(fives_model_path):
            self.fives_wrapper = FIVESClassifierWrapper.load_from_checkpoint(
                fives_model_path, confidence_threshold=confidence_threshold
            )
        else:
            m2 = build_fives_classifier(input_shape=(224, 224, 3), num_classes=4)
            self.fives_wrapper = FIVESClassifierWrapper(m2, confidence_threshold=confidence_threshold)

        # 3. Structural Vessel Segmentation (U-Net)
        if unet_model_path and os.path.exists(unet_model_path):
            self.unet_wrapper = UNetVesselSegmentationWrapper.load_from_checkpoint(
                unet_model_path, threshold=0.5
            )
        else:
            m3 = build_unet(input_shape=(256, 256, 3), initial_filters=16)
            self.unet_wrapper = UNetVesselSegmentationWrapper(m3, threshold=0.5)

        # 4. Explainable AI (Grad-CAM)
        self.gradcam_dr = GradCAMVisualizer(self.ordinal_wrapper.model, target_conv_layer="top_activation")
        self.gradcam_fives = GradCAMVisualizer(self.fives_wrapper.model, target_conv_layer="top_activation")

        print("[Hierarchical Engine] All clinical models loaded successfully.")

    def diagnose(
        self,
        image_source: Union[str, np.ndarray],
        run_segmentation: bool = True,
        run_explainability: bool = True
    ) -> Dict[str, Any]:
        """
        Executes complete multi-model hierarchical inference pipeline.

        Args:
            image_source: File path or RGB uint8 numpy array.
            run_segmentation: Whether to run U-Net vessel analysis.
            run_explainability: Whether to generate Grad-CAM overlays.

        Returns:
            Comprehensive clinical diagnostic report dictionary.
        """
        raw_rgb = load_image(image_source)

        # Preprocessing Lens (Slide 6)
        cropped_rgb = crop_black_borders(raw_rgb)
        enhanced_rgb = apply_clahe(cropped_rgb, channel_mode="lab")

        # Classification input (224x224, normalized [0, 1])
        cls_input = cv2.resize(enhanced_rgb, (224, 224), interpolation=cv2.INTER_AREA)
        cls_norm = cls_input.astype(np.float32) / 255.0

        # -------------------------------------------------------------
        # STEP 1: The Primary Gatekeeper (APTOS Ordinal Regression)
        # -------------------------------------------------------------
        dr_result = self.ordinal_wrapper.predict_single(cls_norm)
        dr_grade = dr_result["grade"]
        dr_confidence = dr_result["confidence"]

        # Decision Tree Logic (Slide 10)
        fallback_result = None
        if dr_grade > 0:
            # Pathway A: Diabetic Retinopathy Positive
            pathway = "DR Positive + Severity Pathway (APTOS)"
            primary_diagnosis = f"Diabetic Retinopathy - Grade {dr_grade} ({dr_result['stage_name']})"
            active_confidence = dr_confidence
            rec_key = dr_result["stage_name"] + " DR"
            target_gradcam_model = self.gradcam_dr
            target_gradcam_idx = max(0, dr_grade - 1)
        else:
            # Pathway B: Grade 0 Fallback to FIVES Classifier
            pathway = "Alternative Pathology Fallback (FIVES)"
            fallback_result = self.fives_wrapper.predict_single(cls_norm)
            fives_code = fallback_result["disease_code"]
            active_confidence = fallback_result["confidence"]

            if fives_code == "Normal":
                primary_diagnosis = "No Diabetic Retinopathy Detected (Normal Retinal Exam)"
                rec_key = "No DR - Normal"
            elif fives_code == "AMD":
                primary_diagnosis = "Suspected Age-Related Macular Degeneration (AMD)"
                rec_key = "No DR - AMD"
            elif fives_code == "Glaucoma":
                primary_diagnosis = "Suspected Glaucoma"
                rec_key = "No DR - Glaucoma"
            else:
                primary_diagnosis = "Diabetic Retinopathy (Detected via secondary screening)"
                rec_key = "Mild DR"

            target_gradcam_model = self.gradcam_fives
            target_gradcam_idx = fallback_result["predicted_class"]

        # -------------------------------------------------------------
        # STEP 2: Confidence Thresholding (Safety Mechanism)
        # -------------------------------------------------------------
        is_high_conf = bool(active_confidence >= self.confidence_threshold)
        if is_high_conf:
            triage_badge = "High Confidence"
            triage_color = "green"
            clinical_recommendation = CLINICAL_RECOMMENDATIONS.get(rec_key, "Follow clinical discretion.")
        else:
            triage_badge = "Uncertain / Manual Review Recommended"
            triage_color = "red"
            clinical_recommendation = CLINICAL_RECOMMENDATIONS["Uncertain"]

        # -------------------------------------------------------------
        # STEP 3: Structural Vessel Analysis (U-Net)
        # -------------------------------------------------------------
        vasculature_data = {}
        if run_segmentation:
            vasculature_data = self.unet_wrapper.analyze_vasculature(cropped_rgb, target_size=(256, 256))

        # -------------------------------------------------------------
        # STEP 4: Visual Explainability (Grad-CAM)
        # -------------------------------------------------------------
        explainability_data = {}
        if run_explainability:
            explainability_data = target_gradcam_model.explain(
                cropped_rgb,
                target_index=target_gradcam_idx,
                alpha=0.42
            )

        # Assemble Comprehensive Clinical Report
        report = {
            "summary": {
                "primary_diagnosis": primary_diagnosis,
                "triage_badge": triage_badge,
                "triage_color": triage_color,
                "confidence_percent": round(active_confidence * 100.0, 1),
                "pathway_taken": pathway,
                "clinical_recommendation": clinical_recommendation,
                "requires_manual_review": not is_high_conf
            },
            "dr_ordinal_grading": dr_result,
            "fallback_pathology": fallback_result,
            "vasculature": {
                "vessel_density_percent": vasculature_data.get("vessel_density_percent", 0.0),
                "vessel_pixel_count": vasculature_data.get("vessel_pixel_count", 0),
                "retina_area_pixels": vasculature_data.get("retina_area_pixels", 0)
            },
            "visual_artifacts": {
                "cropped_fundus": cropped_rgb,
                "preprocessed_224": cls_input,
                "vessel_mask": vasculature_data.get("binary_mask", None),
                "vessel_overlay": vasculature_data.get("overlay_image", None),
                "gradcam_heatmap": explainability_data.get("heatmap_224", None),
                "gradcam_overlay": explainability_data.get("overlay_image", None),
                "hotspot_fraction": explainability_data.get("hotspot_fraction", 0.0)
            }
        }
        return report

