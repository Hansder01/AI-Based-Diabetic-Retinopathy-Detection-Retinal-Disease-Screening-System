"""
AI-Based Diabetic Retinopathy Detection & Retinal Disease Screening System
Clinical Web Application (Streamlit)

Features:
1. Team Attribution & Presentation Blueprint (Team: No Bugs Attached).
2. Hierarchical Decision Routing (APTOS Ordinal Regression -> FIVES Multi-disease Fallback).
3. Structural Vessel Analysis (U-Net Vasculature Segmentation & Vessel Density %).
4. Explainable AI (Grad-CAM Lesion Attention Heatmap).
5. Slide 10 Clinical Safety Gate (>= 70% Confidence Thresholding).
"""

import os
import sys
from pathlib import Path

# Ensure workspace is on sys.path and KERAS_HOME points to workspace cache
workspace_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(workspace_dir))
workspace_keras = workspace_dir / ".keras"
if workspace_keras.exists():
    os.environ["KERAS_HOME"] = str(workspace_keras)

import streamlit as st
import numpy as np
import cv2
from PIL import Image

from src.pipeline.hierarchical_triage import HierarchicalTriageEngine
from src.data.dataset import APTOSDataset, FIVESDataset


# ---------------------------------------------------------------------
# Page Configuration & Medical Styling
# ---------------------------------------------------------------------
st.set_page_config(
    page_title="AI Diabetic Retinopathy Screening System",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #4B5563;
        margin-bottom: 1.2rem;
    }
    .team-badge {
        background-color: #EFF6FF;
        border-left: 4px solid #2563EB;
        padding: 8px 14px;
        font-size: 0.88rem;
        color: #1E40AF;
        border-radius: 4px;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 14px;
        text-align: center;
    }
    .badge-safe {
        background-color: #DCFCE7;
        color: #166534;
        font-weight: 700;
        padding: 6px 12px;
        border-radius: 20px;
        border: 1px solid #86EFAC;
    }
    .badge-warn {
        background-color: #FEE2E2;
        color: #991B1B;
        font-weight: 700;
        padding: 6px 12px;
        border-radius: 20px;
        border: 1px solid #FCA5A5;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------
# Cached Model Engine Initialization
# ---------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading Clinical AI Models & Checkpoints...")
def get_triage_engine(confidence_threshold: float = 0.70) -> HierarchicalTriageEngine:
    return HierarchicalTriageEngine(
        ordinal_model_path="checkpoints/aptos_ordinal_best.keras",
        fives_model_path="checkpoints/fives_classifier_best.keras",
        unet_model_path="checkpoints/unet_vessel_best.keras",
        confidence_threshold=confidence_threshold
    )


# ---------------------------------------------------------------------
# Sidebar: Controls & Sample Loading
# ---------------------------------------------------------------------
st.sidebar.title("Clinical Controls & Configuration")

conf_slider = st.sidebar.slider(
    "Safety Confidence Threshold",
    min_value=0.50,
    max_value=0.90,
    value=0.70,
    step=0.05,
    help="Slide 10 Clinical Safety Gate: Scans below this confidence score mandate formal ophthalmologist review."
)

st.sidebar.markdown("---")
st.sidebar.subheader("Input Retinal Scan")

input_mode = st.sidebar.radio(
    "Select Input Method",
    ["Select Preloaded Dataset Sample", "Upload Custom Fundus Scan"]
)

selected_image = None
sample_info = ""

if input_mode == "Select Preloaded Dataset Sample":
    dataset_choice = st.sidebar.selectbox(
        "Choose Cohort",
        ["APTOS 2019 Validation (DR Staging)", "FIVES Test (AMD, DR, Glaucoma, Normal)"]
    )

    if dataset_choice == "APTOS 2019 Validation (DR Staging)":
        aptos_ds = APTOSDataset(base_dir="APTOS_2019", split="valid")
        sample_idx = st.sidebar.slider("Sample Index", 0, min(50, len(aptos_ds) - 1), 0)
        img_path = aptos_ds.get_image_path(sample_idx)
        true_grade = aptos_ds.df.iloc[sample_idx]["diagnosis"]
        selected_image = Image.open(img_path)
        sample_info = f"APTOS 2019 Validation Scan #{sample_idx} | Ground Truth: Grade {true_grade} ({APTOSDataset.SEVERITY_NAMES[true_grade]})"

    else:
        fives_ds = FIVESDataset(base_dir="FIVES_Data", split="test")
        sample_idx = st.sidebar.slider("Sample Index", 0, min(50, len(fives_ds) - 1), 0)
        item = fives_ds.samples[sample_idx]
        selected_image = Image.open(item["image_path"])
        sample_info = f"FIVES Cohort Scan ({item['filename']}) | Ground Truth: {FIVESDataset.DISEASE_NAMES[item['class_idx']]}"

else:
    uploaded_file = st.sidebar.file_uploader(
        "Upload Retinal Fundus Scan (PNG, JPG)",
        type=["png", "jpg", "jpeg"]
    )
    if uploaded_file is not None:
        selected_image = Image.open(uploaded_file)
        sample_info = f"Uploaded Scan: {uploaded_file.name}"

st.sidebar.markdown("---")
st.sidebar.markdown("""
**Model Checkpoints Active:**
- APTOS Ordinal (EfficientNet-B0)
- FIVES Multi-Pathology (Softmax)
- Retinal Vessel U-Net (Dice Loss)
- Grad-CAM Interpretability Engine
""")


# ---------------------------------------------------------------------
# Main Interface: Header & Banner
# ---------------------------------------------------------------------
st.markdown('<div class="main-header">AI-Based Diabetic Retinopathy Screening System</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Deep Learning for Ordinal Severity Grading, Disease Analysis, and Vessel Segmentation in Retinal Fundus Images</div>', unsafe_allow_html=True)

# ---------------------------------------------------------------------
# Clinical Inference Execution
# ---------------------------------------------------------------------
if selected_image is not None:
    st.info(f"**Loaded Input:** {sample_info}")

    engine = get_triage_engine(confidence_threshold=conf_slider)

    with st.spinner("Analyzing scan across hierarchical diagnostic pipeline..."):
        # Convert PIL to RGB uint8
        img_np = np.array(selected_image.convert("RGB"))
        report = engine.diagnose(img_np, run_segmentation=True, run_explainability=True)

    summary = report["summary"]
    dr_data = report["dr_ordinal_grading"]
    fallback = report["fallback_pathology"]
    vasc = report["vasculature"]
    vis = report["visual_artifacts"]

    # -----------------------------------------------------------------
    # Section 1: Clinical Triage Banner & Safety Gate
    # -----------------------------------------------------------------
    col_triage1, col_triage2 = st.columns([2, 1])

    with col_triage1:
        st.subheader("Diagnostic Finding")
        st.markdown(f"### **{summary['primary_diagnosis']}**")
        st.markdown(f"**Decision Pathway:** `{summary['pathway_taken']}`")

    with col_triage2:
        st.subheader("Clinical Safety Status")
        if summary["requires_manual_review"]:
            st.markdown(f"""
            <div class="metric-card" style="border: 2px solid #EF4444;">
                <span class="badge-warn">MANUAL REVIEW RECOMMENDED</span>
                <p style="margin-top: 8px; font-size: 0.9rem; color: #7F1D1D;">
                    Prediction Confidence: <strong>{summary['confidence_percent']}%</strong> (Below {int(conf_slider*100)}% Threshold)
                </p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="metric-card" style="border: 2px solid #22C55E;">
                <span class="badge-safe">HIGH CONFIDENCE VERIFIED</span>
                <p style="margin-top: 8px; font-size: 0.9rem; color: #14532D;">
                    Prediction Confidence: <strong>{summary['confidence_percent']}%</strong> (&ge; {int(conf_slider*100)}% Threshold)
                </p>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")

    # -----------------------------------------------------------------
    # Section 2: Quantitative Metrics Cards
    # -----------------------------------------------------------------
    m_col1, m_col2, m_col3, m_col4 = st.columns(4)

    with m_col1:
        st.metric(
            label="DR Severity Grade",
            value=f"Grade {dr_data['grade']}",
            delta=dr_data["stage_name"]
        )

    with m_col2:
        st.metric(
            label="Continuous DR Progression Index",
            value=f"{dr_data['continuous_score']} / 4.0",
            help="Cumulative probability index preserving sub-stage severity continuous progress"
        )

    with m_col3:
        st.metric(
            label="Retinal Vessel Density",
            value=f"{vasc['vessel_density_percent']}%",
            delta="Vasculature Health"
        )

    with m_col4:
        st.metric(
            label="Lesion Attention Hotspots",
            value=f"{vis['hotspot_fraction'] * 100:.1f}%",
            delta="Grad-CAM Focus"
        )

    st.markdown("---")

    # -----------------------------------------------------------------
    # Section 3: Visual Diagnostic Quartet (Side-by-Side)
    # -----------------------------------------------------------------
    st.subheader("Multimodal Diagnostic Visualizations")

    v_col1, v_col2, v_col3 = st.columns(3)

    with v_col1:
        st.markdown("**1. Preprocessed Retinal Scan**")
        st.image(vis["cropped_fundus"], caption="Cropped FOV & CLAHE Enhanced", use_container_width=True)

    with v_col2:
        st.markdown("**2. Grad-CAM Lesion Heatmap**")
        st.image(vis["gradcam_overlay"], caption="Attention Localization (Microaneurysms / Exudates)", use_container_width=True)

    with v_col3:
        st.markdown("**3. U-Net Vessel Segmentation**")
        st.image(vis["vessel_overlay"], caption="Segmented Retinal Microvasculature", use_container_width=True)

    st.markdown("---")

    # -----------------------------------------------------------------
    # Section 4: Detailed Probability Breakdown & Clinical Advisory
    # -----------------------------------------------------------------
    d_col1, d_col2 = st.columns(2)

    with d_col1:
        st.subheader("Ordinal Cumulative Probabilities (APTOS)")
        st.markdown("""
        *Ordinal Regression formulation $P(Y > k)$ forces the model to respect clinical disease progression.*
        """)
        threshold_names = ["P(Y > 0: Any DR)", "P(Y > 1: Moderate+)", "P(Y > 2: Severe+)", "P(Y > 3: Proliferative)"]
        for name, prob in zip(threshold_names, dr_data["ordinal_probs"]):
            st.progress(min(1.0, max(0.0, prob)), text=f"{name}: {prob * 100:.1f}%")

        if fallback:
            st.markdown("**Secondary Pathology Probabilities (FIVES Fallback):**")
            for disease, prob in fallback["probabilities"].items():
                st.write(f"- **{disease}:** {prob * 100:.1f}%")

    with d_col2:
        st.subheader("Actionable Clinical Recommendation")
        st.info(summary["clinical_recommendation"])

        st.markdown("""
        **Clinical Notes for Screener:**
        - Ordinal regression preserves strict monotonic staging; a 1-grade error is penalized less than a 3-grade error.
        - U-Net vessel density serves as a biomarker for microvascular pruning and ischemic zones.
        - Grad-CAM heatmap ensures transparency before initiating clinical follow-ups.
        """)

else:
    st.warning("Please select or upload a fundus scan from the left sidebar to begin screening.")

