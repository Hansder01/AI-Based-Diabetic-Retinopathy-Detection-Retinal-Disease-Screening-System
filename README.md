# AI-Based Diabetic Retinopathy Detection & Retinal Disease Screening System

[![Python](https://img.shields.io/badge/Python-3.14%2B-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.22-orange.svg)](https://tensorflow.org/)
[![Keras](https://img.shields.io/badge/Keras-3.0-red.svg)](https://keras.io/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.67-brightgreen.svg)](https://streamlit.io/)

---

## 1. Project Overview

This project implements an end-to-end, clinical-grade deep learning screening framework for retinal fundus imaging based on the architecture outlined in `AI_Diabetic_Retinopathy_Detection.pptx`. 

Traditional screening AI models suffer from two critical shortfalls:
1. **The Binary Bottleneck**: Naive models classify scans as only "DR vs. No DR", completely ignoring biological severity progression and failing to provide actionable triage context.
2. **The Softmax Flaw**: Standard multi-class Softmax models treat disease grades as disjoint, penalizing a 1-grade error (e.g. Mild vs Moderate) identically to a 4-grade error (e.g. Mild vs Proliferative).

### Core Innovations
- **Ordinal Regression Head**: Uses an **EfficientNet-B0** backbone with monotonic cumulative threshold probability outputs ($P(Y > k)$) to preserve biological disease progression.
- **Hierarchical Fallback Routing**: Scans predicted as Grade 0 (No DR) are automatically routed to a secondary classifier trained on the **FIVES dataset** to detect alternative pathologies (AMD, Glaucoma, or Normal).
- **Structural Vessel Segmentation**: A custom **U-Net** architecture ($256\times 256\times 3 \to 256\times 256\times 1$) with skip connections trained using a composite BCE + Soft-Dice loss to segment the retinal microvasculature and compute **Vessel Density %**.
- **Explainable AI (Grad-CAM)**: Visual attention heatmaps computed from the final convolutional layer of EfficientNet-B0 (`top_activation`), localizing microaneurysms, hemorrhages, and exudates.
- **Clinical Safety Gate**: An explicit confidence thresholding safety mechanism ($\ge 70\%$ verified high confidence, $< 70\%$ flagged for mandatory ophthalmologist review).

---

## 2. System Architecture

```
[Input RGB Fundus Scan]
          │
          ▼
┌──────────────────────────────────────────────┐
│             PREPROCESSING ENGINE             │
│ • Circular FOV & Black Border Cropping       │
│ • CLAHE Contrast Normalization (L / Green)   │
│ • Dual-Resolution Resizing (224 & 256)       │
└──────────────────────────────────────────────┘
          │
          ├────────────────────────────────────────────────────┐
          ▼                                                    ▼
┌───────────────────────────────────┐        ┌───────────────────────────────────┐
│       PRIMARY GATEKEEPER          │        │       VESSEL SEGMENTATION         │
│     APTOS EfficientNet-B0         │        │               U-Net               │
│      Ordinal Regression           │        │         Composite Dice Loss       │
└───────────────────────────────────┘        └───────────────────────────────────┘
          │                                                    │
    Grade Prediction                                           ▼
   ┌──────┴──────────────────────────┐               ┌───────────────────┐
   │                                 │               │ Vasculature Mask  │
   ▼ (Grade 1 - 4)                   ▼ (Grade 0)     │ Vessel Density %  │
┌───────────────────────┐  ┌───────────────────────┐ └───────────────────┘
│  DR Severity Staging  │  │    FIVES Fallback     │           │
│  • Mild (1)           │  │  Secondary Classifier │           │
│  • Moderate (2)       │  │  • AMD                │           │
│  • Severe (3)         │  │  • Glaucoma           │           │
│  • Proliferative (4)  │  │  • Normal             │           │
└───────────────────────┘  └───────────────────────┘           │
          │                           │                        │
          └─────────────┬─────────────┘                        │
                        ▼                                      │
          ┌───────────────────────────┐                        │
          │    EXPLAINABLE AI         │                        │
          │    Grad-CAM Heatmap       │                        │
          └───────────────────────────┘                        │
                        │                                      │
                        ▼                                      │
          ┌───────────────────────────┐                        │
          │   CLINICAL SAFETY GATE    │                        │
          │ Confidence Score >= 70%?  │◄───────────────────────┘
          └─────────────┬─────────────┘
                        │
          ┌─────────────┴─────────────┐
          ▼ (>= 70%)                  ▼ (< 70%)
┌───────────────────────────┐ ┌────────────────────────────────────────┐
│ HIGH-CONFIDENCE DIAGNOSIS │ │ UNCERTAIN / MANUAL REVIEW RECOMMENDED  │
└───────────────────────────┘ └────────────────────────────────────────┘
```

---

## 3. Repository Structure

```
Diabetic_Rethinopaty/
├── app.py                             # Interactive Streamlit Clinical Screening Application
├── requirements.txt                   # Complete dependencies
├── README.md                          # Documentation & execution guide
├── AI_Diabetic_Retinopathy_Detection.pptx
├── .keras/models/                     # Cached ImageNet backbone weights (offline-ready)
├── checkpoints/                       # Saved trained model checkpoints (.keras)
│   ├── aptos_ordinal_best.keras       # EfficientNet-B0 Ordinal DR Model (17.1 MB)
│   ├── fives_classifier_best.keras    # EfficientNet-B0 4-Class Softmax Model (17.1 MB)
│   └── unet_vessel_best.keras         # U-Net Vessel Segmentation Model (23.7 MB)
├── exports/samples/                   # Exported visual artifacts (Grad-CAM, overlays, masks)
├── src/
│   ├── data/
│   │   ├── preprocessing.py           # Auto-crop black borders, CLAHE, standardization
│   │   └── dataset.py                 # APTOS 2019 & FIVES dataset loaders & ordinal math
│   ├── models/
│   │   ├── ordinal_model.py           # EfficientNet-B0 Ordinal Regression architecture
│   │   ├── fives_classifier.py        # 4-class Softmax disease classifier
│   │   └── unet.py                    # 4-level U-Net with skip connections
│   ├── training/
│   │   ├── losses.py                  # Ordinal BCE, Soft-Dice, BCEDiceLoss
│   │   ├── metrics.py                 # QWK, MAE, Macro F1, Confusion Matrix
│   │   ├── train_aptos_ordinal.py     # Training routine for DR severity
│   │   ├── train_fives_classifier.py  # Training routine for FIVES disease classification
│   │   └── train_unet_vessels.py      # Training routine for U-Net vessel segmentation
│   ├── explainability/
│   │   └── gradcam.py                 # Grad-CAM engine with OpenCV overlay blending
│   └── pipeline/
│       └── hierarchical_triage.py     # End-to-end multi-model clinical triage orchestrator
└── tests/
    ├── test_phase1_preprocessing.py   # Unit tests for preprocessing & dataset loaders
    ├── test_phase2_model.py           # Unit tests for Ordinal EfficientNet & loss math
    ├── test_phase3_fives_classifier.py# Unit tests for FIVES multi-disease classifier
    ├── test_phase4_unet.py            # Unit tests for U-Net & vessel density
    ├── test_phase5_gradcam.py         # Unit tests for Grad-CAM explainability
    └── test_phase6_triage.py          # Integration tests for hierarchical decision tree
```

---

## 4. Quick Start & Execution Guide

### 4.1 Running the Interactive Clinical Web Application

Launch the Streamlit web dashboard:
```bash
.\venv\Scripts\streamlit.exe run app.py
```
Or with standard Python:
```bash
python -m streamlit run app.py
```

### 4.2 Running the Automated Test Suite

Run all 26 unit and integration tests across the complete pipeline:
```bash
.\venv\Scripts\python.exe -m unittest discover tests
```

---

## 5. Mathematical Formulations

### Ordinal Regression Target Vector
For true grade $Y \in \{0, 1, 2, 3, 4\}$, the model predicts $K - 1 = 4$ monotonic binary thresholds:
$$y_k = \mathbb{I}(Y > k) \quad \text{for } k \in \{0, 1, 2, 3\}$$
- **Grade 0 (No DR)**: `[0, 0, 0, 0]`
- **Grade 1 (Mild)**: `[1, 0, 0, 0]`
- **Grade 2 (Moderate)**: `[1, 1, 0, 0]`
- **Grade 3 (Severe)**: `[1, 1, 1, 0]`
- **Grade 4 (Proliferative)**: `[1, 1, 1, 1]`

Discrete prediction is decoded via threshold sum: $\hat{Y} = \sum_{k=0}^3 \mathbb{I}(\hat{p}_k \ge 0.5)$.  
Continuous progression index: $S = \sum_{k=0}^3 \hat{p}_k \in [0.0, 4.0]$.

### Composite Segmentation Loss
$$\mathcal{L}_{\text{seg}} = 0.5 \cdot \mathcal{L}_{\text{BCE}} + 0.5 \cdot \left(1 - \frac{2 \sum y \hat{y} + 1}{\sum y + \sum \hat{y} + 1}\right)$$

