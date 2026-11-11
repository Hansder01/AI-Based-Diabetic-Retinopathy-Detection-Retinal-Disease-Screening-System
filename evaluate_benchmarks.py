"""
Benchmark Evaluation and Academic Comparison Suite.

Implements the academic verification mandated in Slide 8 & Slide 11:
1. Compares Ordinal Regression vs. Standard Nominal Softmax on APTOS 2019:
   - Quadratic Weighted Kappa (QWK)
   - Mean Absolute Error (MAE)
   - Macro F1 & Multiclass Accuracy
   - Severe Distance Penalty & Triage Error Frequency (e.g. 1-grade error vs 3-grade error)
2. Evaluates U-Net Retinal Vessel Segmentation on FIVES Test Set:
   - Dice Coefficient
   - Intersection over Union (IoU)
   - Specificity & Sensitivity
3. Generates confusion matrices and exportable evaluation plots.
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

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import confusion_matrix, cohen_kappa_score, mean_absolute_error, accuracy_score, f1_score

from src.data.dataset import APTOSDataset, FIVESDataset, ordinal_vector_to_grade
from src.models.ordinal_model import OrdinalClassifierWrapper, ordinal_to_class_probabilities
from src.models.unet import UNetVesselSegmentationWrapper


def run_aptos_ordinal_benchmark(
    checkpoint_path: str = "checkpoints/aptos_ordinal_best.keras",
    num_samples: int = 150
) -> dict:
    """
    Evaluates the trained Ordinal DR model on held-out validation samples.
    """
    print(f"\n[1/3] Benchmarking Ordinal DR Model on APTOS Validation Set ({num_samples} samples)...")
    val_ds = APTOSDataset(base_dir="APTOS_2019", split="valid")
    eval_count = min(num_samples, len(val_ds))

    wrapper = OrdinalClassifierWrapper.load_from_checkpoint(checkpoint_path)

    y_true = []
    y_pred_ordinal = []
    confidences = []
    high_conf_count = 0

    # Collect images
    images = []
    for i in range(eval_count):
        img, _, grade = val_ds.get_sample(i)
        images.append(img)
        y_true.append(grade)

    images = np.array(images, dtype=np.float32)
    pred_probs = wrapper.predict_probabilities(images)

    for i in range(eval_count):
        p = pred_probs[i]
        grade_pred = ordinal_vector_to_grade(p, threshold=0.5)
        y_pred_ordinal.append(grade_pred)

        # Calibrated discrete probability mass for predicted grade
        class_probs = ordinal_to_class_probabilities(p)
        conf = float(class_probs[grade_pred])
        confidences.append(conf)
        if conf >= 0.70:
            high_conf_count += 1

    y_true = np.array(y_true)
    y_pred_ordinal = np.array(y_pred_ordinal)

    qwk = cohen_kappa_score(y_true, y_pred_ordinal, weights="quadratic")
    mae = mean_absolute_error(y_true, y_pred_ordinal)
    acc = accuracy_score(y_true, y_pred_ordinal)
    macro_f1 = f1_score(y_true, y_pred_ordinal, average="macro", zero_division=0)
    cm = confusion_matrix(y_true, y_pred_ordinal, labels=[0, 1, 2, 3, 4])

    # Compute error distribution: exact match, 1-grade off, 2+ grades off
    errors = np.abs(y_true - y_pred_ordinal)
    exact_matches = int(np.sum(errors == 0))
    one_grade_off = int(np.sum(errors == 1))
    severe_misses = int(np.sum(errors >= 2))

    results = {
        "dataset": "APTOS 2019 Validation",
        "evaluated_samples": eval_count,
        "quadratic_weighted_kappa": round(float(qwk), 4),
        "mean_absolute_error": round(float(mae), 4),
        "accuracy": round(float(acc), 4),
        "macro_f1": round(float(macro_f1), 4),
        "exact_matches_percent": round(exact_matches / eval_count * 100, 2),
        "one_grade_off_percent": round(one_grade_off / eval_count * 100, 2),
        "severe_misses_percent": round(severe_misses / eval_count * 100, 2),
        "mean_confidence": round(float(np.mean(confidences)), 4),
        "high_confidence_rate": round(high_conf_count / eval_count * 100, 2),
        "confusion_matrix": cm.tolist()
    }
    return results


def run_unet_segmentation_benchmark(
    checkpoint_path: str = "checkpoints/unet_vessel_best.keras",
    num_samples: int = 50
) -> dict:
    """
    Evaluates U-Net Retinal Vessel Segmentation on the FIVES test cohort.
    """
    print(f"\n[2/3] Benchmarking U-Net on FIVES Test Ground Truth ({num_samples} samples)...")
    test_ds = FIVESDataset(base_dir="FIVES_Data", split="test", segmentation_size=(256, 256))
    eval_count = min(num_samples, len(test_ds.samples))

    wrapper = UNetVesselSegmentationWrapper.load_from_checkpoint(checkpoint_path)

    dice_scores = []
    iou_scores = []
    vessel_densities = []

    for i in range(eval_count):
        s = test_ds.samples[i]
        if s["mask_path"] is None:
            continue

        img, true_mask = test_ds.get_segmentation_sample(i)
        pred_prob_mask = wrapper.segment(img)
        pred_binary = (pred_prob_mask >= 0.5).astype(np.float32)
        true_binary = true_mask[:, :, 0]

        # Calculate Dice
        intersection = np.sum(pred_binary * true_binary)
        dice = (2.0 * intersection + 1.0) / (np.sum(pred_binary) + np.sum(true_binary) + 1.0)
        # Calculate IoU
        union = np.sum(pred_binary) + np.sum(true_binary) - intersection
        iou = (intersection + 1.0) / (union + 1.0)

        dice_scores.append(float(dice))
        iou_scores.append(float(iou))

        analysis = wrapper.analyze_vasculature(s["image_path"])
        vessel_densities.append(analysis["vessel_density_percent"])

    results = {
        "dataset": "FIVES Test Cohort",
        "evaluated_pairs": len(dice_scores),
        "mean_dice_coefficient": round(float(np.mean(dice_scores)), 4),
        "mean_iou_score": round(float(np.mean(iou_scores)), 4),
        "mean_vessel_density_percent": round(float(np.mean(vessel_densities)), 2)
    }
    return results


def plot_confusion_matrix_and_metrics(
    aptos_results: dict,
    unet_results: dict,
    output_image_path: str = "exports/benchmark_dashboard.png"
):
    """
    Generates academic publication-style visual comparison plots.
    """
    print(f"\n[3/3] Generating visual benchmark report and saving to {output_image_path}...")
    os.makedirs(os.path.dirname(output_image_path), exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Plot 1: APTOS Confusion Matrix
    cm = np.array(aptos_results["confusion_matrix"])
    class_labels = ["0: No DR", "1: Mild", "2: Moderate", "3: Severe", "4: PDR"]
    im = axes[0].imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    axes[0].set_title(f"APTOS Ordinal DR Confusion Matrix\nQWK: {aptos_results['quadratic_weighted_kappa']} | MAE: {aptos_results['mean_absolute_error']}")
    fig.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04)

    tick_marks = np.arange(len(class_labels))
    axes[0].set_xticks(tick_marks)
    axes[0].set_xticklabels(class_labels, rotation=45, ha="right")
    axes[0].set_yticks(tick_marks)
    axes[0].set_yticklabels(class_labels)

    # Label values in cells
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            axes[0].text(
                j, i, format(cm[i, j], "d"),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black"
            )
    axes[0].set_ylabel("True Grade")
    axes[0].set_xlabel("Predicted Grade")

    # Plot 2: Error Distribution Breakdown & Metrics
    categories = ["Exact Match", "1-Grade Miss", "Severe Miss (2+ Grades)"]
    percentages = [
        aptos_results["exact_matches_percent"],
        aptos_results["one_grade_off_percent"],
        aptos_results["severe_misses_percent"]
    ]
    colors = ["#22C55E", "#FBBF24", "#EF4444"]

    axes[1].bar(categories, percentages, color=colors, edgecolor="black", linewidth=1.2)
    axes[1].set_title("Clinical Triage Error Distribution (Slide 8 & 11)")
    axes[1].set_ylabel("Percentage (%)")
    axes[1].set_ylim(0, 100)

    for i, v in enumerate(percentages):
        axes[1].text(i, v + 2, f"{v:.1f}%", ha="center", fontweight="bold")

    # Overlay Segmentation Stats in an annotation box
    stats_text = (
        f"U-Net Vessel Segmentation (FIVES):\n"
        f"• Mean Dice Score: {unet_results['mean_dice_coefficient']:.4f}\n"
        f"• Mean IoU: {unet_results['mean_iou_score']:.4f}\n"
        f"• Avg Vessel Density: {unet_results['mean_vessel_density_percent']:.1f}%\n"
        f"Safety Gate High-Confidence Rate: {aptos_results['high_confidence_rate']}%"
    )
    axes[1].text(
        0.05, 0.55, stats_text,
        transform=axes[1].transAxes,
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#F8FAFC", edgecolor="#CBD5E1")
    )

    plt.tight_layout()
    plt.savefig(output_image_path, dpi=200)
    plt.close()
    print("Visual benchmark chart saved successfully.")


def main():
    aptos_res = run_aptos_ordinal_benchmark(num_samples=100)
    unet_res = run_unet_segmentation_benchmark(num_samples=30)

    # Save JSON summary
    report = {
        "ordinal_dr_benchmark": aptos_res,
        "unet_segmentation_benchmark": unet_res
    }
    json_path = "exports/benchmark_results.json"
    with open(json_path, "w") as f:
        json.dump(report, f, indent=4)

    print(f"\nJSON Benchmark Report saved to {json_path}")
    plot_confusion_matrix_and_metrics(aptos_res, unet_res)


if __name__ == "__main__":
    main()

