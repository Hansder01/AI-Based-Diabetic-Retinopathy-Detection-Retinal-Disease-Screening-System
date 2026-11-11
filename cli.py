"""
Batch Retinal Screening CLI Tool for AI Diabetic Retinopathy Detection.

Usage:
    python cli.py --input Fundus_Images/DRD Dataset --max_samples 10 --export_visuals
    python cli.py --image APTOS_2019/val_images/val_images/000c1434d8d7.png
"""

import os
import sys
import argparse
from pathlib import Path

# Ensure workspace is on sys.path and KERAS_HOME points to workspace cache
workspace_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(workspace_dir))
workspace_keras = workspace_dir / ".keras"
if workspace_keras.exists():
    os.environ["KERAS_HOME"] = str(workspace_keras)

import cv2
import pandas as pd
from tqdm import tqdm

from src.pipeline.hierarchical_triage import HierarchicalTriageEngine


def run_batch_screening(
    input_path: str,
    output_dir: str = "exports/cli_results",
    confidence_threshold: float = 0.70,
    max_samples: int = 50,
    export_visuals: bool = True
):
    os.makedirs(output_dir, exist_ok=True)
    if export_visuals:
        vis_dir = os.path.join(output_dir, "visualizations")
        os.makedirs(vis_dir, exist_ok=True)

    print(f"\n{'='*70}")
    print("  AI DIABETIC RETINOPATHY & RETINAL DISEASE CLINICAL SCREENING CLI")
    print(f"{'='*70}")

    engine = HierarchicalTriageEngine(
        ordinal_model_path="checkpoints/aptos_ordinal_best.keras",
        fives_model_path="checkpoints/fives_classifier_best.keras",
        unet_model_path="checkpoints/unet_vessel_best.keras",
        confidence_threshold=confidence_threshold
    )

    # Gather target images
    if os.path.isfile(input_path):
        image_files = [input_path]
    elif os.path.isdir(input_path):
        valid_exts = (".png", ".jpg", ".jpeg", ".bmp")
        image_files = [
            os.path.join(input_path, f) for f in sorted(os.listdir(input_path))
            if f.lower().endswith(valid_exts)
        ]
        if max_samples:
            image_files = image_files[:max_samples]
    else:
        raise FileNotFoundError(f"Invalid input path: {input_path}")

    print(f"\nProcessing {len(image_files)} scans through hierarchical triage...")

    records = []
    for img_path in image_files:
        fname = os.path.basename(img_path)
        try:
            report = engine.diagnose(img_path, run_segmentation=True, run_explainability=True)
            summary = report["summary"]
            dr = report["dr_ordinal_grading"]
            vasc = report["vasculature"]
            vis = report["visual_artifacts"]

            rec = {
                "Filename": fname,
                "Primary_Diagnosis": summary["primary_diagnosis"],
                "DR_Grade": dr["grade"],
                "Severity_Stage": dr["stage_name"],
                "Continuous_Progression_Index": dr["continuous_score"],
                "Confidence_Percent": summary["confidence_percent"],
                "Triage_Badge": summary["triage_badge"],
                "Manual_Review_Mandated": summary["requires_manual_review"],
                "Vessel_Density_Percent": vasc["vessel_density_percent"],
                "Decision_Pathway": summary["pathway_taken"],
                "Clinical_Recommendation": summary["clinical_recommendation"]
            }

            if export_visuals and vis.get("gradcam_overlay") is not None:
                stem = Path(fname).stem
                grad_path = os.path.join(vis_dir, f"{stem}_gradcam.png")
                vessel_path = os.path.join(vis_dir, f"{stem}_vessels.png")
                cv2.imwrite(grad_path, cv2.cvtColor(vis["gradcam_overlay"], cv2.COLOR_RGB2BGR))
                cv2.imwrite(vessel_path, cv2.cvtColor(vis["vessel_overlay"], cv2.COLOR_RGB2BGR))
                rec["GradCAM_Artifact"] = grad_path
                rec["Vessel_Artifact"] = vessel_path

            records.append(rec)

        except Exception as e:
            print(f"Error processing {fname}: {e}")

    df = pd.DataFrame(records)
    csv_out = os.path.join(output_dir, "screening_report.csv")
    df.to_csv(csv_out, index=False)

    print(f"\nBatch Screening Completed!")
    print(f"Summary Report saved to: {csv_out}")

    # Summary Statistics
    total = len(df)
    high_conf = len(df[~df["Manual_Review_Mandated"]])
    manual_review = len(df[df["Manual_Review_Mandated"]])

    print(f"\n--- Clinical Triage Summary (Total: {total} Scans) ---")
    print(f"Verified High Confidence: {high_conf} ({high_conf/total*100:.1f}%)")
    print(f"Manual Review Mandated:   {manual_review} ({manual_review/total*100:.1f}%)")
    if "Severity_Stage" in df:
        print("\nDR Severity Distribution:")
        print(df["Severity_Stage"].value_counts().to_string())


def main():
    parser = argparse.ArgumentParser(description="Clinical Fundus Screening CLI")
    parser.add_argument("--input", type=str, default="Fundus_Images/DRD Dataset", help="Path to image or image directory")
    parser.add_argument("--image", type=str, default=None, help="Path to single image")
    parser.add_argument("--output_dir", type=str, default="exports/cli_results")
    parser.add_argument("--confidence_threshold", type=float, default=0.70)
    parser.add_argument("--max_samples", type=int, default=10)
    parser.add_argument("--no_visuals", action="store_true")
    args = parser.parse_args()

    target = args.image if args.image else args.input
    run_batch_screening(
        input_path=target,
        output_dir=args.output_dir,
        confidence_threshold=args.confidence_threshold,
        max_samples=args.max_samples,
        export_visuals=not args.no_visuals
    )


if __name__ == "__main__":
    main()

