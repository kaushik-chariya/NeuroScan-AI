# demo.py

"""
Demo Script — NeuroScan-AI
----------------------------
Local testing script to verify the full prediction pipeline
without running the Flask app.

Usage:
    python demo.py --input <path_to_mri_slice.png>
    python demo.py --dummy   # runs with a random dummy MRI slice

NOTE:
    Models must be present in artifacts/models/ before running.
    Download from Kaggle after training.
"""

import sys
import json
import argparse
import numpy as np
from pathlib import Path

from logger import logger
from exception import NeuroScanException


def load_mri_image(image_path: str) -> np.ndarray:
    """
    Loads an MRI image from disk.

    Supports:
        - .png, .jpg  → loaded as grayscale numpy array
        - .npy        → loaded directly as numpy array
        - .nii.gz     → loaded via nibabel, middle slice extracted

    Args:
        image_path (str): Path to the MRI image file.

    Returns:
        np.ndarray: MRI slice as numpy array (H, W).

    Raises:
        NeuroScanException: If file not found or format unsupported.
    """
    try:
        path = Path(image_path)

        if not path.exists():
            raise FileNotFoundError(f"Input file not found: {image_path}")

        suffix = path.suffix.lower()

        if suffix in [".png", ".jpg", ".jpeg"]:
            import cv2
            img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise ValueError(f"Could not read image: {image_path}")
            logger.info(f"✅ Loaded image: {path.name} | shape: {img.shape}")
            return img.astype(np.float32)

        elif suffix == ".npy":
            img = np.load(str(path))
            logger.info(f"✅ Loaded .npy: {path.name} | shape: {img.shape}")
            return img.astype(np.float32)

        elif image_path.endswith(".nii.gz") or suffix == ".nii":
            import nibabel as nib
            nii = nib.load(str(path))
            data = nii.get_fdata()
            # Extract middle slice along depth axis
            mid = data.shape[2] // 2
            img = data[:, :, mid]
            logger.info(f"✅ Loaded NIfTI: {path.name} | slice: {mid} | shape: {img.shape}")
            return img.astype(np.float32)

        else:
            raise ValueError(
                f"Unsupported file format: {suffix}\n"
                f"Supported: .png, .jpg, .npy, .nii.gz"
            )

    except Exception as e:
        raise NeuroScanException(e, sys)


def print_report(report: dict) -> None:
    """
    Pretty prints the clinical report to console.

    Args:
        report (dict): Structured report from PredictionPipeline.
    """
    print("\n" + "=" * 60)
    print("🧠  NeuroScan-AI — Clinical Report")
    print("=" * 60)
    print(f"  Scan ID    : {report['scan_id']}")
    print(f"  Timestamp  : {report['timestamp']}")
    print(f"  System     : {report['system']}")
    print("-" * 60)

    cls = report["classification"]
    print(f"\n📊 Classification")
    print(f"  Result     : {cls['label']}")
    print(f"  Confidence : {cls['confidence'] * 100:.2f}%")

    loc = report["localization"]
    print(f"\n📍 Localization (YOLO)")
    if loc["num_regions"] == 0:
        print("  No tumor regions detected.")
    else:
        print(f"  Regions Found: {loc['num_regions']}")
        for r in loc["regions"]:
            bb = r["bounding_box"]
            print(f"  Region {r['region_id']}: {r['class']} | "
                  f"conf: {r['confidence']*100:.1f}% | "
                  f"box: [{bb['x1']}, {bb['y1']}, {bb['x2']}, {bb['y2']}]")

    seg = report.get("segmentation")
    print(f"\n🗺️  Segmentation (UNet)")
    if seg:
        print(f"  Tumor Area : {seg['tumor_area_pct']}% of scan")
        print(f"  Pixels     : {seg['tumor_area_pixels']} / {seg['total_pixels']}")
    else:
        print("  No segmentation performed.")

    print(f"\n📋 Summary")
    print(f"  {report['summary']}")

    print(f"\n⚠️  Disclaimer")
    print(f"  {report['disclaimer']}")
    print("=" * 60 + "\n")


def save_report_json(report: dict, output_path: str = "demo_report.json") -> None:
    """
    Saves the clinical report as a JSON file.

    Args:
        report      (dict): Structured report from PredictionPipeline.
        output_path (str):  Path to save the JSON file.
    """
    try:
        # Remove numpy mask from report before JSON serialization
        report_serializable = {
            k: v for k, v in report.items()
            if k != "segmentation" or v is None
        }
        if report.get("segmentation"):
            report_serializable["segmentation"] = {
                k: v for k, v in report["segmentation"].items()
                if not isinstance(v, np.ndarray)
            }

        with open(output_path, "w") as f:
            json.dump(report_serializable, f, indent=2)

        logger.info(f"✅ Report saved → {output_path}")

    except Exception as e:
        raise NeuroScanException(e, sys)


def main():
    """
    Entry point for demo script.

    Args (CLI):
        --input  : Path to MRI image (.png / .npy / .nii.gz)
        --dummy  : Use a random dummy MRI slice for testing
        --save   : Save report as JSON (default: demo_report.json)
    """
    try:
        parser = argparse.ArgumentParser(
            description="NeuroScan-AI — Demo Prediction Script"
        )
        parser.add_argument(
            "--input", type=str, default=None,
            help="Path to MRI image (.png / .npy / .nii.gz)"
        )
        parser.add_argument(
            "--dummy", action="store_true",
            help="Run with a random dummy MRI slice (no real image needed)"
        )
        parser.add_argument(
            "--save", type=str, default="demo_report.json",
            help="Path to save the output report JSON"
        )
        args = parser.parse_args()

        # ── Validate Args ─────────────────────────────────
        if not args.dummy and args.input is None:
            logger.error("❌ Please provide --input <path> or use --dummy flag")
            parser.print_help()
            sys.exit(1)

        # ── Load MRI ──────────────────────────────────────
        if args.dummy:
            logger.info("🧪 Dummy mode — generating random MRI slice...")
            mri_input = np.random.rand(128, 128).astype(np.float32)
            logger.info(f"✅ Dummy MRI shape: {mri_input.shape}")
        else:
            mri_input = load_mri_image(args.input)

        # ── Run Prediction Pipeline ───────────────────────
        logger.info("🚀 Starting Prediction Pipeline...")
        from src.pipeline.prediction_pipeline import PredictionPipeline

        pipeline = PredictionPipeline()
        report   = pipeline.run(mri_input)

        # ── Print Report ──────────────────────────────────
        print_report(report)

        # ── Save Report ───────────────────────────────────
        save_report_json(report, args.save)

    except Exception as e:
        raise NeuroScanException(e, sys)


# ── Entry Point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    main()