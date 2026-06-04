# src/pipeline/prediction_pipeline.py

"""
Prediction Pipeline — NeuroScan-AI
------------------------------------
Hospital-grade inference pipeline for Brain Tumor Detection.

Inference Flow:
    MRI Input (.nii.gz or .png slice)
        → Preprocessing
        → CNN  → No Tumor  → Report (Normal)
        → CNN  → Tumor     → YOLO  → Bounding Box
                           → UNet  → Segmentation Mask
                           → Final Report (Tumor Details)

Output:
    - Tumor presence (Yes/No)
    - Confidence score
    - Bounding box coordinates (if tumor)
    - Segmentation mask (if tumor)
    - Tumor area percentage
    - Full structured report (dict)

IMPORTANT:
    - No patient identifiable data is logged
    - All predictions are timestamped for audit trail
    - Models are loaded once and cached for performance
"""

import sys
import os
import uuid
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Optional

from logger import logger
from exception import NeuroScanException
from src.configuration import ConfigurationManager
from src.utils.utils import normalize_mri, load_model


class PredictionPipeline:
    """
    Hospital-grade inference pipeline for NeuroScan-AI.

    Loads CNN, YOLO, and UNet models once on initialization
    and caches them for efficient repeated inference.

    Usage:
        pipeline = PredictionPipeline()
        report   = pipeline.run(mri_input)
    """

    def __init__(self):
        """
        Initialize PredictionPipeline.
        Loads all three models into memory on startup.

        Raises:
            NeuroScanException: If config loading or model loading fails.
        """
        try:
            logger.info("🧠 Initializing PredictionPipeline...")

            self.config_manager = ConfigurationManager()
            self.cnn_config     = self.config_manager.get_cnn_trainer_config()
            self.unet_config    = self.config_manager.get_unet_trainer_config()
            self.yolo_config    = self.config_manager.get_yolo_trainer_config()

            # Model cache
            self._cnn_model  = None
            self._unet_model = None
            self._yolo_model = None

            # Load models
            self._load_models()

            logger.info("✅ PredictionPipeline ready")

        except Exception as e:
            raise NeuroScanException(e, sys)

    # ── Model Loading ─────────────────────────────────────────────────────────

    def _load_models(self) -> None:
        """
        Loads CNN, UNet (TensorFlow .h5) and YOLO (PyTorch .pt) models.
        Models are cached as instance attributes.

        Raises:
            NeuroScanException: If any model file is missing or corrupted.
        """
        try:
            # ── CNN ──────────────────────────────────────────
            cnn_path = self.cnn_config.model_path
            if not cnn_path.exists():
                raise FileNotFoundError(
                    f"CNN model not found at: {cnn_path}\n"
                    f"Please download cnn_model.h5 from Kaggle and place in artifacts/models/"
                )
            logger.info(f"📦 Loading CNN model from: {cnn_path}")
            self._cnn_model = load_model(cnn_path, model_type="keras")
            logger.info("✅ CNN model loaded")

            # ── UNet ─────────────────────────────────────────
            unet_path = self.unet_config.model_path
            if not unet_path.exists():
                raise FileNotFoundError(
                    f"UNet model not found at: {unet_path}\n"
                    f"Please download unet_model.h5 from Kaggle and place in artifacts/models/"
                )
            logger.info(f"📦 Loading UNet model from: {unet_path}")
            self._unet_model = load_model(unet_path, model_type="keras")
            logger.info("✅ UNet model loaded")

            # ── YOLO ─────────────────────────────────────────
            yolo_path = self.yolo_config.model_path
            if not yolo_path.exists():
                raise FileNotFoundError(
                    f"YOLO model not found at: {yolo_path}\n"
                    f"Please download yolo_model.pt from Kaggle and place in artifacts/models/"
                )
            logger.info(f"📦 Loading YOLO model from: {yolo_path}")
            self._yolo_model = load_model(yolo_path, model_type="yolo")
            logger.info("✅ YOLO model loaded")

        except Exception as e:
            raise NeuroScanException(e, sys)

    # ── Preprocessing ─────────────────────────────────────────────────────────

    def preprocess(self, mri_input: np.ndarray) -> np.ndarray:
        """
        Preprocesses raw MRI slice for model inference.

        Steps:
            1. Normalize to [0, 1]
            2. Resize to model input size (128x128)
            3. Expand dims for batch dimension

        Args:
            mri_input (np.ndarray): Raw MRI slice (H, W) or (H, W, C).

        Returns:
            np.ndarray: Preprocessed array of shape (1, H, W, C).

        Raises:
            NeuroScanException: If preprocessing fails.
        """
        try:
            import cv2

            logger.info("🔧 Preprocessing MRI input...")

            # Normalize
            mri = normalize_mri(mri_input)

            # Resize
            target_size = self.cnn_config.image_size  # (128, 128)
            mri = cv2.resize(mri, target_size, interpolation=cv2.INTER_LINEAR)

            # Ensure 3 channels (H, W) → (H, W, 3)
            if mri.ndim == 2:
                mri = np.stack([mri] * 3, axis=-1)
            elif mri.shape[-1] == 1:
                mri = np.concatenate([mri] * 3, axis=-1)

            # Add batch dim → (1, H, W, 3)
            mri = np.expand_dims(mri, axis=0)

            logger.info(f"✅ Preprocessed shape: {mri.shape}")
            return mri

        except Exception as e:
            raise NeuroScanException(e, sys)

    # ── CNN Inference ─────────────────────────────────────────────────────────

    def run_cnn(self, preprocessed: np.ndarray) -> dict:
        """
        Runs CNN binary classification on preprocessed MRI.

        Args:
            preprocessed (np.ndarray): Preprocessed MRI (1, H, W, C).

        Returns:
            dict: {
                "tumor_detected": bool,
                "confidence":     float (0.0 - 1.0),
                "label":          str ("Tumor" | "No Tumor")
            }

        Raises:
            NeuroScanException: If CNN inference fails.
        """
        try:
            logger.info("🔬 Running CNN classification...")

            prediction = self._cnn_model.predict(preprocessed, verbose=0)
            confidence = float(prediction[0][0])

            tumor_detected = confidence >= self.yolo_config.confidence_threshold
            label = "Tumor Detected" if tumor_detected else "No Tumor"

            result = {
                "tumor_detected": tumor_detected,
                "confidence":     round(confidence, 4),
                "label":          label
            }

            logger.info(f"🧠 CNN Result → {label} (confidence: {confidence:.4f})")
            return result

        except Exception as e:
            raise NeuroScanException(e, sys)

    # ── YOLO Inference ────────────────────────────────────────────────────────

    def run_yolo(self, mri_input: np.ndarray) -> dict:
        """
        Runs YOLO object detection to localize tumor with bounding box.

        Args:
            mri_input (np.ndarray): Original or preprocessed MRI slice.

        Returns:
            dict: {
                "boxes":       list of [x1, y1, x2, y2],
                "scores":      list of float confidence scores,
                "class_ids":   list of int class ids,
                "class_names": list of str class names,
                "num_detections": int
            }

        Raises:
            NeuroScanException: If YOLO inference fails.
        """
        try:
            logger.info("📦 Running YOLO tumor localization...")

            results = self._yolo_model(
                mri_input,
                conf=self.yolo_config.confidence_threshold,
                iou=self.yolo_config.iou_threshold,
                verbose=False
            )

            boxes      = []
            scores     = []
            class_ids  = []
            class_names = []

            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    boxes.append([round(x1, 2), round(y1, 2),
                                  round(x2, 2), round(y2, 2)])
                    scores.append(round(float(box.conf[0]), 4))
                    cid = int(box.cls[0])
                    class_ids.append(cid)
                    class_names.append(
                        self.yolo_config.class_names[cid]
                        if cid < len(self.yolo_config.class_names)
                        else "Unknown"
                    )

            yolo_result = {
                "boxes":          boxes,
                "scores":         scores,
                "class_ids":      class_ids,
                "class_names":    class_names,
                "num_detections": len(boxes)
            }

            logger.info(f"📍 YOLO → {len(boxes)} tumor region(s) detected")
            for i, (box, score, name) in enumerate(
                zip(boxes, scores, class_names)
            ):
                logger.info(f"   Detection {i+1}: {name} | score: {score} | box: {box}")

            return yolo_result

        except Exception as e:
            raise NeuroScanException(e, sys)

    # ── UNet Inference ────────────────────────────────────────────────────────

    def run_unet(self, preprocessed: np.ndarray) -> dict:
        """
        Runs UNet segmentation to generate pixel-level tumor mask.

        Args:
            preprocessed (np.ndarray): Preprocessed MRI (1, H, W, C).

        Returns:
            dict: {
                "mask":              np.ndarray (H, W) binary mask,
                "tumor_area_pixels": int,
                "total_pixels":      int,
                "tumor_area_pct":    float (percentage of image)
            }

        Raises:
            NeuroScanException: If UNet inference fails.
        """
        try:
            logger.info("🗺️  Running UNet segmentation...")

            mask_pred = self._unet_model.predict(preprocessed, verbose=0)

            # Shape: (1, H, W, num_classes) → (H, W)
            if mask_pred.shape[-1] > 1:
                mask = np.argmax(mask_pred[0], axis=-1)
            else:
                mask = (mask_pred[0, :, :, 0] > 0.5).astype(np.uint8)

            tumor_pixels = int(np.sum(mask > 0))
            total_pixels = int(mask.size)
            tumor_pct    = round((tumor_pixels / total_pixels) * 100, 2)

            unet_result = {
                "mask":              mask,
                "tumor_area_pixels": tumor_pixels,
                "total_pixels":      total_pixels,
                "tumor_area_pct":    tumor_pct
            }

            logger.info(f"🗺️  UNet → Tumor area: {tumor_pct}% of image")
            return unet_result

        except Exception as e:
            raise NeuroScanException(e, sys)

    # ── Report Generation ─────────────────────────────────────────────────────

    def generate_report(
        self,
        cnn_result:  dict,
        yolo_result: Optional[dict],
        unet_result: Optional[dict],
        scan_id:     str
    ) -> dict:
        """
        Generates a structured clinical report from model outputs.

        Args:
            cnn_result  (dict): CNN classification result.
            yolo_result (dict | None): YOLO detection result (None if no tumor).
            unet_result (dict | None): UNet segmentation result (None if no tumor).
            scan_id     (str): Unique scan identifier for audit trail.

        Returns:
            dict: Full structured clinical report.

        Raises:
            NeuroScanException: If report generation fails.
        """
        try:
            logger.info("📋 Generating clinical report...")

            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            report = {
                "scan_id":   scan_id,
                "timestamp": timestamp,
                "system":    "NeuroScan-AI v1.0",

                # CNN Results
                "classification": {
                    "label":          cnn_result["label"],
                    "tumor_detected": cnn_result["tumor_detected"],
                    "confidence":     cnn_result["confidence"]
                },

                # YOLO Results
                "localization": {
                    "num_regions":  yolo_result["num_detections"] if yolo_result else 0,
                    "regions":      [
                        {
                            "region_id":   i + 1,
                            "class":       yolo_result["class_names"][i],
                            "confidence":  yolo_result["scores"][i],
                            "bounding_box": {
                                "x1": yolo_result["boxes"][i][0],
                                "y1": yolo_result["boxes"][i][1],
                                "x2": yolo_result["boxes"][i][2],
                                "y2": yolo_result["boxes"][i][3]
                            }
                        }
                        for i in range(yolo_result["num_detections"])
                    ] if yolo_result else []
                },

                # UNet Results
                "segmentation": {
                    "tumor_area_pct":    unet_result["tumor_area_pct"]    if unet_result else 0.0,
                    "tumor_area_pixels": unet_result["tumor_area_pixels"] if unet_result else 0,
                    "total_pixels":      unet_result["total_pixels"]      if unet_result else 0,
                } if unet_result else None,

                # Clinical Summary
                "summary": (
                    f"Tumor detected with {cnn_result['confidence']*100:.1f}% confidence. "
                    f"{yolo_result['num_detections']} region(s) localized. "
                    f"Tumor occupies {unet_result['tumor_area_pct']}% of the scan."
                ) if cnn_result["tumor_detected"] else (
                    f"No tumor detected. "
                    f"Classification confidence: {cnn_result['confidence']*100:.1f}%."
                ),

                "disclaimer": (
                    "This report is AI-generated and intended to assist medical professionals. "
                    "It does not replace clinical diagnosis by a qualified radiologist."
                )
            }

            logger.info(f"✅ Report generated | scan_id: {scan_id}")
            logger.info(f"   Summary: {report['summary']}")
            return report

        except Exception as e:
            raise NeuroScanException(e, sys)

    # ── Main Inference Run ────────────────────────────────────────────────────

    def run(self, mri_input: np.ndarray) -> dict:
        """
        Full inference pipeline on a single MRI slice.

        Flow:
            MRI Input
                → Preprocess
                → CNN  → No Tumor → Report (Normal)
                → CNN  → Tumor    → YOLO → UNet → Report (Tumor)

        Args:
            mri_input (np.ndarray): Raw MRI slice (H, W) or (H, W, C).

        Returns:
            dict: Full structured clinical report.

        Raises:
            NeuroScanException: If inference fails at any stage.
        """
        try:
            scan_id = str(uuid.uuid4())[:8].upper()

            logger.info("=" * 60)
            logger.info(f"🏥 NeuroScan-AI — Inference Started | scan_id: {scan_id}")
            logger.info("=" * 60)

            # Step 1: Preprocess
            preprocessed = self.preprocess(mri_input)

            # Step 2: CNN Classification
            cnn_result = self.run_cnn(preprocessed)

            yolo_result = None
            unet_result = None

            # Step 3: If tumor detected → YOLO + UNet
            if cnn_result["tumor_detected"]:
                logger.info("⚠️  Tumor detected — running localization and segmentation...")

                # Step 3a: YOLO Localization
                yolo_result = self.run_yolo(mri_input)

                # Step 3b: UNet Segmentation
                unet_result = self.run_unet(preprocessed)

            else:
                logger.info("✅ No tumor detected — skipping YOLO and UNet")

            # Step 4: Generate Report
            report = self.generate_report(
                cnn_result=cnn_result,
                yolo_result=yolo_result,
                unet_result=unet_result,
                scan_id=scan_id
            )

            logger.info("=" * 60)
            logger.info(f"🎉 Inference Complete | scan_id: {scan_id}")
            logger.info("=" * 60)

            return report

        except Exception as e:
            logger.error("❌ Prediction Pipeline Failed!")
            raise NeuroScanException(e, sys)


# ── Entry Point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import numpy as np

    # Simulate a dummy MRI slice for testing
    dummy_mri = np.random.rand(128, 128).astype(np.float32)

    pipeline = PredictionPipeline()
    report   = pipeline.run(dummy_mri)

    print("\n📋 Final Report:")
    import json
    print(json.dumps(
        {k: v for k, v in report.items() if k != "segmentation"},
        indent=2
    ))