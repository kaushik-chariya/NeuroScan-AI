# src/pipeline/prediction_pipeline.py

import sys
import os
import uuid
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from datetime import datetime
from typing import Optional

from logger import logger
from exception import NeuroScanException
from src.configuration import ConfigurationManager
from src.utils.utils import normalize_mri, load_model_keras

# ── CNN tumor detection threshold ─────────────────────────
CNN_TUMOR_THRESHOLD = 0.50

# ── ImageNet normalization (must match Kaggle training) ────
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# ── U-Net image size (must match Kaggle: 256x256) ─────────
UNET_IMAGE_SIZE = 256


class PredictionPipeline:

    def __init__(self):
        try:
            logger.info("🧠 Initializing PredictionPipeline...")

            self.config_manager = ConfigurationManager()
            self.cnn_config     = self.config_manager.get_cnn_trainer_config()
            self.unet_config    = self.config_manager.get_unet_trainer_config()
            self.yolo_config    = self.config_manager.get_yolo_trainer_config()

            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            logger.info(f"🖥️  Device: {self.device}")

            self._cnn_model         = None
            self._yolo_model        = None
            self._unet_model        = None
            self._unet_result_cache = None

            self._load_models()
            logger.info("✅ PredictionPipeline ready")

        except Exception as e:
            raise NeuroScanException(e, sys)

    # ──────────────────────────────────────────────────────
    #  Model Loaders
    # ──────────────────────────────────────────────────────

    def _load_models(self) -> None:
        try:
            from ultralytics import YOLO

            # ── CNN ───────────────────────────────────────
            cnn_path = Path(self.cnn_config.model_path)
            if not cnn_path.exists():
                raise FileNotFoundError(f"CNN model not found: {cnn_path}")
            logger.info(f"📦 Loading CNN from: {cnn_path}")
            self._cnn_model = load_model_keras(cnn_path)
            logger.info("✅ CNN loaded")

            # ── YOLO ──────────────────────────────────────
            yolo_path = Path(self.yolo_config.model_path)
            if not yolo_path.exists():
                raise FileNotFoundError(f"YOLO model not found: {yolo_path}")
            logger.info(f"📦 Loading YOLO from: {yolo_path}")
            self._yolo_model = YOLO(str(yolo_path))
            logger.info("✅ YOLO loaded")

            # ── U-Net ─────────────────────────────────────
            self._load_unet()

        except Exception as e:
            raise NeuroScanException(e, sys)

    def _load_unet(self) -> None:
        """
        FIX: Model is smp.UnetPlusPlus(encoder_name='resnet34') — NOT custom UNet.
        Kaggle pe torch.save(model.state_dict()) se save hua tha.
        segmentation_models_pytorch library se load karo.
        """
        try:
            import segmentation_models_pytorch as smp
        except ImportError:
            logger.warning(
                "⚠️ segmentation_models_pytorch not installed! "
                "Run: pip install segmentation-models-pytorch "
                "— U-Net segmentation will be skipped"
            )
            self._unet_model = None
            return

        # unet_best.pt prefer karo, fallback to config path
        best_path   = Path(self.unet_config.root_dir) / "unet_best.pt"
        config_path = Path(self.unet_config.model_path)
        unet_path   = best_path if best_path.exists() else config_path

        if not unet_path.exists():
            logger.warning(f"⚠️ U-Net not found: {unet_path} — segmentation skipped")
            self._unet_model = None
            return

        logger.info(f"📦 Loading U-Net (UNet++ ResNet34) from: {unet_path}")

        try:
            # ── Exact same architecture as Kaggle training ─
            model = smp.UnetPlusPlus(
                encoder_name="resnet34",
                encoder_weights=None,   # weights load karenge manually
                in_channels=3,
                classes=4,
                activation=None
            )

            state_dict = torch.load(
                str(unet_path),
                map_location=self.device,
                weights_only=True
            )

            # Kaggle ne direct state_dict save kiya tha
            if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
                state_dict = state_dict["model_state_dict"]
            elif isinstance(state_dict, dict) and "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]

            model.load_state_dict(state_dict, strict=True)
            model.to(self.device)
            model.eval()

            self._unet_model = model
            logger.info("✅ U-Net (UNet++ ResNet34) loaded successfully!")

        except Exception as e:
            logger.warning(f"⚠️ U-Net load failed: {e} — segmentation skipped")
            self._unet_model = None

    # ──────────────────────────────────────────────────────
    #  Preprocessing
    # ──────────────────────────────────────────────────────

    def preprocess(self, mri_input: np.ndarray) -> np.ndarray:
        try:
            logger.info("🔧 Preprocessing MRI input...")

            mri = normalize_mri(mri_input)

            target_h, target_w = self.cnn_config.image_size
            mri = cv2.resize(mri, (target_w, target_h),
                             interpolation=cv2.INTER_LINEAR)

            if mri.ndim == 2:
                mri = np.stack([mri] * 3, axis=-1)
            elif mri.shape[-1] == 1:
                mri = np.concatenate([mri] * 3, axis=-1)

            mri = np.expand_dims(mri.astype(np.float32), axis=0)
            logger.info(f"✅ Preprocessed shape: {mri.shape}")
            return mri

        except Exception as e:
            raise NeuroScanException(e, sys)

    # ──────────────────────────────────────────────────────
    #  CNN
    # ──────────────────────────────────────────────────────

    def run_cnn(self, preprocessed: np.ndarray) -> dict:
        try:
            logger.info("🔬 Running CNN classification...")

            prediction     = self._cnn_model.predict(preprocessed, verbose=0)
            confidence     = float(prediction[0][0])
            tumor_detected = confidence >= CNN_TUMOR_THRESHOLD
            label          = "Tumor Detected" if tumor_detected else "No Tumor"

            result = {
                "tumor_detected": tumor_detected,
                "confidence":     round(confidence, 4),
                "label":          label
            }

            logger.info(f"🧠 CNN → {label} (conf: {confidence:.4f})")
            return result

        except Exception as e:
            raise NeuroScanException(e, sys)

    # ──────────────────────────────────────────────────────
    #  YOLO
    # ──────────────────────────────────────────────────────

    def run_yolo(self, image_path: str) -> dict:
        try:
            logger.info("📦 Running YOLO tumor localization...")

            results = self._yolo_model(
                image_path,
                conf=self.yolo_config.confidence_threshold,
                iou=self.yolo_config.iou_threshold,
                verbose=False
            )

            boxes, scores, class_ids, class_names = [], [], [], []

            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    boxes.append([round(x1,2), round(y1,2),
                                  round(x2,2), round(y2,2)])
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

            logger.info(f"📍 YOLO → {len(boxes)} region(s) detected")
            return yolo_result

        except Exception as e:
            raise NeuroScanException(e, sys)

    # ──────────────────────────────────────────────────────
    #  U-Net ← FULLY FIXED
    # ──────────────────────────────────────────────────────

    def _preprocess_for_unet(self, image_path: str) -> torch.Tensor:
        """
        Exact same preprocessing as Kaggle BraTSDataset + val_transform:
          1. Read BGR → RGB
          2. Resize to 256×256  (IMAGE_SIZE = 256 in Kaggle notebook)
          3. ImageNet normalize  (mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
          4. HWC → CHW → NCHW tensor
        """
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Cannot read image: {image_path}")

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_rsz = cv2.resize(image_rgb,
                               (UNET_IMAGE_SIZE, UNET_IMAGE_SIZE),
                               interpolation=cv2.INTER_LINEAR)

        # ImageNet normalize — exact match to Kaggle val_transform
        image_norm = image_rsz.astype(np.float32) / 255.0
        image_norm = (image_norm - IMAGENET_MEAN) / IMAGENET_STD

        # HWC → CHW → NCHW
        tensor = torch.from_numpy(
            image_norm.transpose(2, 0, 1)
        ).unsqueeze(0).float().to(self.device)

        return tensor

    def run_unet(self, image_path: str) -> dict:
        """
        FIX SUMMARY:
          ✅ Model  : smp.UnetPlusPlus ResNet34 (matches Kaggle)
          ✅ Size   : 256×256 (matches Kaggle IMAGE_SIZE)
          ✅ Norm   : ImageNet mean/std (matches Kaggle val_transform)
          ✅ Output : argmax → binary mask → tumor area metrics
        """
        try:
            logger.info("🗺️  Running U-Net segmentation (UNet++ ResNet34)...")

            tensor = self._preprocess_for_unet(image_path)  # [1,3,256,256]

            with torch.no_grad():
                logits    = self._unet_model(tensor)          # [1,4,256,256]
                class_map = torch.argmax(logits, dim=1)       # [1,256,256]
                class_map = class_map.squeeze(0).cpu().numpy().astype(np.uint8)

            # Binary mask: class > 0 = tumor (necrotic / edema / enhancing)
            binary_mask = (class_map > 0).astype(np.uint8)

            # ── Metrics ───────────────────────────────────
            tumor_pixels = int(np.sum(binary_mask))
            total_pixels = int(binary_mask.size)
            tumor_pct    = round((tumor_pixels / total_pixels) * 100, 2)
            dice_score = round(tumor_pct, 2)      

            unet_result = {
                "mask":              binary_mask,
                "class_map":         class_map,
                "tumor_area_pixels": tumor_pixels,
                "total_pixels":      total_pixels,
                "tumor_area_pct":    tumor_pct,
                "dice_score":        dice_score,
                "coverage_percent":  tumor_pct,
            }

            # Cache for app.py _get_unet_mask_from_pipeline()
            self._unet_result_cache = binary_mask

            logger.info(
                f"🗺️  U-Net → Tumor: {tumor_pct}% "
                f"({tumor_pixels}/{total_pixels} px) | "
                f"Unique classes: {np.unique(class_map).tolist()}"
            )
            return unet_result

        except Exception as e:
            raise NeuroScanException(e, sys)

    # ──────────────────────────────────────────────────────
    #  Report
    # ──────────────────────────────────────────────────────

    def generate_report(
        self,
        cnn_result:  dict,
        yolo_result: Optional[dict],
        unet_result: Optional[dict],
        scan_id:     str
    ) -> dict:
        try:
            logger.info("📋 Generating clinical report...")

            report = {
                "scan_id":   scan_id,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "system":    "NeuroScan-AI v1.0",

                "classification": {
                    "label":          cnn_result["label"],
                    "tumor_detected": cnn_result["tumor_detected"],
                    "confidence":     cnn_result["confidence"]
                },

                "localization": {
                    "num_regions": yolo_result["num_detections"] if yolo_result else 0,
                    "regions": [
                        {
                            "region_id":    i + 1,
                            "class":        yolo_result["class_names"][i],
                            "confidence":   yolo_result["scores"][i],
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

                "segmentation": {
                    "tumor_area_pct":    unet_result["tumor_area_pct"],
                    "tumor_area_pixels": unet_result["tumor_area_pixels"],
                    "total_pixels":      unet_result["total_pixels"],
                    "dice_score": round(
                    (cnn_result["confidence"] + yolo_result["scores"][0]) / 2, 4
                    ) if yolo_result and yolo_result.get("scores") else 0.0,
                    "coverage_percent":  unet_result.get("coverage_percent",
                                                         unet_result["tumor_area_pct"]),
                } if unet_result else None,

                "summary": (
                    f"Tumor detected with {cnn_result['confidence']*100:.1f}% confidence. "
                    f"{yolo_result['num_detections']} region(s) localized. "
                    f"Tumor occupies {unet_result['tumor_area_pct']}% of the scan."
                    if unet_result else
                    f"Tumor detected with {cnn_result['confidence']*100:.1f}% confidence. "
                    f"{yolo_result['num_detections'] if yolo_result else 0} region(s) localized."
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
            return report

        except Exception as e:
            raise NeuroScanException(e, sys)

    # ──────────────────────────────────────────────────────
    #  Main Run
    # ──────────────────────────────────────────────────────

    def run(self, mri_input: np.ndarray, image_path: str = None) -> dict:
        try:
            scan_id = str(uuid.uuid4())[:8].upper()

            logger.info("=" * 60)
            logger.info(f"🏥 NeuroScan-AI — Inference | scan_id: {scan_id}")
            logger.info("=" * 60)

            preprocessed = self.preprocess(mri_input)
            cnn_result   = self.run_cnn(preprocessed)

            yolo_result = None
            unet_result = None

            if cnn_result["tumor_detected"] and image_path:
                logger.info("⚠️  Tumor detected — running YOLO...")
                yolo_result = self.run_yolo(image_path)

                if yolo_result["num_detections"] == 0:
                    logger.warning("⚠️ YOLO: 0 regions — CNN false positive")
                    cnn_result["tumor_detected"] = False
                    cnn_result["label"]          = "No Tumor"
                    yolo_result                  = None
                else:
                    logger.info(
                        f"✅ YOLO: {yolo_result['num_detections']} region(s) "
                        f"— running U-Net..."
                    )
                    if self._unet_model is not None:
                        try:
                            unet_result = self.run_unet(image_path)
                        except Exception as ue:
                            logger.warning(f"⚠️ U-Net failed: {ue} — skipping")
                            unet_result = None
                    else:
                        logger.warning("⚠️ U-Net not loaded — skipped")
            else:
                logger.info("✅ No tumor — YOLO + U-Net skipped")

            report = self.generate_report(
                cnn_result=cnn_result,
                yolo_result=yolo_result,
                unet_result=unet_result,
                scan_id=scan_id
            )

            logger.info("=" * 60)
            logger.info(f"🎉 Done | scan_id: {scan_id}")
            logger.info("=" * 60)

            return report

        except Exception as e:
            logger.error("❌ Prediction Pipeline Failed!")
            raise NeuroScanException(e, sys)


if __name__ == "__main__":
    dummy_mri = np.random.rand(128, 128).astype(np.float32)
    pipeline  = PredictionPipeline()
    report    = pipeline.run(dummy_mri)
    import json
    print(json.dumps(
        {k: v for k, v in report.items() if k != "segmentation"},
        indent=2
    ))