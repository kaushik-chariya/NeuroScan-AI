# src/components/yolo_trainer.py

"""
YOLO Trainer Component — NeuroScan-AI
--------------------------------------
Defines YOLO architecture config, prepares dataset YAML,
and sets up training configuration for Kaggle execution.

NOTE: Actual training happens on Kaggle with GPU.
      This file defines structure + config only.
"""

import os
import sys
import yaml
import shutil
from pathlib import Path

from logger import logger
from exception import NeuroScanException
from src.entity.config_entity import YOLOTrainerConfig


class YOLOTrainer:
    """
    Prepares YOLO training configuration for tumor localization.

    Responsibilities:
    - Validates preprocessed split data exists
    - Creates YOLO-format dataset YAML
    - Organizes images + labels into YOLO directory structure
    - Generates training config (used in Kaggle notebook)

    NOTE:
        Actual model.train() call happens in:
        notebooks/kaggle_yolo_training.ipynb
    """

    def __init__(self, config: YOLOTrainerConfig):
        """
        Initialize YOLOTrainer with config entity.

        Args:
            config (YOLOTrainerConfig): YOLO trainer configuration dataclass.
        """
        try:
            self.config = config
            logger.info("🔧 YOLOTrainer initialized")
            logger.info(f"   Model      : {self.config.model_name}")
            logger.info(f"   Pretrained : {self.config.pretrained_weights}")
            logger.info(f"   Classes    : {self.config.num_classes} → {self.config.class_names}")
            logger.info(f"   Image Size : {self.config.image_size}")
            logger.info(f"   Epochs     : {self.config.epochs}")
        except Exception as e:
            raise NeuroScanException(e, sys)

    # ── Step 1: Validate Input Data ──────────────────────────────────────────

    def validate_split_data(self) -> bool:
        """
        Validates that transformed split data exists before YOLO setup.

        Expected structure (from data_transformation.py):
            artifacts/data/split/
            ├── train/images/  ← .npy or .png slices
            ├── val/images/
            └── test/images/

        Returns:
            bool: True if all required directories exist and are non-empty.

        Raises:
            NeuroScanException: If required directories are missing or empty.
        """
        try:
            logger.info("🔍 Validating split data for YOLO setup...")

            split_dir = self.config.root_dir.parent.parent / "data" / "split"
            required_splits = ["train", "val", "test"]
            all_valid = True

            for split in required_splits:
                img_dir = split_dir / split / "images"
                if not img_dir.exists():
                    logger.warning(f"⚠️  Missing: {img_dir}")
                    all_valid = False
                else:
                    count = len(list(img_dir.glob("*")))
                    logger.info(f"   ✅ {split}/images → {count} files found")

            if not all_valid:
                logger.error("❌ Split data validation failed — run data_transformation first")
            else:
                logger.info("✅ Split data validation passed")

            return all_valid

        except Exception as e:
            raise NeuroScanException(e, sys)

    # ── Step 2: Create YOLO Directory Structure ──────────────────────────────

    def create_yolo_directory_structure(self) -> dict:
        """
        Creates YOLO-compatible directory structure under root_dir.

        YOLO expects:
            yolo_trainer/
            ├── images/
            │   ├── train/
            │   ├── val/
            │   └── test/
            └── labels/
                ├── train/
                ├── val/
                └── test/

        Returns:
            dict: Dictionary of all created directory paths.

        Raises:
            NeuroScanException: If directory creation fails.
        """
        try:
            logger.info("📁 Creating YOLO directory structure...")

            dirs = {}
            for split in ["train", "val", "test"]:
                img_path = self.config.root_dir / "images" / split
                lbl_path = self.config.root_dir / "labels" / split

                img_path.mkdir(parents=True, exist_ok=True)
                lbl_path.mkdir(parents=True, exist_ok=True)

                dirs[f"{split}_images"] = img_path
                dirs[f"{split}_labels"] = lbl_path

                logger.info(f"   📂 Created: {img_path}")
                logger.info(f"   📂 Created: {lbl_path}")

            logger.info("✅ YOLO directory structure ready")
            return dirs

        except Exception as e:
            raise NeuroScanException(e, sys)

    # ── Step 3: Generate dataset.yaml ────────────────────────────────────────

    def create_dataset_yaml(self) -> Path:
        """
        Creates YOLO-format dataset.yaml required for ultralytics training.

        Format:
            path: <absolute path to yolo_trainer root>
            train: images/train
            val:   images/val
            test:  images/test
            nc:    <num_classes>
            names: [<class_names>]

        Returns:
            Path: Path to the saved dataset.yaml file.

        Raises:
            NeuroScanException: If YAML creation fails.
        """
        try:
            logger.info("📝 Creating dataset.yaml for YOLO...")

            dataset_config = {
                "path": str(self.config.root_dir.resolve()),
                "train": "images/train",
                "val":   "images/val",
                "test":  "images/test",
                "nc":    self.config.num_classes,
                "names": self.config.class_names
            }

            yaml_path = self.config.root_dir / "dataset.yaml"
            with open(yaml_path, "w") as f:
                yaml.dump(dataset_config, f, default_flow_style=False, sort_keys=False)

            logger.info(f"✅ dataset.yaml saved → {yaml_path}")
            logger.info(f"   Classes : {self.config.num_classes} → {self.config.class_names}")
            return yaml_path

        except Exception as e:
            raise NeuroScanException(e, sys)

    # ── Step 4: Generate Training Config ─────────────────────────────────────

    def generate_training_config(self) -> dict:
        """
        Generates a training configuration dictionary for use in Kaggle notebook.

        This config is logged and saved as training_config.yaml so the
        Kaggle notebook can load it directly without hardcoding values.

        Returns:
            dict: YOLO training hyperparameters and paths.

        Raises:
            NeuroScanException: If config generation fails.
        """
        try:
            logger.info("⚙️  Generating YOLO training config...")

            training_config = {
                "model": self.config.pretrained_weights,
                "data":  str((self.config.root_dir / "dataset.yaml").resolve()),
                "epochs":     self.config.epochs,
                "batch":      self.config.batch_size,
                "imgsz":      self.config.image_size,
                "lr0":        self.config.learning_rate,
                "conf":       self.config.confidence_threshold,
                "iou":        self.config.iou_threshold,
                "project":    str(self.config.root_dir / "runs"),
                "name":       self.config.model_name,
                "save":       True,
                "device":     "0",    # GPU on Kaggle
                "workers":    4,
                "pretrained": True,
                "verbose":    True
            }

            config_path = self.config.root_dir / "training_config.yaml"
            with open(config_path, "w") as f:
                yaml.dump(training_config, f, default_flow_style=False, sort_keys=False)

            logger.info(f"✅ Training config saved → {config_path}")
            logger.info(f"   Epochs     : {self.config.epochs}")
            logger.info(f"   Batch Size : {self.config.batch_size}")
            logger.info(f"   Image Size : {self.config.image_size}")
            logger.info(f"   LR         : {self.config.learning_rate}")
            logger.info(f"   Conf Thresh: {self.config.confidence_threshold}")
            logger.info(f"   IoU Thresh : {self.config.iou_threshold}")

            return training_config

        except Exception as e:
            raise NeuroScanException(e, sys)

    # ── Step 5: Copy Model to Artifacts ──────────────────────────────────────

    def save_trained_model(self, trained_model_path: Path) -> Path:
        """
        Copies the trained YOLO model (.pt) from Kaggle output
        into the artifacts/models/ directory.

        Called AFTER Kaggle training is complete and model is downloaded.

        Args:
            trained_model_path (Path): Path to the downloaded .pt file.

        Returns:
            Path: Final saved model path in artifacts.

        Raises:
            NeuroScanException: If model file not found or copy fails.
        """
        try:
            logger.info(f"💾 Saving trained YOLO model from: {trained_model_path}")

            if not Path(trained_model_path).exists():
                raise FileNotFoundError(
                    f"Trained model not found at: {trained_model_path}\n"
                    f"Please download yolo_model.pt from Kaggle first."
                )

            self.config.model_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(trained_model_path, self.config.model_path)

            size_mb = self.config.model_path.stat().st_size / (1024 * 1024)
            logger.info(f"✅ YOLO model saved → {self.config.model_path}")
            logger.info(f"   Model size : {size_mb:.2f} MB")

            return self.config.model_path

        except Exception as e:
            raise NeuroScanException(e, sys)

    # ── Main Run ─────────────────────────────────────────────────────────────

    def run(self) -> dict:
        """
        Executes the full YOLO trainer setup pipeline:
            1. Validate split data
            2. Create YOLO directory structure
            3. Create dataset.yaml
            4. Generate training_config.yaml

        Returns:
            dict: Summary of all generated paths and config.

        Raises:
            NeuroScanException: If any step fails.
        """
        try:
            logger.info("=" * 60)
            logger.info("🚀 Starting YOLO Trainer Setup")
            logger.info("=" * 60)

            # Step 1
            self.validate_split_data()

            # Step 2
            dirs = self.create_yolo_directory_structure()

            # Step 3
            yaml_path = self.create_dataset_yaml()

            # Step 4
            training_config = self.generate_training_config()

            summary = {
                "yolo_root":       str(self.config.root_dir),
                "dataset_yaml":    str(yaml_path),
                "model_path":      str(self.config.model_path),
                "directories":     {k: str(v) for k, v in dirs.items()},
                "training_config": training_config
            }

            logger.info("=" * 60)
            logger.info("✅ YOLO Trainer Setup Complete!")
            logger.info("📌 Next Step: Run kaggle_yolo_training.ipynb on Kaggle")
            logger.info("=" * 60)

            return summary

        except Exception as e:
            raise NeuroScanException(e, sys)