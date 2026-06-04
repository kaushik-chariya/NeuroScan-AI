# src/pipeline/training_pipeline.py

"""
Training Pipeline — NeuroScan-AI
----------------------------------
Orchestrates the full training pipeline:
    Stage 1: Data Ingestion
    Stage 2: Data Validation
    Stage 3: Data Transformation
    Stage 4: CNN Trainer Setup
    Stage 5: UNet Trainer Setup
    Stage 6: YOLO Trainer Setup

NOTE:
    Actual model training (CNN, UNet, YOLO) happens on Kaggle.
    This pipeline handles data preparation + config generation only.
"""

import sys
from logger import logger
from exception import NeuroScanException
from src.configuration import ConfigurationManager
from src.components.data_ingestion import DataIngestion
from src.components.data_validation import DataValidation
from src.components.data_transformation import DataTransformation
from src.components.cnn_trainer import CNNTrainer
from src.components.unet_trainer import UNetTrainer
from src.components.yolo_trainer import YOLOTrainer


# ── Stage Names ──────────────────────────────────────────────────────────────
STAGE_1 = "Data Ingestion"
STAGE_2 = "Data Validation"
STAGE_3 = "Data Transformation"
STAGE_4 = "CNN Trainer Setup"
STAGE_5 = "UNet Trainer Setup"
STAGE_6 = "YOLO Trainer Setup"


class TrainingPipeline:
    """
    End-to-end training pipeline for NeuroScan-AI.

    Connects all components in sequence:
        DataIngestion → DataValidation → DataTransformation
        → CNNTrainer → UNetTrainer → YOLOTrainer

    Each stage is independently logged and exception-handled.
    """

    def __init__(self):
        """Initialize TrainingPipeline with ConfigurationManager."""
        try:
            self.config_manager = ConfigurationManager()
            logger.info("🧠 TrainingPipeline initialized")
        except Exception as e:
            raise NeuroScanException(e, sys)

    # ── Stage 1: Data Ingestion ──────────────────────────────────────────────

    def run_data_ingestion(self) -> None:
        """
        Stage 1: Load BraTS 2021 NIfTI files and organize raw data.

        Raises:
            NeuroScanException: If data ingestion fails.
        """
        try:
            logger.info("=" * 60)
            logger.info(f"🚀 Stage 1: {STAGE_1} — Started")
            logger.info("=" * 60)

            config = self.config_manager.get_data_ingestion_config()
            ingestion = DataIngestion(config=config)
            ingestion.run()

            logger.info(f"✅ Stage 1: {STAGE_1} — Completed")

        except Exception as e:
            logger.error(f"❌ Stage 1: {STAGE_1} — Failed")
            raise NeuroScanException(e, sys)

    # ── Stage 2: Data Validation ─────────────────────────────────────────────

    def run_data_validation(self) -> None:
        """
        Stage 2: Validate raw data — file existence, format, integrity.

        Raises:
            NeuroScanException: If data validation fails.
        """
        try:
            logger.info("=" * 60)
            logger.info(f"🚀 Stage 2: {STAGE_2} — Started")
            logger.info("=" * 60)

            config = self.config_manager.get_data_validation_config()
            validation = DataValidation(config=config)
            validation.run()

            logger.info(f"✅ Stage 2: {STAGE_2} — Completed")

        except Exception as e:
            logger.error(f"❌ Stage 2: {STAGE_2} — Failed")
            raise NeuroScanException(e, sys)

    # ── Stage 3: Data Transformation ─────────────────────────────────────────

    def run_data_transformation(self) -> None:
        """
        Stage 3: Normalize, resize, augment, and split data into
                 train/val/test sets.

        Raises:
            NeuroScanException: If data transformation fails.
        """
        try:
            logger.info("=" * 60)
            logger.info(f"🚀 Stage 3: {STAGE_3} — Started")
            logger.info("=" * 60)

            config = self.config_manager.get_data_transformation_config()
            transformation = DataTransformation(config=config)
            transformation.run()

            logger.info(f"✅ Stage 3: {STAGE_3} — Completed")

        except Exception as e:
            logger.error(f"❌ Stage 3: {STAGE_3} — Failed")
            raise NeuroScanException(e, sys)

    # ── Stage 4: CNN Trainer Setup ────────────────────────────────────────────

    def run_cnn_trainer(self) -> None:
        """
        Stage 4: Define CNN architecture and compile model.
                 Actual training runs on Kaggle.

        Raises:
            NeuroScanException: If CNN trainer setup fails.
        """
        try:
            logger.info("=" * 60)
            logger.info(f"🚀 Stage 4: {STAGE_4} — Started")
            logger.info("=" * 60)

            config = self.config_manager.get_cnn_trainer_config()
            trainer = CNNTrainer(config=config)
            trainer.run()

            logger.info(f"✅ Stage 4: {STAGE_4} — Completed")

        except Exception as e:
            logger.error(f"❌ Stage 4: {STAGE_4} — Failed")
            raise NeuroScanException(e, sys)

    # ── Stage 5: UNet Trainer Setup ───────────────────────────────────────────

    def run_unet_trainer(self) -> None:
        """
        Stage 5: Define UNet architecture and compile model.
                 Actual training runs on Kaggle.

        Raises:
            NeuroScanException: If UNet trainer setup fails.
        """
        try:
            logger.info("=" * 60)
            logger.info(f"🚀 Stage 5: {STAGE_5} — Started")
            logger.info("=" * 60)

            config = self.config_manager.get_unet_trainer_config()
            trainer = UNetTrainer(config=config)
            trainer.run()

            logger.info(f"✅ Stage 5: {STAGE_5} — Completed")

        except Exception as e:
            logger.error(f"❌ Stage 5: {STAGE_5} — Failed")
            raise NeuroScanException(e, sys)

    # ── Stage 6: YOLO Trainer Setup ───────────────────────────────────────────

    def run_yolo_trainer(self) -> None:
        """
        Stage 6: Create YOLO dataset YAML and training config.
                 Actual training runs on Kaggle.

        Raises:
            NeuroScanException: If YOLO trainer setup fails.
        """
        try:
            logger.info("=" * 60)
            logger.info(f"🚀 Stage 6: {STAGE_6} — Started")
            logger.info("=" * 60)

            config = self.config_manager.get_yolo_trainer_config()
            trainer = YOLOTrainer(config=config)
            trainer.run()

            logger.info(f"✅ Stage 6: {STAGE_6} — Completed")

        except Exception as e:
            logger.error(f"❌ Stage 6: {STAGE_6} — Failed")
            raise NeuroScanException(e, sys)

    # ── Full Pipeline Run ─────────────────────────────────────────────────────

    def run(self) -> None:
        """
        Executes all 6 stages of the training pipeline in sequence.

        Pipeline Flow:
            Stage 1: Data Ingestion
            Stage 2: Data Validation
            Stage 3: Data Transformation
            Stage 4: CNN Trainer Setup
            Stage 5: UNet Trainer Setup
            Stage 6: YOLO Trainer Setup

        Raises:
            NeuroScanException: If any stage fails.
        """
        try:
            logger.info("=" * 60)
            logger.info("🧠 NeuroScan-AI — Training Pipeline Starting")
            logger.info("=" * 60)

            self.run_data_ingestion()
            self.run_data_validation()
            self.run_data_transformation()
            self.run_cnn_trainer()
            self.run_unet_trainer()
            self.run_yolo_trainer()

            logger.info("=" * 60)
            logger.info("🎉 Training Pipeline Completed Successfully!")
            logger.info("📌 Next Step: Run Kaggle notebooks for model training")
            logger.info("   → notebooks/kaggle_cnn_training.ipynb")
            logger.info("   → notebooks/kaggle_unet_training.ipynb")
            logger.info("   → notebooks/kaggle_yolo_training.ipynb")
            logger.info("=" * 60)

        except Exception as e:
            logger.error("❌ Training Pipeline Failed!")
            raise NeuroScanException(e, sys)


# ── Entry Point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    pipeline = TrainingPipeline()
    pipeline.run()