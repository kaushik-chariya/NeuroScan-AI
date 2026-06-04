# src/configuration.py

from pathlib import Path
from src.constants import (
    CONFIG_FILE_PATH,
    PARAMS_FILE_PATH
)
from src.utils.utils import read_yaml, create_directories
from src.entity.config_entity import (
    DataIngestionConfig,
    DataValidationConfig,
    DataTransformationConfig,
    CNNTrainerConfig,
    UNetTrainerConfig,
    YOLOTrainerConfig
)
from logger import logger
from exception import NeuroScanException
import sys


class ConfigurationManager:
    def __init__(
        self,
        config_path: Path = CONFIG_FILE_PATH,
        params_path: Path = PARAMS_FILE_PATH
    ):
        try:
            self.config = read_yaml(config_path)
            self.params = read_yaml(params_path)

            create_directories([
                Path(self.config["data_ingestion"]["root_dir"]),
                Path(self.config["data_validation"]["root_dir"]),
                Path(self.config["data_transformation"]["root_dir"]),
                Path(self.config["cnn_trainer"]["root_dir"]),
                Path(self.config["unet_trainer"]["root_dir"]),
                Path(self.config["yolo_trainer"]["root_dir"]),
            ])

            logger.info("✅ ConfigurationManager initialized successfully")

        except Exception as e:
            raise NeuroScanException(e, sys)


    # ── Data Ingestion ────────────────────────────────
    def get_data_ingestion_config(self) -> DataIngestionConfig:
        try:
            cfg = self.config["data_ingestion"]

            create_directories([Path(cfg["raw_data_dir"])])

            config = DataIngestionConfig(
                root_dir=Path(cfg["root_dir"]),
                raw_data_dir=Path(cfg["raw_data_dir"]),
                dataset_name=cfg["dataset_name"],
                kaggle_dataset=cfg["kaggle_dataset"],
                modalities=cfg["modalities"],
                seg_file_suffix=cfg["seg_file_suffix"]
            )

            logger.info("✅ Data Ingestion config loaded")
            return config

        except Exception as e:
            raise NeuroScanException(e, sys)


    # ── Data Validation ───────────────────────────────
    def get_data_validation_config(self) -> DataValidationConfig:
        try:
            cfg = self.config["data_validation"]

            config = DataValidationConfig(
                root_dir=Path(cfg["root_dir"]),
                status_file=Path(cfg["status_file"]),
                required_modalities=cfg["required_modalities"],
                required_files_per_case=cfg["required_files_per_case"]
            )

            logger.info("✅ Data Validation config loaded")
            return config

        except Exception as e:
            raise NeuroScanException(e, sys)


    # ── Data Transformation ───────────────────────────
    def get_data_transformation_config(self) -> DataTransformationConfig:
        try:
            cfg = self.config["data_transformation"]

            create_directories([
                Path(cfg["root_dir"]),
                Path(cfg["split_dir"])
            ])

            config = DataTransformationConfig(
                root_dir=Path(cfg["root_dir"]),
                split_dir=Path(cfg["split_dir"]),
                image_size=tuple(cfg["image_size"]),
                train_ratio=cfg["train_ratio"],
                val_ratio=cfg["val_ratio"],
                test_ratio=cfg["test_ratio"],
                random_seed=cfg["random_seed"],
                normalize=cfg["normalize"],
                slice_axis=cfg["slice_axis"],
                slice_index=cfg["slice_index"]
            )

            logger.info("✅ Data Transformation config loaded")
            return config

        except Exception as e:
            raise NeuroScanException(e, sys)


    # ── CNN Trainer ───────────────────────────────────
    def get_cnn_trainer_config(self) -> CNNTrainerConfig:
        try:
            cfg = self.config["cnn_trainer"]

            create_directories([Path(cfg["root_dir"])])

            config = CNNTrainerConfig(
                root_dir=Path(cfg["root_dir"]),
                model_name=cfg["model_name"],
                model_path=Path(cfg["model_path"]),
                image_size=tuple(cfg["image_size"]),
                num_classes=cfg["num_classes"],
                epochs=cfg["epochs"],
                batch_size=cfg["batch_size"],
                learning_rate=cfg["learning_rate"],
                dropout_rate=cfg["dropout_rate"],
                optimizer=cfg["optimizer"],
                loss=cfg["loss"],
                metrics=cfg["metrics"]
            )

            logger.info("✅ CNN Trainer config loaded")
            return config

        except Exception as e:
            raise NeuroScanException(e, sys)


    # ── UNet Trainer ──────────────────────────────────
    def get_unet_trainer_config(self) -> UNetTrainerConfig:
        try:
            cfg = self.config["unet_trainer"]

            create_directories([Path(cfg["root_dir"])])

            config = UNetTrainerConfig(
                root_dir=Path(cfg["root_dir"]),
                model_name=cfg["model_name"],
                model_path=Path(cfg["model_path"]),
                image_size=tuple(cfg["image_size"]),
                num_classes=cfg["num_classes"],
                epochs=cfg["epochs"],
                batch_size=cfg["batch_size"],
                learning_rate=cfg["learning_rate"],
                optimizer=cfg["optimizer"],
                loss=cfg["loss"],
                metrics=cfg["metrics"]
            )

            logger.info("✅ UNet Trainer config loaded")
            return config

        except Exception as e:
            raise NeuroScanException(e, sys)


    # ── YOLO Trainer ──────────────────────────────────
    def get_yolo_trainer_config(self) -> YOLOTrainerConfig:
        try:
            cfg = self.config["yolo_trainer"]

            create_directories([Path(cfg["root_dir"])])

            config = YOLOTrainerConfig(
                root_dir=Path(cfg["root_dir"]),
                model_name=cfg["model_name"],
                model_path=Path(cfg["model_path"]),
                pretrained_weights=cfg["pretrained_weights"],
                image_size=cfg["image_size"],
                epochs=cfg["epochs"],
                batch_size=cfg["batch_size"],
                learning_rate=cfg["learning_rate"],
                confidence_threshold=cfg["confidence_threshold"],
                iou_threshold=cfg["iou_threshold"],
                num_classes=cfg["num_classes"],
                class_names=cfg["class_names"]
            )

            logger.info("✅ YOLO Trainer config loaded")
            return config

        except Exception as e:
            raise NeuroScanException(e, sys)