# src/entity/config_entity.py

from dataclasses import dataclass
from pathlib import Path


# ── Data Ingestion ────────────────────────────────────
@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    raw_data_dir: Path
    dataset_name: str
    kaggle_dataset: str
    modalities: list
    seg_file_suffix: str


# ── Data Validation ───────────────────────────────────
@dataclass(frozen=True)
class DataValidationConfig:
    root_dir: Path
    status_file: Path
    required_modalities: list
    required_files_per_case: int


# ── Data Transformation ───────────────────────────────
@dataclass(frozen=True)
class DataTransformationConfig:
    root_dir: Path
    split_dir: Path
    image_size: tuple
    train_ratio: float
    val_ratio: float
    test_ratio: float
    random_seed: int
    normalize: bool
    slice_axis: int
    slice_index: int


# ── CNN Trainer ───────────────────────────────────────
@dataclass(frozen=True)
class CNNTrainerConfig:
    root_dir: Path
    model_name: str
    model_path: Path
    image_size: tuple
    num_classes: int
    epochs: int
    batch_size: int
    learning_rate: float
    dropout_rate: float
    optimizer: str
    loss: str
    metrics: list


# ── UNet Trainer ──────────────────────────────────────
@dataclass(frozen=True)
class UNetTrainerConfig:
    root_dir: Path
    model_name: str
    model_path: Path
    image_size: tuple
    num_classes: int
    epochs: int
    batch_size: int
    learning_rate: float
    optimizer: str
    loss: str
    metrics: list


# ── YOLO Trainer ──────────────────────────────────────
@dataclass(frozen=True)
class YOLOTrainerConfig:
    root_dir: Path
    model_name: str
    model_path: Path
    pretrained_weights: str
    image_size: int
    epochs: int
    batch_size: int
    learning_rate: float
    confidence_threshold: float
    iou_threshold: float
    num_classes: int
    class_names: list