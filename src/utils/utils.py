# src/utils/utils.py

import os
import yaml
import shutil
import numpy as np
from pathlib import Path
from typing import Any, Dict, List
from logger import logger
from exception import NeuroScanException
import sys


# ── Config & Params ──────────────────────────────────

def read_yaml(path: Path) -> Dict:
    """Reads a YAML file and returns a dictionary."""
    try:
        with open(path, "r") as f:
            config = yaml.safe_load(f)
        logger.info(f"✅ YAML loaded successfully: {path}")
        return config
    except Exception as e:
        raise NeuroScanException(e, sys)


def write_yaml(path: Path, data: Dict) -> None:
    """Writes a dictionary to a YAML file."""
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(data, f)
        logger.info(f"✅ YAML saved: {path}")
    except Exception as e:
        raise NeuroScanException(e, sys)


# ── Directory Utils ───────────────────────────────────

def create_directories(paths: List[Path]) -> None:
    """Creates multiple directories at once."""
    try:
        for path in paths:
            os.makedirs(path, exist_ok=True)
            logger.info(f"📁 Directory created: {path}")
    except Exception as e:
        raise NeuroScanException(e, sys)


# ── MRI Preprocessing Utils ───────────────────────────

def normalize_mri(volume: np.ndarray) -> np.ndarray:
    """
    Normalizes MRI volume to range [0, 1].
    Handles zero division safely.
    """
    try:
        min_val = volume.min()
        max_val = volume.max()
        if max_val - min_val == 0:
            return volume
        normalized = (volume - min_val) / (max_val - min_val)
        logger.info(f"✅ MRI normalized | Min: {min_val:.4f} | Max: {max_val:.4f}")
        return normalized
    except Exception as e:
        raise NeuroScanException(e, sys)


# ── Model Utils ───────────────────────────────────────

def save_model(model: Any, path: Path) -> None:
    """Saves a Keras/TF model."""
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        model.save(path)
        logger.info(f"💾 Model saved: {path}")
    except Exception as e:
        raise NeuroScanException(e, sys)
def load_model_keras(path: Path) -> Any:
    try:
        import tensorflow as tf
        import h5py

        # Step 1: Architecture manually banao (same as Kaggle notebook)
        def build_cnn_model(input_shape=(128, 128, 3)):
            inputs = tf.keras.layers.Input(shape=input_shape)

            x = tf.keras.layers.Conv2D(32, (3,3), padding='same')(inputs)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.Activation('relu')(x)
            x = tf.keras.layers.Conv2D(32, (3,3), padding='same')(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.Activation('relu')(x)
            x = tf.keras.layers.MaxPooling2D((2,2))(x)
            x = tf.keras.layers.Dropout(0.25)(x)

            x = tf.keras.layers.Conv2D(64, (3,3), padding='same')(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.Activation('relu')(x)
            x = tf.keras.layers.Conv2D(64, (3,3), padding='same')(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.Activation('relu')(x)
            x = tf.keras.layers.MaxPooling2D((2,2))(x)
            x = tf.keras.layers.Dropout(0.25)(x)

            x = tf.keras.layers.Conv2D(128, (3,3), padding='same')(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.Activation('relu')(x)
            x = tf.keras.layers.Conv2D(128, (3,3), padding='same')(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.Activation('relu')(x)
            x = tf.keras.layers.MaxPooling2D((2,2))(x)
            x = tf.keras.layers.Dropout(0.30)(x)

            x = tf.keras.layers.Conv2D(256, (3,3), padding='same')(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.Activation('relu')(x)
            x = tf.keras.layers.MaxPooling2D((2,2))(x)
            x = tf.keras.layers.Dropout(0.40)(x)

            x = tf.keras.layers.GlobalAveragePooling2D()(x)
            x = tf.keras.layers.Dense(256, activation='relu')(x)
            x = tf.keras.layers.Dropout(0.50)(x)
            x = tf.keras.layers.Dense(128, activation='relu')(x)
            x = tf.keras.layers.Dropout(0.30)(x)
            outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)

            return tf.keras.Model(inputs, outputs)

        # Step 2: Model banao
        model = build_cnn_model()

        # Step 3: Weights load karo
        model.load_weights(str(path))

        logger.info(f"✅ Keras model loaded: {path}")
        return model

    except Exception as e:
        raise NeuroScanException(e, sys)
        


def load_model_torch(path: Path) -> Any:
    """Loads a PyTorch model."""
    try:
        import torch
        model = torch.load(path, map_location="cpu")
        logger.info(f"✅ PyTorch model loaded: {path}")
        return model
    except Exception as e:
        raise NeuroScanException(e, sys)


# ── File Utils ────────────────────────────────────────

def get_file_size(path: Path) -> str:
    """Returns file size in MB."""
    try:
        size = os.path.getsize(path) / (1024 * 1024)
        return f"{size:.2f} MB"
    except Exception as e:
        raise NeuroScanException(e, sys)


def copy_file(src: Path, dest: Path) -> None:
    """Copies a file from src to dest."""
    try:
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        shutil.copy2(src, dest)
        logger.info(f"📋 File copied: {src} → {dest}")
    except Exception as e:
        raise NeuroScanException(e, sys)