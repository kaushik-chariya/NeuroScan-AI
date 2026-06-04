# src/components/unet_trainer.py

import os
import sys
import numpy as np
from pathlib import Path
from typing import Tuple, Dict
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
    TensorBoard
)
from logger import logger
from exception import NeuroScanException
from src.entity.config_entity import UNetTrainerConfig


class UNetTrainer:
    def __init__(self, config: UNetTrainerConfig):
        self.config = config


    # ── Data Loading ──────────────────────────────────

    def load_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Loads train and validation data
        from split directory.
        """
        try:
            split_dir = Path("artifacts/data/split")

            X_train  = np.load(split_dir / "X_train.npy")
            X_val    = np.load(split_dir / "X_val.npy")
            ym_train = np.load(split_dir / "ym_train.npy")
            ym_val   = np.load(split_dir / "ym_val.npy")

            logger.info(f"✅ X_train shape  : {X_train.shape}")
            logger.info(f"✅ X_val shape    : {X_val.shape}")
            logger.info(f"✅ y_train shape  : {ym_train.shape}")
            logger.info(f"✅ y_val shape    : {ym_val.shape}")

            return X_train, X_val, ym_train, ym_val

        except Exception as e:
            raise NeuroScanException(e, sys)


    def preprocess_masks(
        self,
        y_train: np.ndarray,
        y_val: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Preprocesses segmentation masks.
        Maps class 4 → 3 and expands dims.
        """
        try:
            # Map class 4 → 3
            y_train = np.where(y_train == 4, 3, y_train)
            y_val   = np.where(y_val   == 4, 3, y_val)

            # Expand dims → (N, H, W, 1)
            y_train = np.expand_dims(y_train, axis=-1)
            y_val   = np.expand_dims(y_val,   axis=-1)

            logger.info(f"✅ Masks preprocessed | Train: {y_train.shape} | Val: {y_val.shape}")
            return y_train, y_val

        except Exception as e:
            raise NeuroScanException(e, sys)


    # ── Model Architecture ────────────────────────────

    def conv_block(self, x, filters: int, name: str):
        """
        Double convolution block with
        BatchNormalization and ReLU activation.
        """
        x = layers.Conv2D(filters, 3, padding="same", name=f"{name}_conv1")(x)
        x = layers.BatchNormalization(name=f"{name}_bn1")(x)
        x = layers.Activation("relu", name=f"{name}_relu1")(x)
        x = layers.Conv2D(filters, 3, padding="same", name=f"{name}_conv2")(x)
        x = layers.BatchNormalization(name=f"{name}_bn2")(x)
        x = layers.Activation("relu", name=f"{name}_relu2")(x)
        return x


    def attention_gate(self, x, gate, filters: int, name: str):
        """
        Attention Gate — focuses on tumor regions.
        Helps model ignore irrelevant brain areas.
        """
        x_conv = layers.Conv2D(filters, 1, padding="same", name=f"{name}_x_conv")(x)
        g_conv = layers.Conv2D(filters, 1, padding="same", name=f"{name}_g_conv")(gate)

        combined = layers.Add(name=f"{name}_add")([x_conv, g_conv])
        combined = layers.Activation("relu", name=f"{name}_relu")(combined)
        combined = layers.Conv2D(1, 1, padding="same", name=f"{name}_psi")(combined)
        combined = layers.Activation("sigmoid", name=f"{name}_sigmoid")(combined)

        attended = layers.Multiply(name=f"{name}_multiply")([x, combined])
        return attended


    def build_model(self) -> Model:
        """
        Builds Attention UNet model for
        brain tumor segmentation.
        Input  : (128, 128, 4) — 4 MRI modalities
        Output : (128, 128, 4) — 4 class segmentation mask
        """
        try:
            inputs = keras.Input(
                shape=(*self.config.image_size, 4),
                name="input"
            )

            # ── Encoder ───────────────────────────────
            # Block 1
            e1 = self.conv_block(inputs, 64,  "enc1")
            p1 = layers.MaxPooling2D(2, name="pool1")(e1)
            p1 = layers.Dropout(0.1, name="drop1")(p1)

            # Block 2
            e2 = self.conv_block(p1, 128, "enc2")
            p2 = layers.MaxPooling2D(2, name="pool2")(e2)
            p2 = layers.Dropout(0.1, name="drop2")(p2)

            # Block 3
            e3 = self.conv_block(p2, 256, "enc3")
            p3 = layers.MaxPooling2D(2, name="pool3")(e3)
            p3 = layers.Dropout(0.2, name="drop3")(p3)

            # Block 4
            e4 = self.conv_block(p3, 512, "enc4")
            p4 = layers.MaxPooling2D(2, name="pool4")(e4)
            p4 = layers.Dropout(0.2, name="drop4")(p4)

            # ── Bottleneck ────────────────────────────
            b = self.conv_block(p4, 1024, "bottleneck")

            # ── Decoder ───────────────────────────────
            # Block 1
            u1 = layers.Conv2DTranspose(512, 2, strides=2, name="up1")(b)
            a1 = self.attention_gate(e4, u1, 512, "att1")
            u1 = layers.Concatenate(name="concat1")([u1, a1])
            u1 = layers.Dropout(0.2, name="drop5")(u1)
            d1 = self.conv_block(u1, 512, "dec1")

            # Block 2
            u2 = layers.Conv2DTranspose(256, 2, strides=2, name="up2")(d1)
            a2 = self.attention_gate(e3, u2, 256, "att2")
            u2 = layers.Concatenate(name="concat2")([u2, a2])
            u2 = layers.Dropout(0.2, name="drop6")(u2)
            d2 = self.conv_block(u2, 256, "dec2")

            # Block 3
            u3 = layers.Conv2DTranspose(128, 2, strides=2, name="up3")(d2)
            a3 = self.attention_gate(e2, u3, 128, "att3")
            u3 = layers.Concatenate(name="concat3")([u3, a3])
            u3 = layers.Dropout(0.1, name="drop7")(u3)
            d3 = self.conv_block(u3, 128, "dec3")

            # Block 4
            u4 = layers.Conv2DTranspose(64, 2, strides=2, name="up4")(d3)
            a4 = self.attention_gate(e1, u4, 64, "att4")
            u4 = layers.Concatenate(name="concat4")([u4, a4])
            u4 = layers.Dropout(0.1, name="drop8")(u4)
            d4 = self.conv_block(u4, 64, "dec4")

            # ── Output ────────────────────────────────
            outputs = layers.Conv2D(
                self.config.num_classes,
                1,
                activation="softmax",
                name="output"
            )(d4)

            model = Model(inputs, outputs, name="NeuroScan_AttentionUNet")

            logger.info("✅ Attention UNet built successfully")
            logger.info(f"📊 Total parameters: {model.count_params():,}")

            return model

        except Exception as e:
            raise NeuroScanException(e, sys)


    # ── Loss Functions ────────────────────────────────

    def dice_loss(self, y_true, y_pred):
        """
        Dice Loss — handles class imbalance
        better than standard cross entropy.
        """
        smooth    = 1e-6
        y_true_f  = tf.cast(tf.reshape(y_true, [-1]), tf.float32)
        y_pred_f  = tf.reshape(y_pred,  [-1])
        intersect = tf.reduce_sum(y_true_f * y_pred_f)
        return 1 - (2.0 * intersect + smooth) / (
            tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth
        )


    def combined_loss(self, y_true, y_pred):
        """
        Combined Loss = Dice Loss + BCE Loss.
        Best for medical image segmentation.
        """
        bce  = keras.losses.sparse_categorical_crossentropy(y_true, y_pred)
        dice = self.dice_loss(y_true, y_pred)
        return bce + dice


    # ── Metrics ───────────────────────────────────────

    def dice_coefficient(self, y_true, y_pred):
        """Dice Coefficient metric."""
        smooth    = 1e-6
        y_true_f  = tf.cast(tf.reshape(y_true, [-1]), tf.float32)
        y_pred_f  = tf.reshape(tf.argmax(y_pred, axis=-1), [-1])
        y_pred_f  = tf.cast(y_pred_f, tf.float32)
        intersect = tf.reduce_sum(y_true_f * y_pred_f)
        return (2.0 * intersect + smooth) / (
            tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth
        )


    def iou_score(self, y_true, y_pred):
        """IoU (Intersection over Union) metric."""
        smooth    = 1e-6
        y_true_f  = tf.cast(tf.reshape(y_true, [-1]), tf.float32)
        y_pred_f  = tf.cast(tf.reshape(tf.argmax(y_pred, axis=-1), [-1]), tf.float32)
        intersect = tf.reduce_sum(y_true_f * y_pred_f)
        union     = tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) - intersect
        return (intersect + smooth) / (union + smooth)


    # ── Compile + Callbacks ───────────────────────────

    def compile_model(self, model: Model) -> Model:
        """Compiles model with combined loss and metrics."""
        try:
            model.compile(
                optimizer=keras.optimizers.Adam(
                    learning_rate=self.config.learning_rate
                ),
                loss=self.combined_loss,
                metrics=[
                    self.dice_coefficient,
                    self.iou_score,
                    "accuracy"
                ]
            )
            logger.info("✅ UNet model compiled")
            return model

        except Exception as e:
            raise NeuroScanException(e, sys)


    def get_callbacks(self) -> list:
        """Returns training callbacks."""
        try:
            os.makedirs(self.config.root_dir, exist_ok=True)

            callbacks = [
                EarlyStopping(
                    monitor="val_dice_coefficient",
                    patience=10,
                    mode="max",
                    restore_best_weights=True,
                    verbose=1
                ),
                ModelCheckpoint(
                    filepath=str(self.config.model_path),
                    monitor="val_dice_coefficient",
                    save_best_only=True,
                    mode="max",
                    verbose=1
                ),
                ReduceLROnPlateau(
                    monitor="val_loss",
                    factor=0.5,
                    patience=5,
                    min_lr=1e-7,
                    verbose=1
                ),
                TensorBoard(
                    log_dir="logs/unet",
                    histogram_freq=1
                )
            ]

            logger.info("✅ UNet callbacks ready")
            return callbacks

        except Exception as e:
            raise NeuroScanException(e, sys)


    # ── Evaluate ──────────────────────────────────────

    def evaluate_model(
        self,
        model: Model,
        X_test: np.ndarray,
        y_test: np.ndarray
    ) -> Dict:
        """Evaluates model on test data."""
        try:
            y_test = np.where(y_test == 4, 3, y_test)
            y_test = np.expand_dims(y_test, axis=-1)

            results = model.evaluate(X_test, y_test, verbose=0)

            metrics = {
                "test_loss"  : round(results[0], 4),
                "test_dice"  : round(results[1], 4),
                "test_iou"   : round(results[2], 4),
                "test_acc"   : round(results[3], 4),
            }

            logger.info(f"📊 Test Loss : {metrics['test_loss']}")
            logger.info(f"📊 Test Dice : {metrics['test_dice']}")
            logger.info(f"📊 Test IoU  : {metrics['test_iou']}")
            logger.info(f"📊 Test Acc  : {metrics['test_acc']}")

            return metrics

        except Exception as e:
            raise NeuroScanException(e, sys)


    # ── Main Train ────────────────────────────────────

    def train(self) -> Tuple[Model, Dict, Dict]:
        """
        Main method — runs full UNet training pipeline.
        Returns trained model, history and metrics.
        """
        try:
            logger.info("🚀 UNet Training started")

            # Load data
            X_train, X_val, y_train, y_val = self.load_data()

            # Preprocess masks
            y_train, y_val = self.preprocess_masks(y_train, y_val)

            # Build model
            model = self.build_model()

            # Compile
            model = self.compile_model(model)

            # Summary
            model.summary()

            # Callbacks
            callbacks = self.get_callbacks()

            # Train
            logger.info(f"🏋️ Training for {self.config.epochs} epochs...")
            history = model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=self.config.epochs,
                batch_size=self.config.batch_size,
                callbacks=callbacks,
                verbose=1
            )

            logger.info("✅ UNet Training completed!")

            # Evaluate
            X_test = np.load("artifacts/data/split/X_test.npy")
            y_test = np.load("artifacts/data/split/ym_test.npy")
            metrics = self.evaluate_model(model, X_test, y_test)

            # Save
            model.save(str(self.config.model_path))
            logger.info(f"💾 UNet model saved: {self.config.model_path}")

            logger.info("🎉 UNet Pipeline completed!")
            return model, history.history, metrics

        except Exception as e:
            raise NeuroScanException(e, sys)