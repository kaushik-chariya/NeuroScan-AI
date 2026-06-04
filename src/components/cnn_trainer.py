# src/components/cnn_trainer.py

import os
import sys
import numpy as np
from pathlib import Path
from typing import Tuple, Dict
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import EfficientNetB4
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
    TensorBoard
)
from tensorflow.keras.utils import to_categorical
from sklearn.utils.class_weight import compute_class_weight
from logger import logger
from exception import NeuroScanException
from src.entity.config_entity import CNNTrainerConfig


class CNNTrainer:
    def __init__(self, config: CNNTrainerConfig):
        self.config = config


    def load_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Loads train and validation data
        from split directory.
        """
        try:
            split_dir = Path("artifacts/data/split")

            X_train  = np.load(split_dir / "X_train.npy")
            X_val    = np.load(split_dir / "X_val.npy")
            yl_train = np.load(split_dir / "yl_train.npy")
            yl_val   = np.load(split_dir / "yl_val.npy")

            logger.info(f"✅ X_train shape  : {X_train.shape}")
            logger.info(f"✅ X_val shape    : {X_val.shape}")
            logger.info(f"✅ y_train shape  : {yl_train.shape}")
            logger.info(f"✅ y_val shape    : {yl_val.shape}")

            return X_train, X_val, yl_train, yl_val

        except Exception as e:
            raise NeuroScanException(e, sys)


    def preprocess_labels(
        self,
        y_train: np.ndarray,
        y_val: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Converts labels to one-hot encoding.
        """
        try:
            y_train_cat = to_categorical(y_train, num_classes=self.config.num_classes)
            y_val_cat   = to_categorical(y_val,   num_classes=self.config.num_classes)

            logger.info(f"✅ Labels one-hot encoded | Classes: {self.config.num_classes}")
            return y_train_cat, y_val_cat

        except Exception as e:
            raise NeuroScanException(e, sys)


    def compute_class_weights(self, y_train: np.ndarray) -> Dict:
        """
        Computes class weights to handle
        class imbalance in BraTS dataset.
        """
        try:
            classes = np.unique(y_train)
            weights = compute_class_weight(
                class_weight="balanced",
                classes=classes,
                y=y_train
            )
            class_weight_dict = dict(zip(classes, weights))
            logger.info(f"✅ Class weights: {class_weight_dict}")
            return class_weight_dict

        except Exception as e:
            raise NeuroScanException(e, sys)


    def build_model(self) -> Model:
        """
        Builds EfficientNetB4 based CNN model
        with Transfer Learning for brain tumor classification.
        Input: (128, 128, 4) — 4 MRI modalities
        Output: 4 classes softmax
        """
        try:
            inputs = keras.Input(shape=(*self.config.image_size, 4))

            # Convert 4 channel input to 3 channel for EfficientNet
            x = layers.Conv2D(
                filters=3,
                kernel_size=1,
                padding="same",
                name="channel_adapter"
            )(inputs)

            # EfficientNetB4 Backbone — pretrained on ImageNet
            backbone = EfficientNetB4(
                include_top=False,
                weights="imagenet",
                input_tensor=x
            )

            # Freeze first 100 layers — fine tune rest
            for layer in backbone.layers[:100]:
                layer.trainable = False
            for layer in backbone.layers[100:]:
                layer.trainable = True

            x = backbone.output

            # Global Average Pooling
            x = layers.GlobalAveragePooling2D()(x)

            # Dense Block 1
            x = layers.Dense(512)(x)
            x = layers.BatchNormalization()(x)
            x = layers.Activation("relu")(x)
            x = layers.Dropout(self.config.dropout_rate)(x)

            # Dense Block 2
            x = layers.Dense(256)(x)
            x = layers.BatchNormalization()(x)
            x = layers.Activation("relu")(x)
            x = layers.Dropout(self.config.dropout_rate)(x)

            # Dense Block 3
            x = layers.Dense(128)(x)
            x = layers.BatchNormalization()(x)
            x = layers.Activation("relu")(x)
            x = layers.Dropout(self.config.dropout_rate / 2)(x)

            # Output Layer
            outputs = layers.Dense(
                self.config.num_classes,
                activation="softmax"
            )(x)

            model = Model(inputs=inputs, outputs=outputs, name="NeuroScan_CNN")

            logger.info(f"✅ CNN Model built successfully")
            logger.info(f"📊 Total parameters: {model.count_params():,}")

            return model

        except Exception as e:
            raise NeuroScanException(e, sys)


    def compile_model(self, model: Model) -> Model:
        """
        Compiles model with optimizer, loss and metrics.
        """
        try:
            optimizer = keras.optimizers.Adam(
                learning_rate=self.config.learning_rate
            )

            model.compile(
                optimizer=optimizer,
                loss=self.config.loss,
                metrics=[
                    "accuracy",
                    keras.metrics.AUC(name="auc"),
                    keras.metrics.Precision(name="precision"),
                    keras.metrics.Recall(name="recall")
                ]
            )

            logger.info("✅ Model compiled successfully")
            return model

        except Exception as e:
            raise NeuroScanException(e, sys)


    def get_callbacks(self) -> list:
        """
        Returns list of training callbacks:
        EarlyStopping, ModelCheckpoint,
        ReduceLROnPlateau, TensorBoard.
        """
        try:
            os.makedirs(self.config.root_dir, exist_ok=True)

            callbacks = [
                # Stop training if no improvement
                EarlyStopping(
                    monitor="val_loss",
                    patience=10,
                    restore_best_weights=True,
                    verbose=1
                ),

                # Save best model
                ModelCheckpoint(
                    filepath=str(self.config.model_path),
                    monitor="val_accuracy",
                    save_best_only=True,
                    verbose=1
                ),

                # Reduce LR on plateau
                ReduceLROnPlateau(
                    monitor="val_loss",
                    factor=0.5,
                    patience=5,
                    min_lr=1e-7,
                    verbose=1
                ),

                # TensorBoard logs
                TensorBoard(
                    log_dir="logs/cnn",
                    histogram_freq=1
                )
            ]

            logger.info("✅ Callbacks ready")
            return callbacks

        except Exception as e:
            raise NeuroScanException(e, sys)


    def evaluate_model(
        self,
        model: Model,
        X_test: np.ndarray,
        y_test: np.ndarray
    ) -> Dict:
        """
        Evaluates model on test data.
        Returns metrics dictionary.
        """
        try:
            y_test_cat = to_categorical(y_test, num_classes=self.config.num_classes)
            results    = model.evaluate(X_test, y_test_cat, verbose=0)

            metrics = {
                "test_loss"      : round(results[0], 4),
                "test_accuracy"  : round(results[1], 4),
                "test_auc"       : round(results[2], 4),
                "test_precision" : round(results[3], 4),
                "test_recall"    : round(results[4], 4),
            }

            logger.info(f"📊 Test Loss      : {metrics['test_loss']}")
            logger.info(f"📊 Test Accuracy  : {metrics['test_accuracy']}")
            logger.info(f"📊 Test AUC       : {metrics['test_auc']}")
            logger.info(f"📊 Test Precision : {metrics['test_precision']}")
            logger.info(f"📊 Test Recall    : {metrics['test_recall']}")

            return metrics

        except Exception as e:
            raise NeuroScanException(e, sys)


    def train(self) -> Tuple[Model, Dict]:
        """
        Main method — runs full CNN training pipeline.
        Returns trained model and history.
        """
        try:
            logger.info("🚀 CNN Training started")

            # Load data
            X_train, X_val, y_train, y_val = self.load_data()

            # Preprocess labels
            y_train_cat, y_val_cat = self.preprocess_labels(y_train, y_val)

            # Compute class weights
            class_weights = self.compute_class_weights(y_train)

            # Build model
            model = self.build_model()

            # Compile model
            model = self.compile_model(model)

            # Print summary
            model.summary()

            # Get callbacks
            callbacks = self.get_callbacks()

            # Train model
            logger.info(f"🏋️ Training for {self.config.epochs} epochs...")
            history = model.fit(
                X_train, y_train_cat,
                validation_data=(X_val, y_val_cat),
                epochs=self.config.epochs,
                batch_size=self.config.batch_size,
                class_weight=class_weights,
                callbacks=callbacks,
                verbose=1
            )

            logger.info("✅ CNN Training completed!")

            # Load test data and evaluate
            X_test  = np.load("artifacts/data/split/X_test.npy")
            y_test  = np.load("artifacts/data/split/yl_test.npy")
            metrics = self.evaluate_model(model, X_test, y_test)

            # Save final model
            model.save(str(self.config.model_path))
            logger.info(f"💾 Model saved: {self.config.model_path}")

            logger.info("🎉 CNN Pipeline completed!")
            return model, history.history, metrics

        except Exception as e:
            raise NeuroScanException(e, sys)