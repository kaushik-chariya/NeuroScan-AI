# src/components/data_transformation.py

import os
import sys
import numpy as np
import nibabel as nib
import cv2
from pathlib import Path
from typing import Tuple, List
from sklearn.model_selection import train_test_split
from logger import logger
from exception import NeuroScanException
from src.entity.config_entity import DataTransformationConfig


class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config


    def load_patient_data(self, patient_dir: Path) -> Tuple[np.ndarray, np.ndarray]:
        """
        Loads all 4 modalities + segmentation mask
        for a given patient.
        Returns stacked volume and mask.
        """
        try:
            patient_id = patient_dir.name
            modalities = []

            # Load all 4 modalities
            for modality in ["t1", "t1ce", "t2", "flair"]:
                file_path = patient_dir / f"{patient_id}_{modality}.nii.gz"
                img = nib.load(str(file_path))
                data = img.get_fdata()
                modalities.append(data)

            # Stack modalities → shape: (H, W, D, 4)
            volume = np.stack(modalities, axis=-1)

            # Load segmentation mask
            seg_path = patient_dir / f"{patient_id}_seg.nii.gz"
            mask = nib.load(str(seg_path)).get_fdata()

            logger.info(f"✅ Loaded: {patient_id} | Volume: {volume.shape} | Mask: {mask.shape}")
            return volume, mask

        except Exception as e:
            raise NeuroScanException(e, sys)


    def normalize_volume(self, volume: np.ndarray) -> np.ndarray:
        """
        Normalizes each modality independently to [0, 1].
        Handles zero division safely.
        """
        try:
            normalized = np.zeros_like(volume, dtype=np.float32)

            for i in range(volume.shape[-1]):
                modality = volume[..., i]
                min_val  = modality.min()
                max_val  = modality.max()

                if max_val - min_val > 0:
                    normalized[..., i] = (modality - min_val) / (max_val - min_val)
                else:
                    normalized[..., i] = modality

            logger.info(f"✅ Volume normalized | Shape: {normalized.shape}")
            return normalized

        except Exception as e:
            raise NeuroScanException(e, sys)


    def extract_best_slices(
        self,
        volume: np.ndarray,
        mask: np.ndarray,
        num_slices: int = 10
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Extracts slices with maximum tumor area.
        Skips empty/background slices.
        Returns list of best slices and masks.
        """
        try:
            depth = volume.shape[2]
            slice_scores = []

            for i in range(depth):
                mask_slice = mask[:, :, i]
                # Score = number of tumor pixels
                score = np.sum(mask_slice > 0)
                slice_scores.append((i, score))

            # Sort by tumor area — best slices first
            slice_scores.sort(key=lambda x: x[1], reverse=True)

            best_slices  = []
            best_masks   = []

            for idx, score in slice_scores[:num_slices]:
                if score > 0:
                    vol_slice  = volume[:, :, idx, :]   # (H, W, 4)
                    mask_slice = mask[:, :, idx]         # (H, W)
                    best_slices.append(vol_slice)
                    best_masks.append(mask_slice)

            logger.info(f"✅ Best slices extracted: {len(best_slices)}")
            return best_slices, best_masks

        except Exception as e:
            raise NeuroScanException(e, sys)


    def resize_slice(
        self,
        volume_slice: np.ndarray,
        mask_slice: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Resizes volume slice and mask to target image size.
        """
        try:
            target = self.config.image_size  # (128, 128)
            resized_modalities = []

            for i in range(volume_slice.shape[-1]):
                resized = cv2.resize(volume_slice[..., i], target)
                resized_modalities.append(resized)

            resized_volume = np.stack(resized_modalities, axis=-1)
            resized_mask   = cv2.resize(
                mask_slice,
                target,
                interpolation=cv2.INTER_NEAREST
            )

            return resized_volume, resized_mask

        except Exception as e:
            raise NeuroScanException(e, sys)


    def augment_slice(
        self,
        volume_slice: np.ndarray,
        mask_slice: np.ndarray
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Applies augmentation to volume slice and mask.
        Returns original + augmented slices.
        Augmentations: horizontal flip, vertical flip, 90° rotation.
        """
        try:
            augmented_volumes = [volume_slice]
            augmented_masks   = [mask_slice]

            # Horizontal flip
            aug_vol  = np.flip(volume_slice, axis=1).copy()
            aug_mask = np.flip(mask_slice, axis=1).copy()
            augmented_volumes.append(aug_vol)
            augmented_masks.append(aug_mask)

            # Vertical flip
            aug_vol  = np.flip(volume_slice, axis=0).copy()
            aug_mask = np.flip(mask_slice, axis=0).copy()
            augmented_volumes.append(aug_vol)
            augmented_masks.append(aug_mask)

            # 90 degree rotation
            aug_vol  = np.rot90(volume_slice, k=1, axes=(0, 1)).copy()
            aug_mask = np.rot90(mask_slice, k=1, axes=(0, 1)).copy()
            augmented_volumes.append(aug_vol)
            augmented_masks.append(aug_mask)

            return augmented_volumes, augmented_masks

        except Exception as e:
            raise NeuroScanException(e, sys)


    def get_tumor_label(self, mask_slice: np.ndarray) -> int:
        """
        Returns tumor classification label from mask.
        0: Background
        1: Necrotic Tumor Core
        2: Peritumoral Edema
        4: Enhancing Tumor → mapped to 3
        """
        try:
            unique = np.unique(mask_slice)
            unique = unique[unique > 0]  # Remove background

            if len(unique) == 0:
                return 0  # Background

            # Return most dominant tumor class
            dominant = int(np.bincount(
                mask_slice.astype(int).flatten()
            ).argmax())

            # Map class 4 → 3
            if dominant == 4:
                dominant = 3

            return dominant

        except Exception as e:
            raise NeuroScanException(e, sys)


    def process_all_patients(self, patient_dirs: List[Path]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Processes all patients — loads, normalizes,
        extracts best slices, augments.
        Returns X (volumes), y_mask (masks), y_label (labels).
        """
        try:
            all_volumes = []
            all_masks   = []
            all_labels  = []

            for i, patient_dir in enumerate(patient_dirs):
                logger.info(f"🔄 Processing patient {i+1}/{len(patient_dirs)}: {patient_dir.name}")

                # Load data
                volume, mask = self.load_patient_data(patient_dir)

                # Normalize
                volume = self.normalize_volume(volume)

                # Extract best slices
                best_slices, best_masks = self.extract_best_slices(volume, mask)

                for vol_slice, mask_slice in zip(best_slices, best_masks):

                    # Resize
                    vol_slice, mask_slice = self.resize_slice(vol_slice, mask_slice)

                    # Augment
                    aug_volumes, aug_masks = self.augment_slice(vol_slice, mask_slice)

                    for aug_vol, aug_mask in zip(aug_volumes, aug_masks):
                        label = self.get_tumor_label(aug_mask)
                        all_volumes.append(aug_vol)
                        all_masks.append(aug_mask)
                        all_labels.append(label)

            X       = np.array(all_volumes, dtype=np.float32)
            y_mask  = np.array(all_masks,   dtype=np.float32)
            y_label = np.array(all_labels,  dtype=np.int32)

            logger.info(f"✅ Total samples     : {len(X)}")
            logger.info(f"✅ Volume shape      : {X.shape}")
            logger.info(f"✅ Mask shape        : {y_mask.shape}")
            logger.info(f"✅ Labels shape      : {y_label.shape}")

            return X, y_mask, y_label

        except Exception as e:
            raise NeuroScanException(e, sys)


    def split_and_save(
        self,
        X: np.ndarray,
        y_mask: np.ndarray,
        y_label: np.ndarray
    ) -> None:
        """
        Splits data into Train/Val/Test
        and saves as numpy arrays.
        """
        try:
            # Train / Temp split
            X_train, X_temp, ym_train, ym_temp, yl_train, yl_temp = train_test_split(
                X, y_mask, y_label,
                test_size=1 - self.config.train_ratio,
                random_state=self.config.random_seed,
                stratify=y_label
            )

            # Val / Test split
            val_ratio = self.config.val_ratio / (self.config.val_ratio + self.config.test_ratio)
            X_val, X_test, ym_val, ym_test, yl_val, yl_test = train_test_split(
                X_temp, ym_temp, yl_temp,
                test_size=1 - val_ratio,
                random_state=self.config.random_seed,
                stratify=yl_temp
            )

            # Save splits
            split_dir = Path(self.config.split_dir)
            os.makedirs(split_dir, exist_ok=True)

            np.save(split_dir / "X_train.npy",  X_train)
            np.save(split_dir / "X_val.npy",    X_val)
            np.save(split_dir / "X_test.npy",   X_test)
            np.save(split_dir / "ym_train.npy", ym_train)
            np.save(split_dir / "ym_val.npy",   ym_val)
            np.save(split_dir / "ym_test.npy",  ym_test)
            np.save(split_dir / "yl_train.npy", yl_train)
            np.save(split_dir / "yl_val.npy",   yl_val)
            np.save(split_dir / "yl_test.npy",  yl_test)

            logger.info(f"✅ Train samples : {len(X_train)}")
            logger.info(f"✅ Val samples   : {len(X_val)}")
            logger.info(f"✅ Test samples  : {len(X_test)}")
            logger.info(f"💾 Splits saved to: {split_dir}")

        except Exception as e:
            raise NeuroScanException(e, sys)


    def initiate_data_transformation(self, patient_dirs: List[Path]) -> None:
        """
        Main method — runs full data transformation pipeline.
        """
        try:
            logger.info("🚀 Data Transformation started")

            # Process all patients
            X, y_mask, y_label = self.process_all_patients(patient_dirs)

            # Split and save
            self.split_and_save(X, y_mask, y_label)

            logger.info("🎉 Data Transformation completed!")

        except Exception as e:
            raise NeuroScanException(e, sys)