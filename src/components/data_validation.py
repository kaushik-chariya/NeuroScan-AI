# src/components/data_validation.py

import os
import sys
from pathlib import Path
from logger import logger
from exception import NeuroScanException
from src.entity.config_entity import DataValidationConfig


class DataValidation:
    def __init__(self, config: DataValidationConfig):
        self.config = config


    def get_patient_dirs(self) -> list:
        """
        Returns list of all patient directories
        from raw data directory.
        """
        try:
            patient_dirs = sorted([
                d for d in Path(self.config.root_dir).iterdir()
                if d.is_dir()
            ])
            logger.info(f"✅ Total patients found: {len(patient_dirs)}")
            return patient_dirs

        except Exception as e:
            raise NeuroScanException(e, sys)


    def validate_modalities(self, patient_dir: Path) -> bool:
        """
        Checks if all required modality files
        exist for a given patient.
        """
        try:
            patient_id = patient_dir.name

            for modality in self.config.required_modalities:
                file_path = patient_dir / f"{patient_id}_{modality}.nii.gz"
                if not file_path.exists():
                    logger.warning(f"⚠️ Missing modality: {file_path}")
                    return False

            return True

        except Exception as e:
            raise NeuroScanException(e, sys)


    def validate_segmentation(self, patient_dir: Path) -> bool:
        """
        Checks if segmentation file exists
        for a given patient.
        """
        try:
            patient_id = patient_dir.name
            seg_file = patient_dir / f"{patient_id}_seg.nii.gz"

            if not seg_file.exists():
                logger.warning(f"⚠️ Missing segmentation: {seg_file}")
                return False

            return True

        except Exception as e:
            raise NeuroScanException(e, sys)


    def validate_file_count(self, patient_dir: Path) -> bool:
        """
        Checks if total file count matches
        required files per case.
        """
        try:
            files = list(patient_dir.glob("*.nii.gz"))
            if len(files) != self.config.required_files_per_case:
                logger.warning(
                    f"⚠️ {patient_dir.name} — "
                    f"Expected {self.config.required_files_per_case} files, "
                    f"Found {len(files)}"
                )
                return False

            return True

        except Exception as e:
            raise NeuroScanException(e, sys)


    def write_validation_status(self, status: bool) -> None:
        """
        Writes validation status to status file.
        """
        try:
            os.makedirs(
                os.path.dirname(self.config.status_file),
                exist_ok=True
            )
            with open(self.config.status_file, "w") as f:
                f.write(f"Validation Status: {status}")

            logger.info(f"📝 Validation status saved: {self.config.status_file}")

        except Exception as e:
            raise NeuroScanException(e, sys)


    def initiate_data_validation(self) -> bool:
        """
        Main method — runs full data validation.
        Returns True if all patients are valid.
        """
        try:
            logger.info("🚀 Data Validation started")

            patient_dirs = self.get_patient_dirs()

            valid_count   = 0
            invalid_count = 0
            invalid_cases = []

            for patient_dir in patient_dirs:
                modality_ok    = self.validate_modalities(patient_dir)
                seg_ok         = self.validate_segmentation(patient_dir)
                file_count_ok  = self.validate_file_count(patient_dir)

                if modality_ok and seg_ok and file_count_ok:
                    valid_count += 1
                else:
                    invalid_count += 1
                    invalid_cases.append(patient_dir.name)

            # Log summary
            logger.info(f"✅ Valid patients   : {valid_count}")
            logger.info(f"❌ Invalid patients : {invalid_count}")

            if invalid_cases:
                logger.warning(f"⚠️ Invalid cases: {invalid_cases}")

            # Write status
            validation_passed = invalid_count == 0
            self.write_validation_status(validation_passed)

            if validation_passed:
                logger.info("🎉 Data Validation passed!")
            else:
                logger.warning("⚠️ Data Validation failed — check invalid cases")

            return validation_passed

        except Exception as e:
            raise NeuroScanException(e, sys)