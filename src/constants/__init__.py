# src/constants/__init__.py

from pathlib import Path

# ── Project Root ─────────────────────────────────────
ROOT_DIR = Path(".")

# ── Config File Paths ────────────────────────────────
CONFIG_FILE_PATH = ROOT_DIR / "config" / "config.yaml"
PARAMS_FILE_PATH = ROOT_DIR / "params.yaml"

# ── Artifacts Directory ──────────────────────────────
ARTIFACTS_DIR = ROOT_DIR / "artifacts"

# ── Data Directories ─────────────────────────────────
DATA_DIR          = ARTIFACTS_DIR / "data"
RAW_DATA_DIR      = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
SPLIT_DATA_DIR    = DATA_DIR / "split"

# ── Model Directory ──────────────────────────────────
MODELS_DIR = ARTIFACTS_DIR / "models"

# ── Model Names ──────────────────────────────────────
CNN_MODEL_NAME  = "cnn_model.h5"
UNET_MODEL_NAME = "unet_model.h5"
YOLO_MODEL_NAME = "yolo_model.pt"

# ── Model Save Paths ─────────────────────────────────
CNN_MODEL_PATH  = MODELS_DIR / CNN_MODEL_NAME
UNET_MODEL_PATH = MODELS_DIR / UNET_MODEL_NAME
YOLO_MODEL_PATH = MODELS_DIR / YOLO_MODEL_NAME

# ── BraTS 2021 Dataset ───────────────────────────────
IMAGE_SIZE      = (128, 128)
MRI_MODALITIES  = ["t1", "t1ce", "t2", "flair"]
NUM_CLASSES     = 4   # 0: background, 1: necrotic, 2: edema, 4: enhancing tumor
TUMOR_CLASSES   = {
    0: "Background",
    1: "Necrotic Tumor Core",
    2: "Peritumoral Edema",
    4: "Enhancing Tumor"
}

# ── Training Constants ───────────────────────────────
RANDOM_SEED     = 42
TRAIN_RATIO     = 0.70
VAL_RATIO       = 0.15
TEST_RATIO      = 0.15
BATCH_SIZE      = 8
EPOCHS          = 50

# ── Flask App ────────────────────────────────────────
APP_HOST        = "0.0.0.0"
APP_PORT        = 8000
DEBUG_MODE      = False

# ── Logs ─────────────────────────────────────────────
LOGS_DIR        = ROOT_DIR / "logs"