import os
import math

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
TRAIN_DIR   = os.path.join(DATASET_DIR, "train")
VAL_DIR     = os.path.join(DATASET_DIR, "val")
TEST_DIR    = os.path.join(DATASET_DIR, "test")
MODEL_DIR   = os.path.join(BASE_DIR, "models")
MODEL_PATH  = os.path.join(MODEL_DIR, "model.pth")
RUNS_DIR    = os.path.join(MODEL_DIR, "runs")      # one sub-folder per training run
BACKUP_DIR  = os.path.join(MODEL_DIR, "backups")   # timestamped copies of replaced checkpoints
LOG_DIR     = os.path.join(BASE_DIR, "logs")

# ── Model & Training hyperparameters ──
IMAGE_SIZE    = 456          # EfficientNet-B5 native resolution
# ImageNet normalisation. Exported to models/model_meta.json so Android reads the same values.
NORM_MEAN     = [0.485, 0.456, 0.406]
NORM_STD      = [0.229, 0.224, 0.225]
BATCH_SIZE    = 8
EPOCHS        = 20
LEARNING_RATE = 0.0001
NUM_WORKERS   = 2
PIN_MEMORY    = True
ACCUM_STEPS   = 4            # Effective batch = BATCH_SIZE * ACCUM_STEPS = 32
WARMUP_EPOCHS = 10           # Phase 1: freeze backbone, train classifier head

# ── Embedding / Mahalanobis OOD detection ──
EMBEDDING_DIM          = 128   # Penultimate Linear(512, 128) feature dim
MAHALANOBIS_PERCENTILE = 99    # Threshold = this percentile of held-out (val) distances
CLASS_STATS_PATH       = os.path.join(MODEL_DIR, "class_stats.json")
MODEL_META_PATH        = os.path.join(MODEL_DIR, "model_meta.json")
# Tracked in git (models/ is ignored). Source of truth for disease info shown in the app.
DISEASES_JSON_PATH     = os.path.join(BASE_DIR, "diseases.json")

# A covariance for EMBEDDING_DIM features needs far more samples than dimensions,
# otherwise it is badly conditioned and every unseen image looks "out of distribution".
MIN_SAMPLES_PER_CLASS  = 30
MIN_TOTAL_SAMPLES      = 8 * EMBEDDING_DIM
DEFAULT_STATS_TRAIN_PER_CLASS = 600
DEFAULT_STATS_VAL_PER_CLASS   = 150
DEFAULT_STATS_TEST_PER_CLASS  = 50

# ── Thresholds (exported to the app via model_meta.json) ──
CONFIDENCE_THRESHOLD  = 0.60   # Flag low confidence if below 60%
ENTROPY_FACTOR        = 0.75   # entropy threshold = factor * ln(num_classes)

def get_entropy_threshold(num_classes: int) -> float:
    """Scales dynamically with the number of classes (ENTROPY_FACTOR of max Shannon entropy)."""
    return round(ENTROPY_FACTOR * math.log(max(num_classes, 2)), 4)

TRAIN_LOG_FILE    = os.path.join(LOG_DIR, "train.log")
EVALUATE_LOG_FILE = os.path.join(LOG_DIR, "evaluate.log")

VALID_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')

def get_active_classes(directory: str = TRAIN_DIR) -> list:
    """Returns sorted list of class subdirectories that actually contain valid image files."""
    if not os.path.isdir(directory):
        return []
    classes = []
    for entry in os.scandir(directory):
        if entry.is_dir():
            has_images = any(
                f.name.lower().endswith(VALID_EXTENSIONS)
                for f in os.scandir(entry.path)
                if f.is_file()
            )
            if has_images:
                classes.append(entry.name)
    classes.sort()
    return classes

# Default/fallback classes (dynamically discovered from train folder when available)
CLASS_NAMES = get_active_classes(TRAIN_DIR)
if not CLASS_NAMES:
    CLASS_NAMES = ["Bacterial Blight", "Healthy"]
NUM_CLASSES = len(CLASS_NAMES)
