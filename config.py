import os
import math

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
TRAIN_DIR   = os.path.join(DATASET_DIR, "train")
VAL_DIR     = os.path.join(DATASET_DIR, "val")
TEST_DIR    = os.path.join(DATASET_DIR, "test")
MODEL_DIR   = os.path.join(BASE_DIR, "models")
MODEL_PATH  = os.path.join(MODEL_DIR, "model.pth")          # weights only (small, used for export)
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
WARMUP_EPOCHS = 3            # Phase 1: freeze backbone, train classifier head (head converges in 2-3 epochs)
BACKBONE_LR   = 1e-5         # Phase 2 learning rate for the pretrained backbone
HEAD_LR       = 1e-4         # Phase 2 learning rate for the classifier head
WEIGHT_DECAY  = 1e-4         # Applied to weights only (never to biases / norm parameters)
SEED          = 42
EARLY_STOP_PATIENCE = 5      # Phase 2 epochs without val Macro-F1 improvement before stopping (0 = off)
LAST_CHECKPOINT_NAME = "last.pth"   # per-run resume checkpoint (model + optimizer + scheduler)

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
# These are only FALLBACKS. extract_embeddings.py calibrates a softmax temperature and the
# confidence threshold on held-out val data; export_tflite.py prefers the calibrated values.
CONFIDENCE_THRESHOLD  = 0.60   # Flag low confidence if below 60%
ENTROPY_FACTOR        = 0.75   # entropy threshold = factor * ln(num_classes)
# Calibration target: the lowest confidence threshold whose ACCEPTED val predictions reach
# this precision. Raising it rejects more images as "uncertain".
CALIBRATION_TARGET_PRECISION = 0.97
CALIBRATION_MIN_THRESHOLD    = 0.50

def get_entropy_threshold(num_classes: int) -> float:
    """Scales dynamically with the number of classes (ENTROPY_FACTOR of max Shannon entropy)."""
    return round(ENTROPY_FACTOR * math.log(max(num_classes, 2)), 4)

# ── On-device "is this a plant photo?" colour/texture pre-check (exported to the app) ──
# Android HSV hue is in degrees (0-360). Hue 14-106 deg covers brown/yellow/green plant tissue,
# 34-106 deg is the green part. These are heuristics - tune them on a labelled leaf / non-leaf set.
LEAF_CHECK = {
    "min_saturation": 0.12,
    "min_value": 0.16,
    "plant_hue_min": 14.0,
    "plant_hue_max": 106.0,
    "green_hue_min": 34.0,
    "green_hue_max": 106.0,
    "min_plant_ratio": 0.06,
    "min_green_ratio": 0.03,
    "min_edge_ratio": 0.02,
    "max_edge_ratio": 0.65,
}

TRAIN_LOG_FILE    = os.path.join(LOG_DIR, "train.log")
EVALUATE_LOG_FILE = os.path.join(LOG_DIR, "evaluate.log")

VALID_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')

def get_active_classes(directory: str = TRAIN_DIR) -> list:
    """
    Returns the sorted list of class sub-directories that actually contain valid image files.
    This is the ONLY class-discovery helper; it is called lazily (never at import time) so that
    importing config / model never scans the disk. Trained checkpoints store their own
    `class_names`, which is the source of truth after training.
    """
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
