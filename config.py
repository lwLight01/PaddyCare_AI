import os
import math

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
TRAIN_DIR   = os.path.join(DATASET_DIR, "train")
VAL_DIR     = os.path.join(DATASET_DIR, "val")
TEST_DIR    = os.path.join(DATASET_DIR, "test")
MODEL_DIR   = os.path.join(BASE_DIR, "models")
MODEL_PATH  = os.path.join(MODEL_DIR, "model.pth")  
RUNS_DIR    = os.path.join(MODEL_DIR, "runs")  
BACKUP_DIR  = os.path.join(MODEL_DIR, "backups")  
LOG_DIR     = os.path.join(BASE_DIR, "logs")

IMAGE_SIZE    = 456        
NORM_MEAN     = [0.485, 0.456, 0.406]
NORM_STD      = [0.229, 0.224, 0.225]
BATCH_SIZE    = 8
EPOCHS        = 20
LEARNING_RATE = 0.0001
NUM_WORKERS   = 2
PIN_MEMORY    = True
ACCUM_STEPS   = 4         
WARMUP_EPOCHS = 3          
BACKBONE_LR   = 1e-5      
HEAD_LR       = 1e-4       
WEIGHT_DECAY  = 1e-4         
SEED          = 42
EARLY_STOP_PATIENCE = 5     
LAST_CHECKPOINT_NAME = "last.pth" 

EMBEDDING_DIM          = 128 
MAHALANOBIS_PERCENTILE = 99   
CLASS_STATS_PATH       = os.path.join(MODEL_DIR, "class_stats.json")
MODEL_META_PATH        = os.path.join(MODEL_DIR, "model_meta.json")
DISEASES_JSON_PATH     = os.path.join(BASE_DIR, "diseases.json")

MIN_SAMPLES_PER_CLASS  = 30
MIN_TOTAL_SAMPLES      = 8 * EMBEDDING_DIM
DEFAULT_STATS_TRAIN_PER_CLASS = 600
DEFAULT_STATS_VAL_PER_CLASS   = 150
DEFAULT_STATS_TEST_PER_CLASS  = 50

CONFIDENCE_THRESHOLD  = 0.60  
ENTROPY_FACTOR        = 0.75  

CALIBRATION_TARGET_PRECISION = 0.97
CALIBRATION_MIN_THRESHOLD    = 0.50

def get_entropy_threshold(num_classes: int) -> float:
    """Scales dynamically with the number of classes (ENTROPY_FACTOR of max Shannon entropy)."""
    return round(ENTROPY_FACTOR * math.log(max(num_classes, 2)), 4)

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
