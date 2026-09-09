import os
import math

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
TRAIN_DIR   = os.path.join(DATASET_DIR, "train")
VAL_DIR     = os.path.join(DATASET_DIR, "val")
TEST_DIR    = os.path.join(DATASET_DIR, "test")
MODEL_DIR   = os.path.join(BASE_DIR, "models")
MODEL_PATH  = os.path.join(MODEL_DIR, "model.pth")
UPLOAD_DIR  = os.path.join(BASE_DIR, "static", "uploads")
LOG_DIR     = os.path.join(BASE_DIR, "logs")

IMAGE_SIZE    = 456          # EfficientNet-B5 native resolution
BATCH_SIZE    = 8
EPOCHS        = 20
LEARNING_RATE = 0.0001
NUM_WORKERS   = 2
PIN_MEMORY    = True
ACCUM_STEPS   = 4        # Gradient accumulation — effective batch = BATCH_SIZE * ACCUM_STEPS = 32
WARMUP_EPOCHS = 10       # Phase-1: train classifier head only; Phase-2: fine-tune whole network

CLASS_NAMES = [
    "Bacterial Blight",
    "Healthy",
    # ── add new disease names here (alphabetical order!) ──
]
NUM_CLASSES = len(CLASS_NAMES)

# ── Embedding / Mahalanobis OOD detection ──
EMBEDDING_DIM         = 128   # matches the penultimate Linear(512, 128) layer
MAHALANOBIS_PERCENTILE = 99   # auto-calibrate threshold at 99th percentile of training distances
CLASS_STATS_PATH      = os.path.join(MODEL_DIR, "class_stats.json")

CONFIDENCE_THRESHOLD  = 0.60   # only show predictions with >= 60 % confidence
# OOD threshold scales automatically with the number of classes.
# Max Shannon entropy for N classes = ln(N). We flag as OOD when the
# model is more than 75 % of the way to maximum uncertainty.
# e.g.  2 classes → 0.52   |  8 classes → 1.56   |  12 classes → 1.91
OOD_ENTROPY_THRESHOLD = round(0.75 * math.log(max(NUM_CLASSES, 2)), 4)
TOP_K                 = 3     # return up to 3 predictions (useful for 8+ classes)

# Name of the reject class — update when a 'Not Leaf' class is added back
# NOT_LEAF_CLASS = "Not Leaf"

API_LOG_FILE      = os.path.join(LOG_DIR, "api.log")
TRAIN_LOG_FILE    = os.path.join(LOG_DIR, "train.log")
EVALUATE_LOG_FILE = os.path.join(LOG_DIR, "evaluate.log")

TREATMENTS = {
    # Note: Only classes listed in CLASS_NAMES above should have entries here.
    # When you add Blast to CLASS_NAMES, add its treatment back here too.
    "Bacterial Blight": {
        "en": (
            "1. Spray Copper Oxychloride or Streptomycin solution.\n"
            "2. Drain excess water from the field immediately.\n"
            "3. Avoid high nitrogen fertilizer application.\n"
            "4. Use resistant paddy varieties (e.g., IR64).\n"
            "5. Remove and destroy infected plants."
        ),
        "bn": (
            "১. কপার অক্সিক্লোরাইড বা স্ট্রেপটোমাইসিন দ্রবণ স্প্রে করুন।\n"
            "২. মাঠ থেকে অতিরিক্ত পানি দ্রুত নিষ্কাশন করুন।\n"
            "৩. অতিরিক্ত নাইট্রোজেন সার এড়িয়ে চলুন।\n"
            "৪. প্রতিরোধী ধানের জাত (যেমন IR64) ব্যবহার করুন।\n"
            "৫. আক্রান্ত গাছ তুলে নষ্ট করুন।"
        ),
    },
    "Healthy": {
        "en": (
            "Your paddy plant is healthy!\n"
            "1. Continue regular irrigation and fertilization.\n"
            "2. Monitor weekly for any early disease signs.\n"
            "3. Maintain proper field hygiene.\n"
            "4. Apply preventive fungicide if neighbors have infections."
        ),
        "bn": (
            "আপনার ধান গাছ সুস্থ আছে!\n"
            "১. নিয়মিত সেচ ও সার দেওয়া অব্যাহত রাখুন।\n"
            "২. প্রতি সপ্তাহে রোগের প্রাথমিক লক্ষণ পর্যবেক্ষণ করুন।\n"
            "৩. মাঠ পরিষ্কার-পরিচ্ছন্ন রাখুন।\n"
            "৪. আশেপাশে রোগ থাকলে প্রতিরোধমূলক ছত্রাকনাশক দিন।"
        ),
    },
    # "Not Leaf" treatment removed — add back when the class is re-introduced
}

