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
    "Blast",
    "Brown Spot",
    "Healthy",
    "Not Leaf"
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
NOT_LEAF_CLASS = "Not Leaf"

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
    "Blast": {
        "en": (
            "1. Spray Tricyclazole (0.1%) or Isoprothiolane fungicide.\n"
            "2. Apply silicon-based fertilizer to boost resistance.\n"
            "3. Avoid excess nitrogen fertilizer.\n"
            "4. Drain field water periodically.\n"
            "5. Use blast-resistant paddy varieties."
        ),
        "bn": (
            "১. ট্রাইসাইক্লাজোল (০.১%) বা আইসোপ্রোথিওলেন ছত্রাকনাশক স্প্রে করুন।\n"
            "২. সিলিকন-ভিত্তিক সার প্রয়োগ করুন।\n"
            "৩. অতিরিক্ত নাইট্রোজেন সার এড়িয়ে চলুন।\n"
            "৪. মাঠের পানি মাঝে মাঝে নিষ্কাশন করুন।\n"
            "৫. ব্লাস্ট-প্রতিরোধী ধানের জাত ব্যবহার করুন।"
        ),
    },
    "Brown Spot": {
        "en": (
            "1. Spray Mancozeb or Propiconazole fungicide.\n"
            "2. Apply balanced potassium and silicon fertilizer.\n"
            "3. Avoid water stress during the tillering stage.\n"
            "4. Use resistant varieties.\n"
            "5. Remove and burn infected plant debris."
        ),
        "bn": (
            "১. ম্যানকোজেব বা প্রোপিকোনাজোল ছত্রাকনাশক স্প্রে করুন।\n"
            "২. সুষম পটাশিয়াম ও সিলিকন সার প্রয়োগ করুন।\n"
            "৩. কুশি পর্যায়ে পানির চাপ এড়িয়ে চলুন।\n"
            "৪. প্রতিরোধী জাত ব্যবহার করুন।\n"
            "৫. আক্রান্ত গাছের অবশিষ্ট পুড়িয়ে ফেলুন।"
        ),
    },
    "Not Leaf": {
        "en": (
            "This image is not a paddy (rice) leaf.\n"
            "PaddyCare can only diagnose diseases from paddy leaf photos.\n"
            "Please upload a clear, close-up photo of a paddy leaf."
        ),
        "bn": (
            "এই ছবিটি ধান (চাল) পাতার নয়।\n"
            "PaddyCare শুধুমাত্র ধান পাতার ছবি থেকে রোগ নির্ণয় করতে পারে।\n"
            "অনুগ্রহ করে ধান পাতার একটি স্পষ্ট ও কাছের ছবি আপলোড করুন।"
        ),
    },
}

