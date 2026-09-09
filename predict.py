import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import os
import json
import math
import numpy as np

from config import (
    MODEL_PATH, IMAGE_SIZE, CLASS_NAMES, CLASS_STATS_PATH,
    CONFIDENCE_THRESHOLD, OOD_ENTROPY_THRESHOLD, TOP_K, TREATMENTS,
    NOT_LEAF_CLASS,
)
from model import load_model, extract_embedding

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_model = None
_class_stats = None


# ─────────────────────────────────────────────────────────────────────────────
# Paddy-leaf image validator
# ─────────────────────────────────────────────────────────────────────────────

def check_paddy_image(img: Image.Image) -> tuple[bool, dict]:
    """
    Two-stage heuristic to decide if an image is a paddy (rice) leaf.

    Stage 1 – HSV colour analysis (100×100 thumbnail)
    ---------------------------------------------------
    PIL HSV encodes hue as 0–255 (NOT 0–360°).
      brown/yellow/green  hue_deg  14–106  →  PIL hue  10–75
      green specifically  hue_deg  34–106  →  PIL hue  24–75

    Thresholds (tightened vs. previous version):
      plant_ratio  ≥ 0.12   (≥12 % of pixels are plant-coloured)
      green_ratio  ≥ 0.05   (≥5 % of pixels are green)

    Stage 2 – Edge / texture density (grayscale Sobel-like via numpy)
    -------------------------------------------------------------------
    Real leaf images have moderate edge content (leaf veins, borders).
    Very smooth images (plain backgrounds, skies, skin) or very noisy
    images (random textures) are filtered here.
      edge_ratio  between 0.03 and 0.70
    """
    small = img.resize((100, 100)).convert("HSV")
    pixels = small.getdata()

    plant = 0
    green = 0

    for h, s, v in pixels:
        # Lowered saturation threshold: close-up / backlit leaves appear washed-out
        if s < 30 or v < 40:
            continue
        if 10 <= h <= 75:
            plant += 1
        if 24 <= h <= 75:
            green += 1

    total       = len(pixels)
    plant_ratio = plant / total
    green_ratio = green / total

    # Stage 1 gate – relaxed for thin/close-up leaves on bright backgrounds
    color_ok = (plant_ratio >= 0.06) and (green_ratio >= 0.03)

    # Stage 2 – simple edge density on a 64×64 grayscale thumbnail
    gray   = np.array(img.resize((64, 64)).convert("L"), dtype=np.float32)
    gx     = np.abs(np.diff(gray, axis=1))   # horizontal gradient
    gy     = np.abs(np.diff(gray, axis=0))   # vertical gradient
    edge_mean = (gx.mean() + gy.mean()) / 2.0
    # normalise: pixel values 0-255 → ratio relative to 255
    edge_ratio = edge_mean / 255.0
    texture_ok = 0.02 <= edge_ratio <= 0.65

    is_plant = color_ok and texture_ok

    return is_plant, {
        "plant_ratio": round(plant_ratio, 4),
        "green_ratio": round(green_ratio, 4),
        "edge_ratio":  round(float(edge_ratio),  4),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Model & stats loaders
# ─────────────────────────────────────────────────────────────────────────────

def get_model():
    global _model
    if _model is None:
        _model = load_model(MODEL_PATH, device)
    return _model


def get_class_stats() -> dict | None:
    """Load Mahalanobis class statistics from class_stats.json (if available)."""
    global _class_stats
    if _class_stats is None:
        if not os.path.exists(CLASS_STATS_PATH):
            print(f"[predict] Warning: {CLASS_STATS_PATH} not found — Mahalanobis check disabled.")
            return None
        with open(CLASS_STATS_PATH, "r", encoding="utf-8") as f:
            raw = json.load(f)
        # Convert lists back to numpy arrays for efficient computation
        _class_stats = {
            "class_means": {
                name: np.array(vec, dtype=np.float32)
                for name, vec in raw["class_means"].items()
            },
            "precision_matrix": np.array(raw["precision_matrix"], dtype=np.float32),
            "threshold": raw["threshold"],
        }
    return _class_stats


def mahalanobis_distance(embedding: np.ndarray, class_mean: np.ndarray,
                         precision: np.ndarray) -> float:
    """Compute Mahalanobis distance: sqrt((x-μ)ᵀ Σ⁻¹ (x-μ))"""
    diff = embedding - class_mean
    return float(np.sqrt(np.dot(diff, precision @ diff)))


from dataset_utils import PadToSquare

infer_transform = transforms.Compose([
    PadToSquare(),
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


# ─────────────────────────────────────────────────────────────────────────────
# Main prediction function — 5-Layer OOD rejection
# ─────────────────────────────────────────────────────────────────────────────

def predict(image_path: str, lang: str = "bn") -> dict:
    if not os.path.exists(image_path):
        return {"status": "error", "message": "Image file not found."}

    try:
        img = Image.open(image_path).convert("RGB")
    except Exception as e:
        return {"status": "error", "message": f"Cannot open image: {e}"}

    # ── Layer 1: Paddy-leaf visual gate (HSV colour + edge texture) ─────────
    is_plant, color_stats = check_paddy_image(img)
    if not is_plant:
        return {
            "status": "not_paddy",
            "message": (
                "🚫 এই ছবিটি ধান পাতার নয়।\n"
                "PaddyCare শুধুমাত্র ধান (rice) পাতার রোগ শনাক্ত করতে পারে।\n"
                "অনুগ্রহ করে একটি স্পষ্ট ধান পাতার ছবি আপলোড করুন।\n\n"
                "🚫 This is not a paddy (rice) leaf image.\n"
                "PaddyCare can only analyse paddy leaf photos.\n"
                "Please upload a clear, close-up photo of a paddy leaf."
            ),
            "predictions": [],
            "debug": color_stats,
        }

    # ── Layer 2: Model inference + "Not Leaf" class check ───────────────────
    tensor = infer_transform(img).unsqueeze(0).to(device)

    model = get_model()
    logits, embedding = extract_embedding(model, tensor)
    probs = F.softmax(logits, dim=1)[0]

    max_conf = probs.max().item()
    top_class_idx = probs.argmax().item()
    top_class_name = CLASS_NAMES[top_class_idx]

    # If the model itself predicts "Not Leaf" → reject
    if top_class_name == NOT_LEAF_CLASS:
        return {
            "status": "not_paddy",
            "message": (
                "🚫 এই ছবিটি ধান পাতার নয়।\n"
                "PaddyCare শুধুমাত্র ধান (rice) পাতার রোগ শনাক্ত করতে পারে।\n"
                "অনুগ্রহ করে একটি স্পষ্ট ধান পাতার ছবি আপলোড করুন।\n\n"
                "🚫 This is not a paddy (rice) leaf image.\n"
                "PaddyCare can only analyse paddy leaf photos.\n"
                "Please upload a clear, close-up photo of a paddy leaf."
            ),
            "predictions": [],
            "debug": {
                **color_stats,
                "max_conf": round(max_conf, 4),
                "predicted_class": top_class_name,
            },
        }

    # ── Layer 3: Mahalanobis distance check ─────────────────────────────────
    stats = get_class_stats()
    if stats is not None:
        emb_np = embedding[0].cpu().numpy()
        class_mean = stats["class_means"].get(top_class_name)

        if class_mean is not None:
            m_dist = mahalanobis_distance(emb_np, class_mean, stats["precision_matrix"])

            if m_dist > stats["threshold"]:
                return {
                    "status": "not_paddy",
                    "message": (
                        "🚫 এই ছবিটি ধান পাতার মতো দেখালেও, মডেল এটিকে পরিচিত ধান পাতা হিসেবে "
                        "চিনতে পারছে না।\n"
                        "অনুগ্রহ করে একটি স্পষ্ট ধান পাতার ছবি আপলোড করুন।\n\n"
                        "🚫 Although this image looks plant-like, the model does not recognise it "
                        "as a known paddy leaf.\n"
                        "Please upload a clear, close-up photo of a paddy leaf."
                    ),
                    "predictions": [],
                    "debug": {
                        **color_stats,
                        "max_conf": round(max_conf, 4),
                        "mahalanobis_dist": round(m_dist, 4),
                        "threshold": stats["threshold"],
                    },
                }

    # ── Layer 4: Confidence gate ────────────────────────────────────────────
    # Shannon entropy (nats).  Max for N classes = ln(N).
    entropy = -sum(p * math.log(p + 1e-9) for p in probs.tolist())

    if (max_conf < CONFIDENCE_THRESHOLD) or (entropy > OOD_ENTROPY_THRESHOLD):
        return {
            "status": "low_confidence",
            "message": (
                "⚠️ ছবিটি অস্পষ্ট বা মডেল আত্মবিশ্বাসের সাথে শনাক্ত করতে পারছে না।\n"
                "সঠিক ফলাফলের জন্য ধান পাতার আরও পরিষ্কার ও কাছের ছবি আপলোড করুন।\n\n"
                "⚠️ The image is unclear or the model cannot identify it confidently.\n"
                "Please upload a sharper, closer photo of the paddy leaf."
            ),
            "predictions": [],
            "debug": {
                **color_stats,
                "max_conf": round(max_conf, 4),
                "entropy":  round(entropy,  4),
            },
        }

    # ── Layer 5: Build top-K results (filter "Not Leaf" from output) ────────
    # Top-1 already cleared the gates above.
    # Secondary predictions are included if they are ≥ 10 % — enough to be
    # meaningful without being noise.
    SECONDARY_THRESHOLD = 10.0
    top_probs, top_indices = torch.topk(probs, k=min(TOP_K, len(CLASS_NAMES)))
    predictions = []

    for rank, (prob, idx) in enumerate(zip(top_probs.tolist(), top_indices.tolist())):
        confidence_pct = round(prob * 100, 2)
        disease_name = CLASS_NAMES[idx]

        # Never show "Not Leaf" as a result to the user
        if disease_name == NOT_LEAF_CLASS:
            continue

        # Top-1: already validated by Layer 4 gate, always include
        # Others: must clear the secondary floor
        if rank > 0 and confidence_pct < SECONDARY_THRESHOLD:
            break  # topk is sorted descending, so no point continuing

        treatment = TREATMENTS.get(disease_name, {}).get(lang, "")
        predictions.append({
            "disease":    disease_name,
            "confidence": confidence_pct,
            "treatment":  treatment,
        })

    # Edge-case: all top predictions were "Not Leaf"
    if not predictions:
        return {
            "status": "low_confidence",
            "message": (
                "⚠️ ছবিটি অস্পষ্ট বা মডেল আত্মবিশ্বাসের সাথে শনাক্ত করতে পারছে না।\n"
                "সঠিক ফলাফলের জন্য ধান পাতার আরও পরিষ্কার ও কাছের ছবি আপলোড করুন।\n\n"
                "⚠️ The image is unclear or the model cannot identify it confidently.\n"
                "Please upload a sharper, closer photo of the paddy leaf."
            ),
            "predictions": [],
            "debug": {
                **color_stats,
                "max_conf": round(max_conf, 4),
                "entropy":  round(entropy,  4),
            },
        }

    return {"status": "ok", "predictions": predictions}


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python predict.py <image_path>")
        sys.exit(1)

    result = predict(sys.argv[1], lang="bn")
    print(json.dumps(result, ensure_ascii=False, indent=2))
