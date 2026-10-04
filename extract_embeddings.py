"""
extract_embeddings.py
=====================
Computes per-class embedding statistics (mean vectors + shared precision matrix)
from the trained model for Mahalanobis-based OOD detection.

Run this AFTER training and BEFORE export:
    python extract_embeddings.py

Outputs:
    models/class_stats.json
    PaddyCareAndroid/app/src/main/assets/class_stats.json (if Android project exists)
"""

import os
import json
import shutil
import logging
import numpy as np
import torch
from torch.utils.data import DataLoader

from config import (
    MODEL_PATH, MODEL_DIR, LOG_DIR, CLASS_STATS_PATH,
    BATCH_SIZE, NUM_WORKERS, EMBEDDING_DIM, MAHALANOBIS_PERCENTILE,
    TRAIN_DIR,
)
from model import load_model, extract_embedding
from dataset_utils import val_transform, SafeImageFolder

os.makedirs(LOG_DIR, exist_ok=True)
logging.basicConfig(
    filename=os.path.join(LOG_DIR, "embeddings.log"),
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    encoding="utf-8",
)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
if not any(isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler) for h in logging.getLogger().handlers):
    logging.getLogger().addHandler(console)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def extract_and_save(max_samples_per_class: int = 0):
    """
    1. Load trained model dynamically
    2. Pass training images through it (eval mode, val_transform)
    3. Collect 128-dim embeddings per class
    4. Compute per-class mean + shared covariance → invert to precision matrix
    5. Auto-calibrate Mahalanobis threshold at MAHALANOBIS_PERCENTILE
    6. Save everything to CLASS_STATS_PATH and copy to Android assets
    """
    logging.info("=" * 60)
    logging.info("Extracting embeddings for Mahalanobis OOD detection ...")
    logging.info(f"Device: {device}")

    # ── Load model ──
    if not os.path.exists(MODEL_PATH):
        logging.error(f"Model not found: {MODEL_PATH}. Train first.")
        return None

    model = load_model(MODEL_PATH, device)

    # ── Load training data with val_transform (no augmentation, deterministic) ──
    train_dataset = SafeImageFolder(root=TRAIN_DIR, transform=val_transform)
    if len(train_dataset) == 0:
        logging.error(f"No training samples found in {TRAIN_DIR}.")
        return None

    class_names = train_dataset.classes
    logging.info(f"Classes: {class_names}")

    if max_samples_per_class > 0:
        targets = np.array(train_dataset.targets)
        selected_indices = []
        for cls_idx in range(len(class_names)):
            cls_indices = np.where(targets == cls_idx)[0]
            selected_indices.extend(cls_indices[:max_samples_per_class])
        dataset_to_load = torch.utils.data.Subset(train_dataset, selected_indices)
        logging.info(f"Using {len(dataset_to_load)} samples ({max_samples_per_class}/class) for fast calibration.")
    else:
        dataset_to_load = train_dataset
        logging.info(f"Training samples: {len(train_dataset)}")

    train_loader = DataLoader(
        dataset_to_load,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0 if device.type == "cpu" else NUM_WORKERS,
    )

    # ── Extract embeddings ──
    all_embeddings = []
    all_labels = []

    for batch_idx, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        _, embeddings = extract_embedding(model, images)
        all_embeddings.append(embeddings.cpu().numpy())
        all_labels.append(labels.numpy())

        if (batch_idx + 1) % 10 == 0:
            logging.info(f"  Processed {min((batch_idx + 1) * BATCH_SIZE, len(dataset_to_load))} / {len(dataset_to_load)} samples...")

    all_embeddings = np.concatenate(all_embeddings, axis=0)  # (N, 128)
    all_labels = np.concatenate(all_labels, axis=0)           # (N,)

    logging.info(f"Embeddings shape: {all_embeddings.shape}")

    # ── Compute per-class means ──
    class_means = {}
    valid_classes = []
    for cls_idx, cls_name in enumerate(class_names):
        mask = all_labels == cls_idx
        cls_embeddings = all_embeddings[mask]
        if len(cls_embeddings) == 0:
            logging.warning(f"  Class '{cls_name}' has 0 samples; skipping OOD mean.")
            continue

        class_means[cls_name] = cls_embeddings.mean(axis=0)
        valid_classes.append(cls_name)
        logging.info(f"  {cls_name}: {cls_embeddings.shape[0]} samples")

    if not class_means:
        logging.error("No valid class means could be calculated.")
        return None

    # ── Compute shared covariance matrix ──
    centred = np.zeros_like(all_embeddings)
    for cls_idx, cls_name in enumerate(class_names):
        if cls_name not in class_means:
            continue
        mask = all_labels == cls_idx
        centred[mask] = all_embeddings[mask] - class_means[cls_name]

    # Shared covariance = (1/N) * X_centred^T @ X_centred
    cov = np.cov(centred, rowvar=False)  # (128, 128)

    # Add small ridge for numerical stability
    cov += np.eye(EMBEDDING_DIM) * 1e-6

    # Precision matrix = inverse covariance
    precision = np.linalg.inv(cov)
    logging.info(f"Covariance matrix: {cov.shape}, condition number: {np.linalg.cond(cov):.2f}")

    # ── Compute Mahalanobis distances for training samples ──
    distances = []
    for i in range(len(all_embeddings)):
        cls_idx = all_labels[i]
        cls_name = class_names[cls_idx]
        if cls_name not in class_means:
            continue
        diff = all_embeddings[i] - class_means[cls_name]
        dist = np.sqrt(np.dot(diff, precision @ diff))
        distances.append(dist)

    distances = np.array(distances)
    if len(distances) == 0:
        threshold = 10.0
    else:
        threshold = float(np.percentile(distances, MAHALANOBIS_PERCENTILE))

    logging.info(f"Mahalanobis distances — mean: {distances.mean():.2f}, "
                 f"std: {distances.std():.2f}, "
                 f"max: {distances.max():.2f}")
    logging.info(f"Threshold ({MAHALANOBIS_PERCENTILE}th percentile): {threshold:.4f}")

    # ── Save to JSON ──
    stats = {
        "class_names": valid_classes,
        "embedding_dim": EMBEDDING_DIM,
        "class_means": {name: mean.tolist() for name, mean in class_means.items()},
        "precision_matrix": precision.tolist(),
        "threshold": round(threshold, 4),
        "percentile": MAHALANOBIS_PERCENTILE,
        "num_train_samples": len(all_embeddings),
        "distance_stats": {
            "mean": round(float(distances.mean()), 4),
            "std": round(float(distances.std()), 4),
            "max": round(float(distances.max()), 4),
        },
    }

    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(CLASS_STATS_PATH, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    logging.info(f"Class stats saved → {CLASS_STATS_PATH}")

    # Also automatically copy to Android assets folder if it exists
    android_assets_dir = os.path.join(os.path.dirname(MODEL_DIR), "PaddyCareAndroid", "app", "src", "main", "assets")
    if os.path.isdir(android_assets_dir):
        android_stats_path = os.path.join(android_assets_dir, "class_stats.json")
        try:
            shutil.copy2(CLASS_STATS_PATH, android_stats_path)
            logging.info(f"Copied class stats to Android assets → {android_stats_path}")
        except Exception as e:
            logging.warning(f"Could not copy stats to Android assets: {e}")

    logging.info("=" * 60)
    print(f"\n[embeddings] Class stats saved to: {CLASS_STATS_PATH}")
    print(f"[embeddings] Threshold: {threshold:.4f} ({MAHALANOBIS_PERCENTILE}th percentile)")
    print(f"[embeddings] Classes: {valid_classes}")

    return stats


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Extract embeddings for Mahalanobis OOD detection")
    parser.add_argument("--max-samples", type=int, default=0, help="Max samples per class (0 for all, e.g. 100 for fast calibration)")
    args = parser.parse_args()
    extract_and_save(max_samples_per_class=args.max_samples)
