"""
extract_embeddings.py
=====================
Computes per-class embedding statistics (mean vectors + shared precision matrix)
from the trained model for Mahalanobis-based OOD detection.

Run this AFTER training and BEFORE export:
    python extract_embeddings.py

Outputs:
    models/class_stats.json
"""

import os
import json
import logging
import numpy as np
import torch
from torch.utils.data import DataLoader

from config import (
    MODEL_PATH, MODEL_DIR, LOG_DIR, CLASS_STATS_PATH,
    BATCH_SIZE, NUM_WORKERS, EMBEDDING_DIM, MAHALANOBIS_PERCENTILE,
    TRAIN_DIR, IMAGE_SIZE,
)
from model import build_model, extract_embedding
from dataset_utils import val_transform

import torchvision.datasets as datasets

os.makedirs(LOG_DIR, exist_ok=True)
logging.basicConfig(
    filename=os.path.join(LOG_DIR, "embeddings.log"),
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    encoding="utf-8",
)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
logging.getLogger().addHandler(console)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def extract_and_save():
    """
    1. Load trained model
    2. Pass every training image through it (eval mode, val_transform)
    3. Collect 128-dim embeddings per class
    4. Compute per-class mean + shared covariance → invert to precision matrix
    5. Auto-calibrate Mahalanobis threshold at MAHALANOBIS_PERCENTILE
    6. Save everything to CLASS_STATS_PATH
    """
    logging.info("=" * 60)
    logging.info("Extracting embeddings for Mahalanobis OOD detection ...")
    logging.info(f"Device: {device}")

    # ── Load model ──
    if not os.path.exists(MODEL_PATH):
        logging.error(f"Model not found: {MODEL_PATH}. Train first.")
        return

    model = build_model(pretrained=False)
    checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    logging.info(f"Loaded model from epoch {checkpoint.get('epoch', '?')}")

    # ── Load training data with val_transform (no augmentation, deterministic) ──
    train_dataset = datasets.ImageFolder(root=TRAIN_DIR, transform=val_transform)
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )
    class_names = train_dataset.classes
    logging.info(f"Classes: {class_names}")
    logging.info(f"Training samples: {len(train_dataset)}")

    # ── Extract embeddings ──
    all_embeddings = []
    all_labels = []

    for batch_idx, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        _, embeddings = extract_embedding(model, images)
        all_embeddings.append(embeddings.cpu().numpy())
        all_labels.append(labels.numpy())

        if (batch_idx + 1) % 50 == 0:
            logging.info(f"  Processed {(batch_idx + 1) * BATCH_SIZE} / {len(train_dataset)} images")

    all_embeddings = np.concatenate(all_embeddings, axis=0)  # (N, 128)
    all_labels = np.concatenate(all_labels, axis=0)           # (N,)

    logging.info(f"Embeddings shape: {all_embeddings.shape}")

    # ── Compute per-class means ──
    class_means = {}
    for cls_idx, cls_name in enumerate(class_names):
        mask = all_labels == cls_idx
        cls_embeddings = all_embeddings[mask]
        class_means[cls_name] = cls_embeddings.mean(axis=0)
        logging.info(f"  {cls_name}: {cls_embeddings.shape[0]} samples")

    # ── Compute shared covariance matrix ──
    # Centre each embedding by subtracting its class mean
    centred = np.zeros_like(all_embeddings)
    for cls_idx, cls_name in enumerate(class_names):
        mask = all_labels == cls_idx
        centred[mask] = all_embeddings[mask] - class_means[cls_name]

    # Shared covariance = (1/N) * X_centred^T @ X_centred
    cov = np.cov(centred, rowvar=False)  # (128, 128)

    # Add small ridge for numerical stability
    cov += np.eye(EMBEDDING_DIM) * 1e-6

    # Precision matrix = inverse covariance
    precision = np.linalg.inv(cov)  # (128, 128)
    logging.info(f"Covariance matrix: {cov.shape}, condition number: {np.linalg.cond(cov):.2f}")

    # ── Compute Mahalanobis distances for all training samples ──
    # D(x, μ_k) = sqrt((x - μ_k)^T @ Σ^(-1) @ (x - μ_k))
    # We compute distance to the correct class for each sample
    distances = []
    for i in range(len(all_embeddings)):
        cls_idx = all_labels[i]
        cls_name = class_names[cls_idx]
        diff = all_embeddings[i] - class_means[cls_name]
        dist = np.sqrt(np.dot(diff, precision @ diff))
        distances.append(dist)

    distances = np.array(distances)

    # Auto-calibrate threshold at the specified percentile
    threshold = float(np.percentile(distances, MAHALANOBIS_PERCENTILE))
    logging.info(f"Mahalanobis distances — mean: {distances.mean():.2f}, "
                 f"std: {distances.std():.2f}, "
                 f"max: {distances.max():.2f}")
    logging.info(f"Threshold ({MAHALANOBIS_PERCENTILE}th percentile): {threshold:.4f}")

    # ── Save to JSON ──
    stats = {
        "class_names": class_names,
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
    logging.info("=" * 60)

    print(f"\n[embeddings] Class stats saved to: {CLASS_STATS_PATH}")
    print(f"[embeddings] Threshold: {threshold:.4f} ({MAHALANOBIS_PERCENTILE}th percentile)")
    print(f"[embeddings] Classes: {class_names}")

    return stats


if __name__ == "__main__":
    extract_and_save()
