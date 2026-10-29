import os
import json
import shutil
import logging
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.covariance import LedoitWolf

from config import (
    MODEL_PATH, MODEL_DIR, LOG_DIR, CLASS_STATS_PATH, TRAIN_DIR, VAL_DIR, TEST_DIR,
    BATCH_SIZE, EMBEDDING_DIM, MAHALANOBIS_PERCENTILE,
    MIN_SAMPLES_PER_CLASS, MIN_TOTAL_SAMPLES,
    DEFAULT_STATS_TRAIN_PER_CLASS, DEFAULT_STATS_VAL_PER_CLASS, DEFAULT_STATS_TEST_PER_CLASS,
)
from model import load_model, extract_embedding
from dataset_utils import val_transform, SafeImageFolder, subset_per_class
from log_utils import setup_logging

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MAX_TEST_FLAG_RATE = 0.10
MIN_VAL_SAMPLES = 50


def android_assets_dir() -> str:
    return os.path.join(os.path.dirname(MODEL_DIR), "PaddyCareAndroid", "app", "src", "main", "assets")


def remove_stale_stats(stats_path: str = CLASS_STATS_PATH) -> None:
    """Remove an unusable stats file. The Android copy is only removed for the production stats path."""
    paths = [stats_path]
    if os.path.abspath(stats_path) == os.path.abspath(CLASS_STATS_PATH):
        paths.append(os.path.join(android_assets_dir(), "class_stats.json"))
    for path in paths:
        if os.path.exists(path):
            try:
                os.remove(path)
                logging.warning(f"Removed stale/unusable stats file: {path}")
            except OSError as e:
                logging.warning(f"Could not remove {path}: {e}")


def _load_split(root: str, class_names: list, per_class: int, seed: int):
    if not os.path.isdir(root):
        return None
    ds = SafeImageFolder(root=root, transform=val_transform, classes=class_names)
    if len(ds) == 0:
        return None
    if per_class > 0:
        ds, _ = subset_per_class(ds, per_class, seed=seed)
    return ds


def _embed(model, dataset, tag: str):
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    embs, labels, preds = [], [], []
    n = len(dataset)
    for batch_idx, (images, y) in enumerate(loader):
        logits, emb = extract_embedding(model, images.to(device))
        embs.append(emb.cpu().numpy())
        preds.append(logits.argmax(1).cpu().numpy())
        labels.append(y.numpy())
        if (batch_idx + 1) % 10 == 0:
            logging.info(f"  [{tag}] {min((batch_idx + 1) * BATCH_SIZE, n)} / {n}")
    return np.concatenate(embs), np.concatenate(labels), np.concatenate(preds)


def _distances(emb, preds, class_means, precision, class_names):
    out = np.full(len(emb), np.nan)
    for idx, name in enumerate(class_names):
        mean = class_means.get(name)
        mask = preds == idx
        if mean is None or not mask.any():
            continue
        diff = emb[mask] - mean
        sq = np.einsum("ij,jk,ik->i", diff, precision, diff)
        out[mask] = np.sqrt(np.maximum(sq, 0.0))
    return out


def extract_and_save(
    train_per_class: int = DEFAULT_STATS_TRAIN_PER_CLASS,
    val_per_class: int = DEFAULT_STATS_VAL_PER_CLASS,
    test_per_class: int = DEFAULT_STATS_TEST_PER_CLASS,
    model_path: str = MODEL_PATH,
    output_path: str = CLASS_STATS_PATH,
):

    logging.info("=" * 60)
    logging.info("Extracting embeddings for Mahalanobis OOD detection ...")
    logging.info(f"Device: {device}")

    if not os.path.exists(model_path):
        logging.error(f"Model not found: {model_path}. Train first.")
        return None

    model = load_model(model_path, device)
    class_names = model.class_names
    if not class_names:
        logging.error("Checkpoint has no class_names; cannot map embeddings to classes.")
        return None
    logging.info(f"Classes (from checkpoint): {class_names}")

    train_ds = _load_split(TRAIN_DIR, class_names, train_per_class, seed=1)
    val_ds   = _load_split(VAL_DIR,   class_names, val_per_class,   seed=2)
    test_ds  = _load_split(TEST_DIR,  class_names, test_per_class,  seed=3)
    if train_ds is None:
        logging.error(f"No training samples found in {TRAIN_DIR}.")
        return None
    if val_ds is None or len(val_ds) < MIN_VAL_SAMPLES:
        logging.error(
            f"Need at least {MIN_VAL_SAMPLES} validation images in {VAL_DIR} to calibrate the "
            f"OOD threshold on held-out data (found {0 if val_ds is None else len(val_ds)}). "
            "Stats NOT written."
        )
        remove_stale_stats(output_path)
        return None

    emb, labels, _ = _embed(model, train_ds, "train")
    logging.info(f"Train embeddings: {emb.shape}")

    class_means, kept = {}, []
    for idx, name in enumerate(class_names):
        cls_emb = emb[labels == idx]
        if len(cls_emb) < MIN_SAMPLES_PER_CLASS:
            logging.warning(f"  '{name}': {len(cls_emb)} samples (< {MIN_SAMPLES_PER_CLASS}); "
                            "excluded from OOD stats (the app skips the check for it).")
            continue
        class_means[name] = cls_emb.mean(axis=0)
        kept.append(name)
        logging.info(f"  '{name}': {len(cls_emb)} samples")

    used = np.isin(labels, [class_names.index(n) for n in kept])
    if not kept or used.sum() < MIN_TOTAL_SAMPLES:
        logging.error(
            f"Only {int(used.sum())} usable samples for a {EMBEDDING_DIM}-dim covariance "
            f"(need >= {MIN_TOTAL_SAMPLES}). Stats NOT written - use a larger --train-per-class."
        )
        remove_stale_stats(output_path)
        return None

    mean_table = np.stack([class_means.get(n, np.zeros(emb.shape[1])) for n in class_names])
    centred = emb[used] - mean_table[labels[used]]
    lw = LedoitWolf(assume_centered=True).fit(centred)
    precision = lw.precision_
    logging.info(f"Ledoit-Wolf shrinkage = {lw.shrinkage_:.4f} | "
                 f"cov condition number = {np.linalg.cond(lw.covariance_):.1f}")

    v_emb, _, v_pred = _embed(model, val_ds, "val")
    v_dist = _distances(v_emb, v_pred, class_means, precision, class_names)
    v_dist = v_dist[~np.isnan(v_dist)]
    if len(v_dist) < MIN_VAL_SAMPLES:
        logging.error("Too few validation images predicted as a class with OOD stats. Stats NOT written.")
        remove_stale_stats(output_path)
        return None
    threshold = float(np.percentile(v_dist, MAHALANOBIS_PERCENTILE))
    logging.info(f"Val distances: mean {v_dist.mean():.2f}, std {v_dist.std():.2f}, max {v_dist.max():.2f}")
    logging.info(f"Threshold ({MAHALANOBIS_PERCENTILE}th percentile of val): {threshold:.4f}")

    test_flag_rate = None
    if test_ds is not None and len(test_ds) > 0:
        t_emb, _, t_pred = _embed(model, test_ds, "test")
        t_dist = _distances(t_emb, t_pred, class_means, precision, class_names)
        t_dist = t_dist[~np.isnan(t_dist)]
        if len(t_dist):
            test_flag_rate = float((t_dist > threshold).mean())
            logging.info(f"Held-out TEST images flagged as OOD: {test_flag_rate * 100:.1f}% "
                         f"({int((t_dist > threshold).sum())}/{len(t_dist)}) - expected ~{100 - MAHALANOBIS_PERCENTILE}%")
            if test_flag_rate > MAX_TEST_FLAG_RATE:
                logging.error(
                    f"{test_flag_rate * 100:.1f}% of known-good test images would be rejected "
                    f"(limit {MAX_TEST_FLAG_RATE * 100:.0f}%). Stats NOT written."
                )
                remove_stale_stats(output_path)
                return None
    else:
        logging.warning("No test images found; OOD stats could not be verified on held-out data.")

    stats = {
        "checkpoint_fingerprint": model.checkpoint_fingerprint,
        "class_names": kept,
        "embedding_dim": EMBEDDING_DIM,
        "class_means": {n: m.tolist() for n, m in class_means.items()},
        "precision_matrix": precision.tolist(),
        "threshold": round(threshold, 4),
        "percentile": MAHALANOBIS_PERCENTILE,
        "num_train_samples": int(used.sum()),
        "num_val_samples": int(len(v_dist)),
        "shrinkage": round(float(lw.shrinkage_), 6),
        "test_flag_rate": None if test_flag_rate is None else round(test_flag_rate, 4),
        "distance_stats": {
            "mean": round(float(v_dist.mean()), 4),
            "std": round(float(v_dist.std()), 4),
            "max": round(float(v_dist.max()), 4),
        },
    }

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    logging.info(f"Class stats saved -> {output_path}")

    assets = android_assets_dir()
    is_production = os.path.abspath(output_path) == os.path.abspath(CLASS_STATS_PATH)
    if is_production and os.path.isdir(assets):
        try:
            shutil.copy2(output_path, os.path.join(assets, "class_stats.json"))
            logging.info(f"Copied class stats to Android assets -> {assets}")
        except OSError as e:
            logging.warning(f"Could not copy stats to Android assets: {e}")

    logging.info("=" * 60)
    return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract embeddings for Mahalanobis OOD detection")
    parser.add_argument("--train-per-class", type=int, default=DEFAULT_STATS_TRAIN_PER_CLASS,
                        help="Train images per class for means/covariance (0 = all). "
                             f"Needs >= {MIN_TOTAL_SAMPLES} total.")
    parser.add_argument("--val-per-class", type=int, default=DEFAULT_STATS_VAL_PER_CLASS,
                        help="Val images per class for threshold calibration (0 = all)")
    parser.add_argument("--test-per-class", type=int, default=DEFAULT_STATS_TEST_PER_CLASS,
                        help="Test images per class for verification (0 = all)")
    parser.add_argument("--model", default=MODEL_PATH, help="Checkpoint to use")
    args = parser.parse_args()

    setup_logging(os.path.join(LOG_DIR, "embeddings.log"))
    result = extract_and_save(args.train_per_class, args.val_per_class, args.test_per_class, args.model)
    raise SystemExit(0 if result else 1)
