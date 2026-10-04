import os
import json
import time
import shutil
import random
import logging
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, accuracy_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import (
    MODEL_PATH, MODEL_DIR, LOG_DIR, EPOCHS, LEARNING_RATE,
    TRAIN_LOG_FILE, ACCUM_STEPS, WARMUP_EPOCHS, BATCH_SIZE
)
from model import build_model, unfreeze_all
from dataset_utils import get_dataloaders

# ── Seed for reproducibility ──
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

logging.basicConfig(
    filename=TRAIN_LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    encoding="utf-8",
)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
# Avoid duplicate console handlers if re-run
if not any(isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler) for h in logging.getLogger().handlers):
    logging.getLogger().addHandler(console)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
use_amp = (device.type == "cuda")
logging.info(f"Device: {device} | Mixed Precision (AMP): {use_amp}")


def save_training_curves(history: dict):
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(epochs, history["train_loss"], "b-o", label="Train Loss")
    axes[0].plot(epochs, history["val_loss"],   "r-o", label="Val Loss")
    axes[0].set_title("Training & Validation Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, history["train_acc"], "b-o", label="Train Acc")
    axes[1].plot(epochs, history["val_acc"],   "r-o", label="Val Acc")
    if "val_macro_f1" in history:
        axes[1].plot(epochs, [f * 100 for f in history["val_macro_f1"]], "g--s", label="Val Macro F1")
    axes[1].set_title("Training & Validation Metrics")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Percentage (%)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(LOG_DIR, "training_curves.png")
    plt.savefig(save_path, dpi=150)
    plt.close()
    logging.info(f"Training curves saved -> {save_path}")


def train_one_epoch(model, loader, criterion, optimizer, scaler, phase: int, accum_steps: int = ACCUM_STEPS):
    model.train()
    # In Phase 1 warmup, keep backbone BatchNorm stats frozen in eval mode
    if phase == 1:
        model.features.eval()

    running_loss, correct, total = 0.0, 0, 0
    optimizer.zero_grad()

    for batch_idx, (images, labels) in enumerate(loader):
        images, labels = images.to(device), labels.to(device)

        with torch.cuda.amp.autocast(enabled=use_amp):
            outputs = model(images)
            loss = criterion(outputs, labels) / accum_steps

        scaler.scale(loss).backward()

        if (batch_idx + 1) % accum_steps == 0 or (batch_idx + 1) == len(loader):
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        running_loss += loss.item() * accum_steps * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct  += (preds == labels).sum().item()
        total    += labels.size(0)

        if (batch_idx + 1) % 10 == 0:
            logging.info(
                f"  Batch [{batch_idx+1}/{len(loader)}]  "
                f"Loss: {loss.item() * accum_steps:.4f}"
            )

    epoch_loss = running_loss / total
    epoch_acc  = 100.0 * correct / total
    return epoch_loss, epoch_acc


def validate(model, loader, criterion):
    model.eval()
    running_loss = 0.0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            with torch.cuda.amp.autocast(enabled=use_amp):
                outputs = model(images)
                loss    = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    total = len(all_labels)
    epoch_loss = running_loss / total
    epoch_acc  = 100.0 * accuracy_score(all_labels, all_preds)
    macro_f1   = f1_score(all_labels, all_preds, average="macro")

    return epoch_loss, epoch_acc, macro_f1


def train():
    # ── Checkpoint safety: backup previous model.pth if it exists ──
    if os.path.exists(MODEL_PATH):
        backup_path = os.path.join(MODEL_DIR, "model_backup.pth")
        try:
            shutil.copy2(MODEL_PATH, backup_path)
            logging.info(f"Backed up existing model to: {backup_path}")
        except Exception as e:
            logging.warning(f"Could not backup existing model: {e}")

    loaders, class_names, train_dataset = get_dataloaders()
    num_classes = len(class_names)

    # Note: WeightedRandomSampler already balances batches, so we use standard smoothed loss
    # to avoid compound double-weighting (which over-corrects when scaling to 10+ classes).
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    model = build_model(num_classes=num_classes, pretrained=True).to(device)

    # Phase 1: train only classifier head
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LEARNING_RATE,
        weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=max(1, WARMUP_EPOCHS), T_mult=1, eta_min=1e-6
    )
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    history = {
        "train_loss": [], "train_acc": [],
        "val_loss":   [], "val_acc":   [],
        "val_macro_f1": []
    }

    best_macro_f1 = 0.0
    best_val_acc  = 0.0
    best_epoch    = 0
    start_time    = time.time()
    phase         = 1

    logging.info("=" * 60)
    logging.info("PaddyCare AI - Training Started (EfficientNet-B5)")
    logging.info(f"Classes        : {class_names} ({num_classes} total)")
    logging.info(f"Epochs         : {EPOCHS}  |  LR: {LEARNING_RATE}")
    logging.info(f"Warmup epochs  : {WARMUP_EPOCHS}  (Phase-1 classifier only)")
    logging.info(f"Accum steps    : {ACCUM_STEPS}  (effective batch = {BATCH_SIZE * ACCUM_STEPS})")
    logging.info("=" * 60)

    for epoch in range(1, EPOCHS + 1):
        # ── Phase transition: enter Phase 2 after warmup ──
        if epoch == WARMUP_EPOCHS + 1 and phase == 1:
            phase = 2
            logging.info("\n" + "─" * 60)
            logging.info(f"  ➤ PHASE 2 STARTED (epoch {epoch}): Unfreezing all layers")
            logging.info(f"  ➤ Differential LR: backbone = 1e-5, head = 1e-4")
            logging.info("─" * 60)

            unfreeze_all(model)

            # Differential learning rates for fine-tuning
            optimizer = torch.optim.AdamW([
                {"params": model.features.parameters(), "lr": 1e-5},
                {"params": model.classifier.parameters(), "lr": 1e-4},
            ], weight_decay=1e-4)

            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, T_0=max(1, EPOCHS - WARMUP_EPOCHS), T_mult=1, eta_min=1e-7
            )

        logging.info(f"\nEpoch [{epoch}/{EPOCHS}]  Phase {phase}")

        t_loss, t_acc = train_one_epoch(model, loaders["train"], criterion, optimizer, scaler, phase)
        v_loss, v_acc, v_f1 = validate(model, loaders["val"], criterion)

        scheduler.step()

        history["train_loss"].append(t_loss)
        history["train_acc"].append(t_acc)
        history["val_loss"].append(v_loss)
        history["val_acc"].append(v_acc)
        history["val_macro_f1"].append(v_f1)

        logging.info(f"  Train  -> Loss: {t_loss:.4f}  Acc: {t_acc:.2f}%")
        logging.info(f"  Val    -> Loss: {v_loss:.4f}  Acc: {v_acc:.2f}%  Macro-F1: {v_f1:.4f}")

        # Save checkpoint based on Macro-F1 (balanced across all classes)
        if v_f1 > best_macro_f1:
            best_macro_f1 = v_f1
            best_val_acc  = v_acc
            best_epoch    = epoch
            torch.save(
                {
                    "epoch":                epoch,
                    "model_state_dict":     model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_acc":              v_acc,
                    "val_macro_f1":         v_f1,
                    "class_names":          class_names,
                    "num_classes":          num_classes,
                    "backbone":             "efficientnet_b5",
                    "image_size":           456,
                },
                MODEL_PATH,
            )
            logging.info(f"  [SAVED] Best model at epoch {epoch} (Val Acc: {v_acc:.2f}%, Macro-F1: {v_f1:.4f})")

    elapsed = time.time() - start_time
    logging.info("\n" + "=" * 60)
    logging.info(f"Training complete in {elapsed/60:.1f} min")
    logging.info(f"Best Val Macro-F1: {best_macro_f1:.4f} (Acc: {best_val_acc:.2f}%, Epoch {best_epoch})")

    history_path = os.path.join(LOG_DIR, "history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    save_training_curves(history)
    print(f"\n[train] Model saved to   : {MODEL_PATH}")
    print(f"[train] Training curves  : {LOG_DIR}/training_curves.png")


if __name__ == "__main__":
    train()

    # After training completes, extract embeddings for Mahalanobis OOD detection
    try:
        import extract_embeddings
        extract_embeddings.extract_and_save()
    except Exception as e:
        print(f"\n⚠️ Could not extract embeddings automatically: {e}")
        print("Please run manually: python extract_embeddings.py")
