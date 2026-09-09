import os
import json
import time
import logging
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import (
    MODEL_PATH, MODEL_DIR, LOG_DIR, EPOCHS, LEARNING_RATE,
    CLASS_NAMES, TRAIN_LOG_FILE, ACCUM_STEPS, WARMUP_EPOCHS, BATCH_SIZE
)
from model import build_model, unfreeze_all
from dataset_utils import get_dataloaders, get_class_weights

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
logging.getLogger().addHandler(console)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"Device: {device}")


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
    axes[1].set_title("Training & Validation Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(LOG_DIR, "training_curves.png")
    plt.savefig(save_path, dpi=150)
    plt.close()
    logging.info(f"Training curves saved -> {save_path}")


def train_one_epoch(model, loader, criterion, optimizer, accum_steps: int = ACCUM_STEPS):
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    optimizer.zero_grad()
    for batch_idx, (images, labels) in enumerate(loader):
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        # Scale loss by accum_steps so gradients average rather than sum
        loss = criterion(outputs, labels) / accum_steps
        loss.backward()

        if (batch_idx + 1) % accum_steps == 0 or (batch_idx + 1) == len(loader):
            optimizer.step()
            optimizer.zero_grad()

        running_loss += loss.item() * accum_steps * images.size(0)  # undo scaling for logging
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
    running_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss    = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct  += (preds == labels).sum().item()
            total    += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc  = 100.0 * correct / total
    return epoch_loss, epoch_acc


def train():
    # get_dataloaders now also returns train_dataset for class-weight computation
    loaders, class_names, train_dataset = get_dataloaders()

    class_weights = get_class_weights(train_dataset).to(device)
    criterion = nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=0.1,
    )

    model = build_model(pretrained=True).to(device)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LEARNING_RATE,
        weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=1, eta_min=1e-6
    )

    history = {
        "train_loss": [], "train_acc": [],
        "val_loss":   [], "val_acc":   [],
    }

    best_val_acc = 0.0
    best_epoch   = 0
    start_time   = time.time()
    phase        = 1

    logging.info("=" * 60)
    logging.info("PaddyCare AI - Training Started (EfficientNet-B5)")
    logging.info(f"Classes        : {class_names}")
    logging.info(f"Epochs         : {EPOCHS}  |  LR: {LEARNING_RATE}")
    logging.info(f"Warmup epochs  : {WARMUP_EPOCHS}  (Phase-1 classifier only)")
    logging.info(f"Accum steps    : {ACCUM_STEPS}  (effective batch = {BATCH_SIZE * ACCUM_STEPS})")
    logging.info(f"Class weights  : {class_weights.tolist()}")
    logging.info("=" * 60)

    for epoch in range(1, EPOCHS + 1):

        # ── Phase transition: enter Phase 2 after warmup ──
        if epoch == WARMUP_EPOCHS + 1 and phase == 1:
            phase = 2
            logging.info("\n" + "─" * 60)
            logging.info(f"  ➤ PHASE 2 STARTED (epoch {epoch}): Unfreezing all layers")
            logging.info(f"  ➤ Switching to fine-tuning LR = 1e-5")
            logging.info("─" * 60)

            unfreeze_all(model)

            # Rebuild optimizer with a much smaller LR for gentle fine-tuning
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=1e-5,
                weight_decay=1e-4,
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, T_0=max(1, EPOCHS - WARMUP_EPOCHS), T_mult=1, eta_min=1e-7
            )

        logging.info(f"\nEpoch [{epoch}/{EPOCHS}]  Phase {phase}")

        t_loss, t_acc = train_one_epoch(model, loaders["train"], criterion, optimizer)
        v_loss, v_acc = validate(model, loaders["val"], criterion)

        scheduler.step(epoch - 1 + 0.0)

        history["train_loss"].append(t_loss)
        history["train_acc"].append(t_acc)
        history["val_loss"].append(v_loss)
        history["val_acc"].append(v_acc)

        logging.info(
            f"  Train  -> Loss: {t_loss:.4f}  Acc: {t_acc:.2f}%"
        )
        logging.info(
            f"  Val    -> Loss: {v_loss:.4f}  Acc: {v_acc:.2f}%"
        )

        if v_acc > best_val_acc:
            best_val_acc = v_acc
            best_epoch   = epoch
            torch.save(
                {
                    "epoch":               epoch,
                    "model_state_dict":    model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_acc":             v_acc,
                    "class_names":         class_names,
                    "backbone":            "efficientnet_b5",
                },
                MODEL_PATH,
            )
            logging.info(f"  [SAVED] Best model at epoch {epoch}  (Val Acc: {v_acc:.2f}%)")

    elapsed = time.time() - start_time
    logging.info("\n" + "=" * 60)
    logging.info(f"Training complete in {elapsed/60:.1f} min")
    logging.info(f"Best Val Acc: {best_val_acc:.2f}%  (Epoch {best_epoch})")

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
