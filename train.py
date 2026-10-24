import os
import json
import time
import shutil
import random
import logging
import argparse
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, accuracy_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import (
    MODEL_PATH, MODEL_DIR, RUNS_DIR, BACKUP_DIR, LOG_DIR, CLASS_STATS_PATH,
    EPOCHS as DEFAULT_EPOCHS, LEARNING_RATE, BACKBONE_LR, HEAD_LR, WEIGHT_DECAY,
    TRAIN_LOG_FILE, ACCUM_STEPS, WARMUP_EPOCHS as DEFAULT_WARMUP, BATCH_SIZE,
    IMAGE_SIZE, SEED, EARLY_STOP_PATIENCE, LAST_CHECKPOINT_NAME,
)
from model import build_model, unfreeze_all, set_bn_eval
from dataset_utils import get_dataloaders, split_fingerprint
from log_utils import setup_logging


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def save_training_curves(history: dict, save_dir: str):
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
    save_path = os.path.join(save_dir, "training_curves.png")
    plt.savefig(save_path, dpi=150)
    plt.close()
    logging.info(f"Training curves saved -> {save_path}")
    return save_path


def make_param_groups(parts: list, weight_decay: float = WEIGHT_DECAY) -> list:
    groups = []
    for module, lr in parts:
        decay, no_decay = [], []
        for param in module.parameters():
            if not param.requires_grad:
                continue
            (no_decay if param.ndim <= 1 else decay).append(param)
        if decay:
            groups.append({"params": decay, "lr": lr, "weight_decay": weight_decay})
        if no_decay:
            groups.append({"params": no_decay, "lr": lr, "weight_decay": 0.0})
    return groups


def make_phase1_optim(model, warmup_epochs: int):
    optimizer = torch.optim.AdamW(make_param_groups([(model.classifier, LEARNING_RATE)]))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=max(1, warmup_epochs), T_mult=1, eta_min=1e-6
    )
    return optimizer, scheduler


def make_phase2_optim(model, epochs: int, warmup_epochs: int):
    unfreeze_all(model, keep_bn_frozen=True)
    optimizer = torch.optim.AdamW(make_param_groups([
        (model.features, BACKBONE_LR),
        (model.classifier, HEAD_LR),
    ]))
    remaining_epochs = max(1, epochs - warmup_epochs)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=remaining_epochs, T_mult=1, eta_min=1e-7
    )
    return optimizer, scheduler


def train_one_epoch(model, loader, criterion, optimizer, scaler, phase: int, device, use_amp: bool,
                    accum_steps: int = ACCUM_STEPS):
    model.train()
    if phase == 1:
        model.features.eval()
    set_bn_eval(model)

    running_loss, correct, total = 0.0, 0, 0
    optimizer.zero_grad()

    for batch_idx, (images, labels) in enumerate(loader):
        images, labels = images.to(device), labels.to(device)

        with torch.amp.autocast("cuda", enabled=use_amp):
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


def validate(model, loader, criterion, device, use_amp: bool):
    model.eval()
    running_loss = 0.0
    all_preds, all_labels = [], []

    with torch.inference_mode():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            with torch.amp.autocast("cuda", enabled=use_amp):
                outputs = model(images)
                loss    = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    total = len(all_labels)
    epoch_loss = running_loss / total
    epoch_acc  = 100.0 * accuracy_score(all_labels, all_preds)
    macro_f1   = f1_score(all_labels, all_preds, average="macro", zero_division=0)

    return epoch_loss, epoch_acc, macro_f1


def should_promote_checkpoint(new_meta: dict, current_path: str = MODEL_PATH, force: bool = False) -> tuple[bool, str]:

    if not os.path.exists(current_path):
        return True, "No existing model.pth found; promoting new model."
    if force:
        return True, "--force given; promoting new model without comparison."

    if new_meta.get("subset", 0):
        return False, (f"Run used --subset {new_meta['subset']}; subset runs are never promoted "
                       "automatically (use --force to override).")

    try:
        old_cp = torch.load(current_path, map_location="cpu", weights_only=True)
    except Exception as e:
        return True, f"Could not inspect current checkpoint ({e}); promoting new model."

    old_classes = old_cp.get("class_names")
    new_classes = new_meta.get("class_names")
    if old_classes != new_classes:
        return False, (f"Class list changed ({old_classes} -> {new_classes}); scores are not comparable. "
                       "Check the run and use --force to promote.")

    old_f1 = old_cp.get("val_macro_f1")
    if old_f1 is None:
        old_acc = old_cp.get("val_acc", 0.0)
        return True, f"Current checkpoint lacks val_macro_f1 (acc={old_acc:.2f}%); promoting new model."

    old_fp = old_cp.get("val_fingerprint")
    new_fp = new_meta.get("val_fingerprint")
    if old_fp is None or old_fp != new_fp:
        return False, (f"Validation set differs (current={old_fp}, new={new_fp}); Macro-F1 scores are not "
                       "comparable. Evaluate both on the test set and use --force to promote.")

    new_f1 = new_meta.get("val_macro_f1", 0.0)
    if new_f1 >= old_f1:
        return True, f"New Macro-F1 ({new_f1:.4f}) >= Current ({old_f1:.4f}); promoting."
    return False, f"Current Macro-F1 ({old_f1:.4f}) is higher than New ({new_f1:.4f}); keeping current."


def train(
    epochs: int = None,
    warmup_epochs: int = DEFAULT_WARMUP,
    subset_per_class: int = 0,
    smoke: bool = False,
    extract_stats: bool = True,
    resume_path: str = None,
    force_promote: bool = False,
    patience: int = EARLY_STOP_PATIENCE,
):
    set_seed(SEED)
    device = get_device()
    use_amp = (device.type == "cuda")

    resume_state = None
    if resume_path:
        resume_state = torch.load(resume_path, map_location="cpu", weights_only=False)
        run_dir   = os.path.dirname(os.path.abspath(resume_path))
        timestamp = resume_state["timestamp"]
        warmup_epochs    = resume_state["warmup_epochs"]
        subset_per_class = resume_state["subset"]
        epochs = epochs or resume_state["epochs"]
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = os.path.join(RUNS_DIR, f"run_{timestamp}")
    epochs = epochs or DEFAULT_EPOCHS

    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(BACKUP_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    loaders, class_names, _ = get_dataloaders(max_per_class=subset_per_class)
    num_classes = len(class_names)
    val_fingerprint = split_fingerprint(loaders["val"].dataset)
    val_size = len(loaders["val"].dataset)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    model = build_model(num_classes=num_classes, pretrained=resume_state is None).to(device)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    history = {
        "train_loss": [], "train_acc": [],
        "val_loss":   [], "val_acc":   [],
        "val_macro_f1": []
    }
    best_macro_f1 = 0.0
    best_val_acc  = 0.0
    best_epoch    = 0
    epochs_no_improve = 0
    phase         = 1
    start_epoch   = 1

    if resume_state is not None:
        if resume_state["class_names"] != class_names:
            raise ValueError(f"Cannot resume: dataset classes {class_names} != checkpoint "
                             f"classes {resume_state['class_names']}")
        model.load_state_dict(resume_state["model_state_dict"])
        phase = resume_state["phase"]
        if phase == 2:
            optimizer, scheduler = make_phase2_optim(model, epochs, warmup_epochs)
        else:
            optimizer, scheduler = make_phase1_optim(model, warmup_epochs)
        optimizer.load_state_dict(resume_state["optimizer_state_dict"])
        scheduler.load_state_dict(resume_state["scheduler_state_dict"])
        scaler.load_state_dict(resume_state["scaler_state_dict"])
        history       = resume_state["history"]
        best_macro_f1 = resume_state["best_macro_f1"]
        best_val_acc  = resume_state["best_val_acc"]
        best_epoch    = resume_state["best_epoch"]
        epochs_no_improve = resume_state["epochs_no_improve"]
        start_epoch   = resume_state["epoch"] + 1
    else:
        optimizer, scheduler = make_phase1_optim(model, warmup_epochs)

    start_time    = time.time()
    run_best_model_path = os.path.join(run_dir, "model.pth")
    last_path = os.path.join(run_dir, LAST_CHECKPOINT_NAME)

    effective_batch = BATCH_SIZE * ACCUM_STEPS
    logging.info("=" * 60)
    logging.info(f"PaddyCare AI - Training {'Resumed' if resume_state else 'Started'} (EfficientNet-B5)")
    logging.info(f"Run output dir : {run_dir}")
    logging.info(f"Device         : {device} | AMP: {use_amp}")
    logging.info(f"Classes        : {class_names} ({num_classes} total)")
    logging.info(f"Epochs         : {epochs}  |  Phase-1 LR: {LEARNING_RATE}")
    logging.info(f"Warmup epochs  : {warmup_epochs} (Phase 1 classifier only)")
    logging.info(f"Accum steps    : {ACCUM_STEPS} (effective batch = {effective_batch})")
    logging.info(f"Early stopping : patience {patience} (Phase 2, 0 = off)")
    logging.info(f"Val set        : {val_size} images, fingerprint {val_fingerprint}")
    if subset_per_class > 0:
        logging.info(f"Subset limit   : {subset_per_class} images per class")
    if smoke:
        logging.info("SMOKE MODE     : Enabled (weights will NOT overwrite production model.pth)")
    logging.info("=" * 60)

    for epoch in range(start_epoch, epochs + 1):
        if epoch == warmup_epochs + 1 and phase == 1:
            phase = 2
            logging.info("\n" + "─" * 60)
            logging.info(f"  ➤ PHASE 2 STARTED (epoch {epoch}): Unfreezing all layers (BatchNorm stays frozen)")
            logging.info(f"  ➤ Differential LR: backbone = {BACKBONE_LR}, head = {HEAD_LR}")
            logging.info("─" * 60)
            optimizer, scheduler = make_phase2_optim(model, epochs, warmup_epochs)

        logging.info(f"\nEpoch [{epoch}/{epochs}]  Phase {phase}")

        t_loss, t_acc = train_one_epoch(model, loaders["train"], criterion, optimizer, scaler, phase,
                                        device, use_amp)
        v_loss, v_acc, v_f1 = validate(model, loaders["val"], criterion, device, use_amp)

        scheduler.step()

        history["train_loss"].append(t_loss)
        history["train_acc"].append(t_acc)
        history["val_loss"].append(v_loss)
        history["val_acc"].append(v_acc)
        history["val_macro_f1"].append(v_f1)

        logging.info(f"  Train -> Loss: {t_loss:.4f}  Acc: {t_acc:.2f}%")
        logging.info(f"  Val   -> Loss: {v_loss:.4f}  Acc: {v_acc:.2f}%  Macro-F1: {v_f1:.4f}")

        checkpoint_meta = {
            "class_names":     class_names,
            "num_classes":     num_classes,
            "backbone":        "efficientnet_b5",
            "image_size":      IMAGE_SIZE,
            "timestamp":       timestamp,
            "subset":          subset_per_class,
            "val_fingerprint": val_fingerprint,
            "val_size":        val_size,
        }

        if v_f1 > best_macro_f1 or best_epoch == 0:
            best_macro_f1 = v_f1
            best_val_acc  = v_acc
            best_epoch    = epoch
            epochs_no_improve = 0
            torch.save({
                **checkpoint_meta,
                "epoch":            epoch,
                "model_state_dict": model.state_dict(),
                "val_acc":          v_acc,
                "val_macro_f1":     v_f1,
            }, run_best_model_path)
            logging.info(f"  [SAVED] Run best model at epoch {epoch} (Acc: {v_acc:.2f}%, F1: {v_f1:.4f})")
        elif phase == 2:
            epochs_no_improve += 1

        torch.save({
            **checkpoint_meta,
            "epoch":                epoch,
            "epochs":               epochs,
            "warmup_epochs":        warmup_epochs,
            "phase":                phase,
            "model_state_dict":     model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict":    scaler.state_dict(),
            "history":              history,
            "best_macro_f1":        best_macro_f1,
            "best_val_acc":         best_val_acc,
            "best_epoch":           best_epoch,
            "epochs_no_improve":    epochs_no_improve,
        }, last_path)

        if phase == 2 and patience > 0 and epochs_no_improve >= patience:
            logging.info(f"  Early stopping: no val Macro-F1 improvement for {patience} Phase-2 epochs.")
            break

    elapsed = time.time() - start_time
    logging.info("\n" + "=" * 60)
    logging.info(f"Training finished in {elapsed/60:.1f} min")
    logging.info(f"Best Val Macro-F1: {best_macro_f1:.4f} (Acc: {best_val_acc:.2f}%, Epoch {best_epoch})")

    history_path = os.path.join(run_dir, "history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    curves_path = save_training_curves(history, run_dir)

    if smoke:
        logging.info(f"Smoke test complete. Run model kept in {run_best_model_path} (not promoted).")
        return run_best_model_path

    best_meta = {
        "val_macro_f1":    best_macro_f1,
        "val_acc":         best_val_acc,
        "class_names":     class_names,
        "subset":          subset_per_class,
        "val_fingerprint": val_fingerprint,
    }
    promote, reason = should_promote_checkpoint(best_meta, MODEL_PATH, force=force_promote)
    logging.info(f"[Checkpoint Safety] {reason}")

    if promote:
        if os.path.exists(MODEL_PATH):
            backup_file = os.path.join(BACKUP_DIR, f"model_backup_{timestamp}.pth")
            shutil.copy2(MODEL_PATH, backup_file)
            logging.info(f"Previous production checkpoint backed up to: {backup_file}")

        shutil.copy2(run_best_model_path, MODEL_PATH)
        shutil.copy2(curves_path, os.path.join(LOG_DIR, "training_curves.png"))
        shutil.copy2(history_path, os.path.join(LOG_DIR, "history.json"))
        logging.info(f"PROMOTED new best model -> {MODEL_PATH}")
    else:
        logging.info(f"Run best model remains in {run_best_model_path}; {MODEL_PATH} was preserved.")

    import extract_embeddings
    target_for_stats = MODEL_PATH if promote else run_best_model_path
    stats_out = CLASS_STATS_PATH if promote else os.path.join(run_dir, "class_stats.json")
    if extract_stats:
        try:
            logging.info("\nStarting OOD embedding stats extraction...")
            result = extract_embeddings.extract_and_save(model_path=target_for_stats, output_path=stats_out)
            if result is None and promote:
                logging.warning("OOD stats were not produced for the new production model; "
                                "export_tflite.py will refuse to ship without them.")
        except Exception as e:
            logging.warning(f"Could not extract embeddings automatically: {e}")
            logging.info("You can run manually: python extract_embeddings.py")
            if promote:
                extract_embeddings.remove_stale_stats(CLASS_STATS_PATH)
    elif promote:
        logging.warning("--no-stats: removing the previous model's OOD stats so they cannot ship with the "
                        "new model. Run `python extract_embeddings.py` before exporting.")
        extract_embeddings.remove_stale_stats(CLASS_STATS_PATH)

    return target_for_stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PaddyCare AI EfficientNet-B5 classifier")
    parser.add_argument("--epochs", type=int, default=None, help=f"Total training epochs (default {DEFAULT_EPOCHS})")
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP, help="Phase 1 warmup epochs")
    parser.add_argument("--subset", type=int, default=0, help="Limit images per class (0 = full dataset)")
    parser.add_argument("--smoke", action="store_true", help="Quick dry run; do not promote checkpoint")
    parser.add_argument("--no-stats", action="store_true", help="Skip OOD stats extraction at the end")
    parser.add_argument("--resume", default=None, help=f"Resume from a run's {LAST_CHECKPOINT_NAME}")
    parser.add_argument("--force", action="store_true",
                        help="Promote the run even if its score is not comparable with the current model")
    parser.add_argument("--patience", type=int, default=EARLY_STOP_PATIENCE,
                        help="Early-stopping patience in Phase-2 epochs (0 = off)")
    args = parser.parse_args()

    setup_logging(TRAIN_LOG_FILE)
    train(
        epochs=args.epochs,
        warmup_epochs=args.warmup,
        subset_per_class=args.subset,
        smoke=args.smoke,
        extract_stats=not args.no_stats,
        resume_path=args.resume,
        force_promote=args.force,
        patience=args.patience,
    )
