import os
import json
import logging
import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
)
from torch.utils.data import DataLoader

from config import (
    MODEL_PATH, LOG_DIR, BATCH_SIZE,
    NUM_WORKERS, TEST_DIR, EVALUATE_LOG_FILE
)
from model import load_model
from dataset_utils import val_transform, SafeImageFolder, subset_per_class
from log_utils import setup_logging

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def plot_confusion_matrix(cm: np.ndarray, class_names: list):
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
    )
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title("Confusion Matrix - PaddyCare AI")
    plt.tight_layout()
    path = os.path.join(LOG_DIR, "confusion_matrix.png")
    plt.savefig(path, dpi=150)
    plt.close()
    logging.info(f"Confusion matrix saved -> {path}")


def evaluate(model_path: str = MODEL_PATH, subset: int = 0):
    if not os.path.isdir(TEST_DIR):
        logging.error(f"Test directory is missing: {TEST_DIR}")
        return
    if not os.path.exists(model_path):
        logging.error(f"Model not found: {model_path}")
        return

    model = load_model(model_path, device)
    class_names = model.class_names
    if not class_names:
        logging.error("Checkpoint has no class_names; cannot evaluate reliably.")
        return

    # Force the test set onto the CHECKPOINT's class order so labels always line up.
    test_dataset = SafeImageFolder(root=TEST_DIR, transform=val_transform, classes=class_names)
    if len(test_dataset) == 0:
        logging.error(f"No test images found in {TEST_DIR} for classes {class_names}")
        return

    if subset > 0:
        test_dataset, _ = subset_per_class(test_dataset, subset)

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )
    logging.info(f"Test samples : {len(test_dataset)}")
    logging.info(f"Classes      : {class_names}")

    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)

    label_ids = list(range(len(class_names)))
    present   = sorted(set(all_labels.tolist()))
    absent    = [class_names[i] for i in label_ids if i not in present]
    if absent:
        logging.warning(f"Classes with no test images (excluded from macro-F1): {absent}")

    acc      = accuracy_score(all_labels, all_preds) * 100
    macro_f1 = f1_score(all_labels, all_preds, labels=present, average="macro", zero_division=0)
    report   = classification_report(
        all_labels, all_preds, labels=label_ids, target_names=class_names, zero_division=0
    )
    cm       = confusion_matrix(all_labels, all_preds, labels=label_ids)

    logging.info(f"\nOverall Accuracy : {acc:.2f}%")
    logging.info(f"Macro-F1 Score   : {macro_f1:.4f}")
    logging.info(f"\nClassification Report:\n{report}")

    results = {
        "overall_accuracy": round(acc, 2),
        "macro_f1": round(float(macro_f1), 4),
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
        "classes": class_names,
    }
    results_path = os.path.join(LOG_DIR, "evaluate_results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logging.info(f"Results saved -> {results_path}")

    plot_confusion_matrix(cm, class_names)

    print(f"\n[evaluate] Overall Accuracy : {acc:.2f}%")
    print(f"[evaluate] Macro-F1 Score   : {macro_f1:.4f}")
    print(f"[evaluate] Results saved to : {LOG_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a checkpoint on the test set")
    parser.add_argument("--model", default=MODEL_PATH, help="Checkpoint to evaluate")
    parser.add_argument("--subset", type=int, default=0, help="Evaluate on at most N images per class")
    args = parser.parse_args()
    setup_logging(EVALUATE_LOG_FILE)
    evaluate(args.model, subset=args.subset)
