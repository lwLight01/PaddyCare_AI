import os
import json
import logging
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
from dataset_utils import val_transform, SafeImageFolder

os.makedirs(LOG_DIR, exist_ok=True)
logging.basicConfig(
    filename=EVALUATE_LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    encoding="utf-8",
)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
if not any(isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler) for h in logging.getLogger().handlers):
    logging.getLogger().addHandler(console)

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


def evaluate():
    if not os.path.isdir(TEST_DIR):
        logging.error(f"Test directory is missing: {TEST_DIR}")
        return

    test_dataset = SafeImageFolder(root=TEST_DIR, transform=val_transform)
    if len(test_dataset) == 0:
        logging.error(f"No test images found in {TEST_DIR}")
        return

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )
    class_names = test_dataset.classes
    logging.info(f"Test samples : {len(test_dataset)}")
    logging.info(f"Classes      : {class_names}")

    model = load_model(MODEL_PATH, device)

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

    acc     = accuracy_score(all_labels, all_preds) * 100
    macro_f1 = f1_score(all_labels, all_preds, average="macro")
    report  = classification_report(all_labels, all_preds, target_names=class_names)
    cm      = confusion_matrix(all_labels, all_preds)

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
    evaluate()
