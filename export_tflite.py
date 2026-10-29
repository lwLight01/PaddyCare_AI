from __future__ import annotations

import os
import sys
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import shutil
import argparse
import numpy as np
import torch
import torch.nn as nn

import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    IMAGE_SIZE, MODEL_DIR, MODEL_PATH, CLASS_STATS_PATH, DISEASES_JSON_PATH,
    MODEL_META_PATH, NORM_MEAN, NORM_STD, CONFIDENCE_THRESHOLD, ENTROPY_FACTOR,
    TEST_DIR,
)
from model import (
    build_model as _build_model, DualOutputWrapper, load_checkpoint, checkpoint_fingerprint,
    forward_with_embedding,
)

PTH_PATH    = MODEL_PATH
ONNX_PATH   = os.path.join(MODEL_DIR, "model.onnx")
TFLITE_DIR  = os.path.join(MODEL_DIR, "tflite_output")
TFLITE_PATH = os.path.join(MODEL_DIR, "model.tflite")
LABELS_PATH = os.path.join(MODEL_DIR, "labels.txt")

DEFAULT_PARITY_TOL = 1e-2
PARITY_SAMPLES = 4


def step1_load_pytorch(model_path: str = PTH_PATH) -> tuple[nn.Module, list, str]:
    print("\n[Step 1/4] Loading PyTorch model (EfficientNet-B5) ...")
    if not os.path.exists(model_path):
        print(f"Model file not found: {model_path}")
        print("   Run 'python train.py' first to train the model.")
        sys.exit(1)

    try:
        checkpoint = load_checkpoint(model_path, map_location="cpu")
    except ValueError as e:
        print(f"{e}")
        sys.exit(1)
    state_dict = checkpoint["model_state_dict"]
    class_names = list(checkpoint["class_names"])
    num_classes = len(class_names)

    model = _build_model(num_classes=num_classes, pretrained=False)
    model.load_state_dict(state_dict)
    model.eval()

    fingerprint = checkpoint_fingerprint(model_path)
    epoch    = checkpoint.get("epoch", "?")
    val_acc  = checkpoint.get("val_acc", "?")
    backbone = checkpoint.get("backbone", "efficientnet_b5")

    print(f"  Backbone : {backbone}")
    print(f"  Input    : {IMAGE_SIZE}x{IMAGE_SIZE}")
    print(f"  Loaded model from epoch {epoch} (Val Acc: {val_acc})")
    print(f"  Classes  : {class_names} ({num_classes} total)")
    print(f"  Checkpoint fingerprint: {fingerprint}")
    return model, class_names, fingerprint


def step2_export_onnx(model: nn.Module):
    print("\n[Step 2/4] Exporting to ONNX (Dual Output: logits + embedding) ...")
    wrapper = DualOutputWrapper(model)
    wrapper.eval()

    dummy_input = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE)

    torch.onnx.export(
        wrapper,
        dummy_input,
        ONNX_PATH,
        export_params=True,
        opset_version=18,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["logits", "embedding"],
    )

    size_mb = os.path.getsize(ONNX_PATH) / (1024 * 1024)
    print(f"ONNX model saved: {ONNX_PATH} ({size_mb:.1f} MB)")

    try:
        import onnx
        onnx_model = onnx.load(ONNX_PATH)
        onnx.checker.check_model(onnx_model)
        print(" ONNX model validation passed")
    except ImportError:
        print(" onnx package not installed, skipping ONNX validation check.")


def step3_convert_tflite(quantize: bool = False) -> bool:
    """Convert ONNX -> TFLite. Returns True if the dynamic-range quantized model was produced."""
    print("\n[Step 3/4] Converting ONNX → TFLite ...")

    try:
        import onnx2tf
    except ImportError:
        print(
            "onnx2tf not installed. "
            "Run: pip install onnx2tf tensorflow"
        )
        sys.exit(1)

    os.makedirs(TFLITE_DIR, exist_ok=True)

    onnx2tf.convert(
        input_onnx_file_path=ONNX_PATH,
        output_folder_path=TFLITE_DIR,
        non_verbose=True,
    )

    generated_tflite = os.path.join(TFLITE_DIR, "model_float32.tflite")
    if not os.path.exists(generated_tflite):
        print(f"   Float32 TFLite model not found: {generated_tflite}")
        sys.exit(1)

    shutil.copy2(generated_tflite, TFLITE_PATH)
    size_mb = os.path.getsize(TFLITE_PATH) / (1024 * 1024)
    print(f"   TFLite model saved: {TFLITE_PATH} ({size_mb:.1f} MB (float32))")

    if not quantize:
        return False

    print("  Applying dynamic-range quantization (int8 weights, float activations) ...")
    try:
        import tensorflow as tf
    except ImportError:
        print("  TensorFlow not installed, keeping the float32 model.")
        return False
    converter = tf.lite.TFLiteConverter.from_saved_model(TFLITE_DIR)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    quantized_model = converter.convert()

    with open(TFLITE_PATH, "wb") as f:
        f.write(quantized_model)

    q_size_mb = os.path.getsize(TFLITE_PATH) / (1024 * 1024)
    print(f"  Quantized model saved: {TFLITE_PATH} ({q_size_mb:.1f} MB (dynamic-range int8))")
    return True


def _run_tflite(interpreter, x_nchw: np.ndarray) -> list:
    """Run one NCHW float image through the (NHWC) TFLite model; returns all output tensors."""
    inp = interpreter.get_input_details()[0]
    nhwc = np.transpose(x_nchw, (0, 2, 3, 1)).astype(inp["dtype"])
    interpreter.set_tensor(inp["index"], nhwc)
    interpreter.invoke()
    return [interpreter.get_tensor(o["index"]) for o in interpreter.get_output_details()]


def _match_outputs(outs: list, pt_logits: np.ndarray, pt_emb: np.ndarray):
    """Find which TFLite output is the logits and which the embedding by comparing values with PyTorch.

    Returns (max_abs_error, logits_index, embedding_index) or None if no pairing has the right shapes.
    """
    best = None
    for li in range(len(outs)):
        for ei in range(len(outs)):
            if li == ei or outs[li].shape != pt_logits.shape or outs[ei].shape != pt_emb.shape:
                continue
            err = max(float(np.abs(outs[li] - pt_logits).max()), float(np.abs(outs[ei] - pt_emb).max()))
            if best is None or err < best[0]:
                best = (err, li, ei)
    return best


def _load_eval_dataset(class_names: list, per_class: int):
    """Real test images (validation transform) or None when no test split is available."""
    from dataset_utils import val_transform, SafeImageFolder, subset_per_class
    if not os.path.isdir(TEST_DIR):
        return None
    ds = SafeImageFolder(root=TEST_DIR, transform=val_transform, classes=class_names)
    if len(ds) == 0:
        return None
    if per_class > 0:
        ds, _ = subset_per_class(ds, per_class)
    return ds


def step4_validate_tflite(model: nn.Module, class_names: list, quantized: bool,
                          eval_per_class: int, parity_tol: float) -> dict | None:
    """Check the TFLite model against PyTorch on real images and measure its accuracy.

    Returns the discovered output indices and test metrics (None if TensorFlow is unavailable).
    """
    print("\n[Step 4/4] Validating TFLite model against PyTorch ...")

    try:
        import tensorflow as tf
    except ImportError:
        print("  TensorFlow not installed, skipping validation (the app will fall back to output-shape detection).")
        return None

    interpreter = tf.lite.Interpreter(model_path=TFLITE_PATH)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_shape = tuple(int(d) for d in input_details[0]["shape"])
    expected_shape = (1, IMAGE_SIZE, IMAGE_SIZE, 3)

    print(f"  Input  shape: {input_shape}  dtype: {input_details[0]['dtype']}")
    for i, out_det in enumerate(output_details):
        print(f"  Output [{i}] shape: {tuple(out_det['shape'])}  name: {out_det['name']}")

    if input_shape != expected_shape:
        print(f"  ERROR: TFLite input is {input_shape}, but the app feeds NHWC {expected_shape}.")
        sys.exit(1)

    eval_ds = _load_eval_dataset(class_names, eval_per_class)
    if eval_ds is not None:
        samples = [eval_ds[i][0] for i in np.linspace(0, len(eval_ds) - 1, PARITY_SAMPLES, dtype=int)]
    else:
        print("  No test images found; using random inputs for the parity check and skipping accuracy.")
        samples = [torch.randn(3, IMAGE_SIZE, IMAGE_SIZE) for _ in range(PARITY_SAMPLES)]

    logits_idx = emb_idx = None
    worst = 0.0
    model.eval()
    for n, x in enumerate(samples):
        with torch.inference_mode():
            pt_logits, pt_emb = forward_with_embedding(model, x[None])
        pt_logits, pt_emb = pt_logits.numpy(), pt_emb.numpy()
        outs = _run_tflite(interpreter, x[None].numpy())
        if n == 0:
            match = _match_outputs(outs, pt_logits, pt_emb)
            if match is None:
                print("  ERROR: no TFLite outputs have the expected logits/embedding shapes.")
                sys.exit(1)
            _, logits_idx, emb_idx = match
        worst = max(worst,
                    float(np.abs(outs[logits_idx] - pt_logits).max()),
                    float(np.abs(outs[emb_idx] - pt_emb).max()))
        if int(outs[logits_idx].argmax()) != int(pt_logits.argmax()):
            print(f"  ERROR: TFLite and PyTorch predict different classes on parity sample {n}.")
            sys.exit(1)

    print(f"  Logits output index: {logits_idx} | embedding output index: {emb_idx}")
    print(f"  Max |PyTorch - TFLite| over {len(samples)} images: {worst:.5f} (tolerance {parity_tol})")
    if worst > parity_tol:
        if quantized:
            print("  WARNING: difference exceeds tolerance (expected for quantized models). Check the accuracy below.")
        else:
            print("  ERROR: float32 TFLite output deviates from PyTorch. The conversion is not trustworthy.")
            sys.exit(1)

    info = {"logits_output_index": int(logits_idx), "embedding_output_index": int(emb_idx)}

    if eval_ds is not None:
        from sklearn.metrics import accuracy_score, f1_score
        preds, labels = [], []
        for i in range(len(eval_ds)):
            x, y = eval_ds[i]
            preds.append(int(_run_tflite(interpreter, x[None].numpy())[logits_idx].argmax()))
            labels.append(int(y))
        acc = 100.0 * float(accuracy_score(labels, preds))
        f1 = float(f1_score(labels, preds, average="macro", zero_division=0))
        print(f"  TFLite test accuracy: {acc:.2f}% | Macro-F1: {f1:.4f} ({len(labels)} images)")
        info.update({"tflite_test_accuracy": round(acc, 2), "tflite_test_macro_f1": round(f1, 4),
                     "tflite_test_samples": len(labels)})

    print("  TFLite model matches PyTorch.")
    return info


def validate_diseases_json(class_names: list):
    if not os.path.exists(DISEASES_JSON_PATH):
        raise FileNotFoundError(f"Missing {DISEASES_JSON_PATH}. Required for app disease information.")
    with open(DISEASES_JSON_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    diseases = data.get("diseases", data) if isinstance(data, dict) else {}
    missing = [c for c in class_names if c not in diseases and c != "Not Leaf"]
    if missing:
        raise ValueError(
            f"The following model classes are missing from {DISEASES_JSON_PATH}: {missing}\n"
            "Please update diseases.json before exporting to Android."
        )
    print(f"   diseases.json validated ({len(class_names)} classes covered)")


def check_ood_stats(fingerprint: str, require: bool) -> bool:
    """True if class_stats.json exists AND was computed from this exact checkpoint."""
    problem = None
    if not os.path.exists(CLASS_STATS_PATH):
        problem = ("class_stats.json not found. On-device out-of-distribution (Mahalanobis) detection "
                   "will be DISABLED. Run `python extract_embeddings.py`, then export again.")
    else:
        with open(CLASS_STATS_PATH, "r", encoding="utf-8") as f:
            stats_fp = json.load(f).get("checkpoint_fingerprint")
        if stats_fp != fingerprint:
            problem = (f"class_stats.json belongs to a different (or unknown) checkpoint "
                       f"(stats: {stats_fp}, model: {fingerprint}), so it will NOT be shipped. "
                       "Run `python extract_embeddings.py`, then export again.")
    if problem is None:
        return True
    if require:
        print(f"\n   ERROR: {problem}")
        sys.exit(1)
    print(f"\n   {'!' * 60}\n   WARNING: {problem}\n   {'!' * 60}")
    return False


def write_model_meta(class_names: list, fingerprint: str, ood_enabled: bool, tflite_info: dict | None,
                     quantized: bool, backbone: str = "efficientnet_b5"):
    meta = {
        "image_size": IMAGE_SIZE,
        "mean": NORM_MEAN,
        "std": NORM_STD,
        "backbone": backbone,
        "classes": class_names,
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "entropy_factor": ENTROPY_FACTOR,
        "checkpoint_fingerprint": fingerprint,
        "precision": "dynamic_range_int8" if quantized else "float32",
        "ood_stats_included": ood_enabled,
    }
    if tflite_info:
        meta.update(tflite_info)
    with open(MODEL_META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"   Model metadata saved: {MODEL_META_PATH}")
    return MODEL_META_PATH


def create_labels_file(class_names: list):
    """Create labels.txt matching the model checkpoint classes."""
    with open(LABELS_PATH, "w", encoding="utf-8") as f:
        for name in class_names:
            f.write(name + "\n")
    print(f"\n   Labels file saved: {LABELS_PATH} ({len(class_names)} classes)")


def copy_assets_to_android(ship_stats: bool):
    assets_dir = os.path.join(os.path.dirname(MODEL_DIR), "PaddyCareAndroid", "app", "src", "main", "assets")
    if not os.path.isdir(assets_dir):
        return

    print("\n   Syncing assets to Android project...")
    files_to_copy = [
        (TFLITE_PATH, "model.tflite"),
        (LABELS_PATH, "labels.txt"),
    ]
    if os.path.exists(MODEL_META_PATH):
        files_to_copy.append((MODEL_META_PATH, "model_meta.json"))
    if ship_stats:
        files_to_copy.append((CLASS_STATS_PATH, "class_stats.json"))
    if os.path.exists(DISEASES_JSON_PATH):
        files_to_copy.append((DISEASES_JSON_PATH, "diseases.json"))

    for src, dst_name in files_to_copy:
        if os.path.exists(src):
            dst = os.path.join(assets_dir, dst_name)
            shutil.copy2(src, dst)
            print(f"    -> Copied {dst_name} to assets/")

    stale = os.path.join(assets_dir, "class_stats.json")
    if not ship_stats and os.path.exists(stale):
        os.remove(stale)
        print("    -> Removed stale class_stats.json from assets/ (it does not match this model)")


def main():
    parser = argparse.ArgumentParser(description="Convert PaddyCare PyTorch model to TFLite")
    parser.add_argument(
        "--quantize", action="store_true",
        help="Apply dynamic-range quantization (int8 weights, float activations) to reduce model size"
    )
    parser.add_argument("--model", default=PTH_PATH, help="Checkpoint to export")
    parser.add_argument("--require-stats", action="store_true",
                        help="Fail instead of warn when matching class_stats.json (OOD stats) is missing")
    parser.add_argument("--eval-per-class", type=int, default=25,
                        help="Real test images per class used to measure TFLite accuracy (0 = all)")
    parser.add_argument("--parity-tol", type=float, default=DEFAULT_PARITY_TOL,
                        help="Max allowed |PyTorch - TFLite| output difference for float32 exports")
    args = parser.parse_args()

    print("=" * 60)
    print("  PaddyCare AI — Model Export Pipeline")
    print("  PyTorch (.pth) → ONNX (.onnx) → TFLite (.tflite)")
    print("=" * 60)

    model, class_names, fingerprint = step1_load_pytorch(args.model)
    validate_diseases_json(class_names)
    ood_enabled = check_ood_stats(fingerprint, args.require_stats)
    step2_export_onnx(model)
    quantized = step3_convert_tflite(quantize=args.quantize)
    tflite_info = step4_validate_tflite(model, class_names, quantized, args.eval_per_class, args.parity_tol)

    # Only after the model passed validation do we write metadata and sync assets.
    write_model_meta(class_names, fingerprint, ood_enabled, tflite_info, quantized)
    create_labels_file(class_names)
    copy_assets_to_android(ship_stats=ood_enabled)

    print("\n" + "=" * 60)
    print("   EXPORT COMPLETE!")
    print(f"  TFLite model: {TFLITE_PATH}")
    print(f"  Labels file : {LABELS_PATH}")
    if not ood_enabled:
        print("  NOTE: shipped WITHOUT OOD stats - the app cannot reject non-leaf images via Mahalanobis.")
    print("=" * 60)


if __name__ == "__main__":
    main()
