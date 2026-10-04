"""
export_tflite.py
================
Converts a trained PaddyCare AI PyTorch model to TensorFlow Lite format
for on-device inference in the Android app.

Pipeline:  model.pth  →  model.onnx  →  model.tflite

Usage:
    python export_tflite.py                     # default float32
    python export_tflite.py --quantize          # int8 quantization (~4x smaller)

Requirements:
    pip install onnx onnxruntime onnx2tf tensorflow flatbuffers
"""

import os
import sys
import shutil
import argparse
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import IMAGE_SIZE, MODEL_DIR, MODEL_PATH, CLASS_STATS_PATH, DISEASES_JSON_PATH
from model import build_model as _build_model, DualOutputWrapper

PTH_PATH    = MODEL_PATH
ONNX_PATH   = os.path.join(MODEL_DIR, "model.onnx")
TFLITE_DIR  = os.path.join(MODEL_DIR, "tflite_output")
TFLITE_PATH = os.path.join(MODEL_DIR, "model.tflite")
LABELS_PATH = os.path.join(MODEL_DIR, "labels.txt")


def step1_load_pytorch() -> tuple[nn.Module, list]:
    """Load the trained EfficientNet-B5 model from checkpoint dynamically."""
    print("\n[Step 1/4] Loading PyTorch model (EfficientNet-B5) ...")
    if not os.path.exists(PTH_PATH):
        print(f"❌ Model file not found: {PTH_PATH}")
        print("   Run 'python train.py' first to train the model.")
        sys.exit(1)

    checkpoint = torch.load(PTH_PATH, map_location="cpu", weights_only=True)
    state_dict = checkpoint["model_state_dict"]

    if "class_names" in checkpoint:
        class_names = checkpoint["class_names"]
        num_classes = len(class_names)
    elif "classifier.7.weight" in state_dict:
        num_classes = state_dict["classifier.7.weight"].shape[0]
        class_names = [f"Class_{i}" for i in range(num_classes)]
    else:
        num_classes = 2
        class_names = ["Bacterial Blight", "Healthy"]

    model = _build_model(num_classes=num_classes, pretrained=False)
    model.load_state_dict(state_dict)
    model.eval()

    epoch    = checkpoint.get("epoch", "?")
    val_acc  = checkpoint.get("val_acc", "?")
    backbone = checkpoint.get("backbone", "efficientnet_b5")

    print(f"  Backbone : {backbone}")
    print(f"  Input    : {IMAGE_SIZE}x{IMAGE_SIZE}")
    print(f"  Loaded model from epoch {epoch} (Val Acc: {val_acc})")
    print(f"  Classes  : {class_names} ({num_classes} total)")
    return model, class_names


def step2_export_onnx(model: nn.Module):
    """Export PyTorch model to ONNX format (Dual Output)."""
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
    print(f"  ✅ ONNX model saved: {ONNX_PATH} ({size_mb:.1f} MB)")

    # Validate ONNX model
    try:
        import onnx
        onnx_model = onnx.load(ONNX_PATH)
        onnx.checker.check_model(onnx_model)
        print("  ✅ ONNX model validation passed")
    except ImportError:
        print("  ⚠️ onnx package not installed, skipping ONNX validation check.")


def step3_convert_tflite(quantize: bool = False):
    """Convert ONNX model to TensorFlow Lite, optionally with INT8 quantization."""
    print("\n[Step 3/4] Converting ONNX → TFLite ...")

    try:
        import onnx2tf
    except ImportError:
        print(
            "❌ onnx2tf not installed. "
            "Run: pip install onnx2tf tensorflow"
        )
        sys.exit(1)

    os.makedirs(TFLITE_DIR, exist_ok=True)

    # Convert ONNX → TensorFlow/TFLite
    onnx2tf.convert(
        input_onnx_file_path=ONNX_PATH,
        output_folder_path=TFLITE_DIR,
        non_verbose=True,
    )

    generated_tflite = os.path.join(TFLITE_DIR, "model_float32.tflite")
    if not os.path.exists(generated_tflite):
        print(f"  ❌ Float32 TFLite model not found: {generated_tflite}")
        sys.exit(1)

    shutil.copy2(generated_tflite, TFLITE_PATH)
    size_mb = os.path.getsize(TFLITE_PATH) / (1024 * 1024)
    print(f"  ✅ TFLite model saved: {TFLITE_PATH} ({size_mb:.1f} MB (float32))")

    if quantize:
        print("  Applying dynamic-range INT8 quantization ...")
        try:
            import tensorflow as tf
            converter = tf.lite.TFLiteConverter.from_saved_model(TFLITE_DIR)
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            quantized_model = converter.convert()

            with open(TFLITE_PATH, "wb") as f:
                f.write(quantized_model)

            q_size_mb = os.path.getsize(TFLITE_PATH) / (1024 * 1024)
            print(f"  ✅ Quantized model saved: {TFLITE_PATH} ({q_size_mb:.1f} MB (int8))")
        except ImportError:
            print("  ⚠️ TensorFlow not installed, skipping quantization.")


def step4_validate_tflite(class_names: list):
    """Validate the TFLite model by running a dummy inference."""
    print("\n[Step 4/4] Validating TFLite model (Dual Output) ...")

    try:
        import tensorflow as tf
    except ImportError:
        print("  ⚠️ TensorFlow not installed, skipping validation.")
        return

    interpreter = tf.lite.Interpreter(model_path=TFLITE_PATH)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    input_shape = input_details[0]["shape"]
    input_dtype = input_details[0]["dtype"]

    print(f"  Input  shape: {input_shape}  dtype: {input_dtype}")
    for i, out_det in enumerate(output_details):
        print(f"  Output [{i}] shape: {out_det['shape']}  dtype: {out_det['dtype']}  name: {out_det['name']}")

    dummy_data = np.random.randn(*input_shape).astype(input_dtype)
    interpreter.set_tensor(input_details[0]["index"], dummy_data)
    interpreter.invoke()

    out_1 = interpreter.get_tensor(output_details[0]["index"])
    out_2 = interpreter.get_tensor(output_details[1]["index"])

    if out_1.shape[1] == len(class_names):
        logits = out_1
        embedding = out_2
    else:
        logits = out_2
        embedding = out_1

    logits = logits.astype(np.float32)
    exp_output = np.exp(logits[0] - np.max(logits[0]))
    probs = exp_output / exp_output.sum()

    print("  Dummy inference output (softmax):")
    for i, (name, prob) in enumerate(zip(class_names, probs)):
        print(f"    [{i}] {name}: {prob * 100:.2f}%")

    print(f"  Embedding output shape: {embedding.shape}")
    print("  ✅ TFLite model works correctly!")


def create_labels_file(class_names: list):
    """Create labels.txt matching the model checkpoint classes."""
    with open(LABELS_PATH, "w", encoding="utf-8") as f:
        for name in class_names:
            f.write(name + "\n")
    print(f"\n  ✅ Labels file saved: {LABELS_PATH} ({len(class_names)} classes)")


def copy_assets_to_android():
    """Copies exported files to PaddyCareAndroid assets directory if available."""
    assets_dir = os.path.join(os.path.dirname(MODEL_DIR), "PaddyCareAndroid", "app", "src", "main", "assets")
    if not os.path.isdir(assets_dir):
        return

    print("\n  📦 Syncing assets to Android project...")
    files_to_copy = [
        (TFLITE_PATH, "model.tflite"),
        (LABELS_PATH, "labels.txt"),
    ]
    if os.path.exists(CLASS_STATS_PATH):
        files_to_copy.append((CLASS_STATS_PATH, "class_stats.json"))
    if os.path.exists(DISEASES_JSON_PATH):
        files_to_copy.append((DISEASES_JSON_PATH, "diseases.json"))

    for src, dst_name in files_to_copy:
        if os.path.exists(src):
            dst = os.path.join(assets_dir, dst_name)
            shutil.copy2(src, dst)
            print(f"    -> Copied {dst_name} to assets/")


def main():
    parser = argparse.ArgumentParser(description="Convert PaddyCare PyTorch model to TFLite")
    parser.add_argument(
        "--quantize", action="store_true",
        help="Apply dynamic-range quantization to reduce model size"
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  PaddyCare AI — Model Export Pipeline")
    print("  PyTorch (.pth) → ONNX (.onnx) → TFLite (.tflite)")
    print("=" * 60)

    model, class_names = step1_load_pytorch()
    step2_export_onnx(model)
    step3_convert_tflite(quantize=args.quantize)
    step4_validate_tflite(class_names)
    create_labels_file(class_names)
    copy_assets_to_android()

    print("\n" + "=" * 60)
    print("  ✅ EXPORT COMPLETE!")
    print(f"  TFLite model: {TFLITE_PATH}")
    print(f"  Labels file : {LABELS_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    main()
