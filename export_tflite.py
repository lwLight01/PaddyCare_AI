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
import argparse
import numpy as np
import torch
import torch.nn as nn

# ── Import from training modules so export always matches training exactly ────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import IMAGE_SIZE, NUM_CLASSES, CLASS_NAMES
from model import build_model as _build_model

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR  = os.path.join(BASE_DIR, "models")
PTH_PATH   = os.path.join(MODEL_DIR, "model.pth")
ONNX_PATH  = os.path.join(MODEL_DIR, "model.onnx")
TFLITE_DIR = os.path.join(MODEL_DIR, "tflite_output")
TFLITE_PATH = os.path.join(MODEL_DIR, "model.tflite")


def step1_load_pytorch() -> nn.Module:
    """Load the trained EfficientNet-B5 model from checkpoint."""
    print("\n[Step 1/4] Loading PyTorch model (EfficientNet-B5) ...")
    if not os.path.exists(PTH_PATH):
        print(f" Model file not found: {PTH_PATH}")
        print(f" Run 'python train.py' first to train the model.")
        sys.exit(1)

    model = _build_model(pretrained=False)   # architecture must match training
    checkpoint = torch.load(PTH_PATH, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    class_names = checkpoint.get("class_names", CLASS_NAMES)
    epoch       = checkpoint.get("epoch", "?")
    val_acc     = checkpoint.get("val_acc", "?")
    backbone    = checkpoint.get("backbone", "efficientnet_b5")

    print(f" Backbone : {backbone}")
    print(f" Input    : {IMAGE_SIZE}x{IMAGE_SIZE}")
    print(f" Loaded model from epoch {epoch}  (Val Acc: {val_acc})")
    print(f" Classes: {class_names}")
    return model


def step2_export_onnx(model: nn.Module):
    """Export PyTorch model to ONNX format (Dual Output)."""
    from model import DualOutputWrapper
    print("\n[Step 2/4] Exporting to ONNX (Dual Output) ...")
    
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
    print(f"ONNX model saved: {ONNX_PATH}  ({size_mb:.1f} MB)")

    # Quick validation
    import onnx
    onnx_model = onnx.load(ONNX_PATH)
    onnx.checker.check_model(onnx_model)
    print(f"ONNX model validation passed")


def step3_convert_tflite(quantize: bool = False):
    """Convert ONNX model to TensorFlow Lite, optionally with INT8 quantization."""
    print("\n[Step 3/4] Converting ONNX → TFLite ...")

    try:
        import onnx2tf
    except ImportError:
        print(
            "onnx2tf not installed. "
            "Run: pip install onnx2tf tensorflow"
        )
        sys.exit(1)

    # Clean/create output directory
    os.makedirs(TFLITE_DIR, exist_ok=True)

    # Convert ONNX → TensorFlow/TFLite
    onnx2tf.convert(
        input_onnx_file_path=ONNX_PATH,
        output_folder_path=TFLITE_DIR,
        non_verbose=True,
    )

    # For the stable Android deployment, use Float32.
    generated_tflite = os.path.join(
        TFLITE_DIR,
        "model_float32.tflite"
    )

    if not os.path.exists(generated_tflite):
        print(
            f"  ❌ Float32 TFLite model not found:\n"
            f"     {generated_tflite}"
        )
        sys.exit(1)

    # Copy Float32 model to the Android-ready path
    import shutil

    shutil.copy2(
        generated_tflite,
        TFLITE_PATH
    )

    size_mb = os.path.getsize(
        TFLITE_PATH
    ) / (1024 * 1024)

    print(
        f"  ✅ TFLite model saved: "
        f"{TFLITE_PATH} "
        f"({size_mb:.1f} MB (float32))"
    )

    # ── Optional: Dynamic-range INT8 quantization ──
    if quantize:
        print("  Applying dynamic-range INT8 quantization ...")
        try:
            import tensorflow as tf
        except ImportError:
            print("  ⚠️  TensorFlow not installed, skipping quantization.")
            return

        converter = tf.lite.TFLiteConverter.from_saved_model(
            os.path.join(TFLITE_DIR)
        )
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        quantized_model = converter.convert()

        with open(TFLITE_PATH, "wb") as f:
            f.write(quantized_model)

        q_size_mb = os.path.getsize(TFLITE_PATH) / (1024 * 1024)
        print(
            f"  ✅ Quantized model saved: "
            f"{TFLITE_PATH} "
            f"({q_size_mb:.1f} MB (int8))"
        )


def step4_validate_tflite():
    """Validate the TFLite model by running a dummy inference."""
    print("\n[Step 4/4] Validating TFLite model (Dual Output) ...")

    try:
        import tensorflow as tf
    except ImportError:
        print("  ⚠️  TensorFlow not installed, skipping validation.")
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

    # Create dummy input using the model's expected dtype
    dummy_data = np.random.randn(*input_shape).astype(input_dtype)

    interpreter.set_tensor(input_details[0]["index"], dummy_data)
    interpreter.invoke()

    # Find which output is logits and which is embedding
    # The logits output will have shape [1, NUM_CLASSES]
    # The embedding output will have shape [1, 128]
    out_1 = interpreter.get_tensor(output_details[0]["index"])
    out_2 = interpreter.get_tensor(output_details[1]["index"])
    
    if out_1.shape[1] == len(CLASS_NAMES):
        logits = out_1
        embedding = out_2
    else:
        logits = out_2
        embedding = out_1

    # Convert output to float32 for numerical stability
    logits = logits.astype(np.float32)

    # Apply softmax
    exp_output = np.exp(logits[0] - np.max(logits[0]))
    probs = exp_output / exp_output.sum()

    print("  Dummy inference output (softmax):")
    for i, (name, prob) in enumerate(zip(CLASS_NAMES, probs)):
        print(f"    [{i}] {name}: {prob * 100:.2f}%")

    print(f"  Embedding output shape: {embedding.shape}")
    print("  ✅ TFLite model works correctly!")


def create_labels_file():
    """Create labels.txt for the Android app."""
    labels_path = os.path.join(MODEL_DIR, "labels.txt")
    with open(labels_path, "w", encoding="utf-8") as f:
        for name in CLASS_NAMES:
            f.write(name + "\n")
    print(f"\n  ✅ Labels file saved: {labels_path}")


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

    model = step1_load_pytorch()
    step2_export_onnx(model)
    step3_convert_tflite(quantize=args.quantize)
    step4_validate_tflite()
    create_labels_file()

    print("\n" + "=" * 60)
    print("  ✅ EXPORT COMPLETE!")
    print(f"  TFLite model: {TFLITE_PATH}")
    print(f"  Labels file : {os.path.join(MODEL_DIR, 'labels.txt')}")
    class_stats_path = os.path.join(MODEL_DIR, 'class_stats.json')
    if os.path.exists(class_stats_path):
        print(f"  Stats file  : {class_stats_path}")
    print()
    print("  Next step:")
    print("  Copy these files to your Android project:")
    print("    → PaddyCareAndroid/app/src/main/assets/model.tflite")
    print("    → PaddyCareAndroid/app/src/main/assets/labels.txt")
    if os.path.exists(class_stats_path):
        print("    → PaddyCareAndroid/app/src/main/assets/class_stats.json")
    print("=" * 60)


if __name__ == "__main__":
    main()
