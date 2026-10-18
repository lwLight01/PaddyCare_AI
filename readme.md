# PaddyCare AI 🌾

PaddyCare AI is a comprehensive solution for detecting paddy leaf diseases (Bacterial Blight and Healthy) using Deep Learning (PyTorch & TensorFlow Lite) and a Native Android App (Kotlin & Jetpack Compose).

This repository contains:
1. The **PyTorch** training pipeline to train a custom **EfficientNet-B5** model with two-phase transfer learning.
2. The conversion pipeline from PyTorch `.pth` to **TensorFlow Lite** `.tflite`.
3. The **Native Android App** code running completely offline on the user's phone.

---

## 1. Train the PyTorch Model

### Prerequisites
Make sure you have Python installed, then install the required dependencies:
```bash
pip install -r requirements.txt
```

### Dataset Structure
Organize your dataset inside the `dataset` folder:
```text
dataset/
├── train/
│   ├── Bacterial Blight/
│   └── Healthy/
├── val/
│   ├── Bacterial Blight/
│   └── Healthy/
└── test/
    ├── Bacterial Blight/
    └── Healthy/
```
*(Any additional classes placed into `train/` will be discovered automatically).*

### Training
Run the training script (EfficientNet-B5 with two-phase warmup + differential fine-tuning):
```bash
python train.py
```
Options:
- `python train.py --smoke` : Fast dry-run without overwriting the production model.
- `python train.py --subset 100` : Limit to 100 images per class for rapid testing.
- `python train.py --epochs 25 --warmup 8` : Custom epoch scheduling.
- `python train.py --no-stats` : Skip automatic OOD covariance extraction.

### Evaluation
Evaluate the trained model against the held-out test split:
```bash
python evaluate.py
```
Options:
- `python evaluate.py --model models/model.pth` : Specify checkpoint.

### OOD Calibration (Mahalanobis Distance)
Compute class embedding statistics (Ledoit-Wolf shrinkage covariance) for on-device out-of-distribution leaf detection:
```bash
python extract_embeddings.py
```
Options:
- `python extract_embeddings.py --train-per-class 600 --val-per-class 150 --test-per-class 50`

---

## 2. Convert Model to TensorFlow Lite

To run the model on an Android phone offline, convert it from PyTorch to TensorFlow Lite.

### Requirements
Install the ONNX and TensorFlow conversion libraries:
```bash
pip install onnx onnxruntime onnx2tf tensorflow flatbuffers
```

### Export Script
Run the export script. It will validate `diseases.json`, export `model_meta.json`, convert `model.pth` -> `model.onnx` -> `model.tflite` (dual outputs: logits + 128-dim embeddings), and auto-sync all required files directly to the Android app's assets folder:
```bash
python export_tflite.py
```
*Note: You can use `python export_tflite.py --quantize` to shrink the model size using int8 quantization.*

The export script outputs and syncs:
- `models/model.tflite` -> `PaddyCareAndroid/app/src/main/assets/model.tflite`
- `models/labels.txt` -> `PaddyCareAndroid/app/src/main/assets/labels.txt`
- `models/model_meta.json` -> `PaddyCareAndroid/app/src/main/assets/model_meta.json`
- `diseases.json` -> `PaddyCareAndroid/app/src/main/assets/diseases.json`
- `models/class_stats.json` -> `PaddyCareAndroid/app/src/main/assets/class_stats.json` (when generated)

---

## 3. Native Android App Development

The `PaddyCareAndroid` folder contains the fully native Android app built with **Kotlin** and **Jetpack Compose**. It requires NO internet connection to scan leaves.

### Setup
1. Run `python export_tflite.py` (it will auto-copy the assets into `PaddyCareAndroid/app/src/main/assets/`).
2. Open **Android Studio** and select **Open Project**. Choose the `PaddyCareAndroid` folder.
3. Click **"Sync Project with Gradle Files"**.

### Tech Stack
- **UI:** Jetpack Compose (Material 3)
- **AI Inference:** TensorFlow Lite (on-device)
- **Image Processing:** CameraX, Coil
- **Local Storage:** Room Database (for Scan History)

### Run the App
Connect your Android phone via USB (with Developer Options / USB Debugging enabled) or use an Android Emulator, then click **Run 'app'** in Android Studio.

---

## 4. Publish to Google Play Store

We have included a detailed step-by-step guide to publishing the App. 
Please read the **[Publishing Guide](logs_md/android_PUBLISH_GUIDE.md)** for detailed instructions on:
1. Creating a Google Play Developer Account
2. Generating a Signed App Bundle (`.aab`) in Android Studio
3. Creating Store Listing Assets
4. Uploading to the Play Store Console
