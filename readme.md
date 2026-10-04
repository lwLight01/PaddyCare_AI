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
Organize your dataset inside the `dataset` folder as follows:
```text
dataset/
├── train/
│   ├── Bacterial Blight/
│   ├── Blast/
│   └── Healthy/
└── val/
    ├── Bacterial Blight/
    ├── Blast/
    └── Healthy/
```
*(Test set is optional)*

### Training
Run the training script. It will train the EfficientNet-B5 architecture (with a two-phase warmup + fine-tuning strategy) and save the best model to `models/model.pth`.
```bash
python train.py
```

### Evaluation
Evaluate the trained model against the validation set:
```bash
python evaluate.py
```

---

## 2. Convert Model to TensorFlow Lite

To run the model on an Android phone offline, it must be converted from PyTorch to TensorFlow Lite.

### Requirements
Install the ONNX and TensorFlow conversion libraries:
```bash
pip install onnx onnxruntime onnx2tf tensorflow flatbuffers
```

### Export Script
Run the export script. It will convert `model.pth` -> `model.onnx` -> `model.tflite` and generate a `labels.txt` file.
```bash
python export_tflite.py
```
*Note: You can use `python export_tflite.py --quantize` to shrink the model size using int8 quantization.*

The script outputs two files:
- `models/model.tflite`
- `models/labels.txt`

---

## 3. Native Android App Development

The `PaddyCareAndroid` folder contains the fully native Android app built with **Kotlin** and **Jetpack Compose**. It requires NO internet connection to scan leaves.

### Setup
1. Copy your newly exported TFLite files to the Android assets folder:
   - Copy `models/model.tflite` to `PaddyCareAndroid/app/src/main/assets/model.tflite`
   - Copy `models/labels.txt` to `PaddyCareAndroid/app/src/main/assets/labels.txt`
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
