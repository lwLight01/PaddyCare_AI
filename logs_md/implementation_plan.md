# 5-Layer OOD Rejection System

Add a "Not Leaf" reject class (Layer 2) and Mahalanobis distance detection (Layer 3) to PaddyCare AI's inference pipeline — both Python and Android.

## User Review Required

> [!IMPORTANT]
> The TFLite model will now output **two tensors** (logits + 128-dim embedding) instead of one. This means you **must re-export** the model after retraining. The old single-output `model.tflite` will not work with the updated Android app.

> [!WARNING]
> The "Not Leaf" class requires you to create a `dataset/train/Not Leaf/` folder with ~1000-2000 images of green/brown non-leaf objects **before retraining**. The code changes below prepare the infrastructure — you supply the images.

## Open Questions

None — design was agreed upon in the discussion. All thresholds are auto-calibrated from training data.

## Proposed Changes

### Python ML Pipeline

---

#### [MODIFY] [`config.py`](file:///d:/Model_final/paddy_ai/config.py)

- Add `"Not Leaf"` to `CLASS_NAMES` (alphabetical position: after "Healthy")
- Add `TREATMENTS` entry for "Not Leaf" (bilingual rejection message)
- Add `MAHALANOBIS_PERCENTILE = 99` constant (used to auto-calibrate threshold)
- Add `CLASS_STATS_PATH` pointing to `models/class_stats.json`

---

#### [MODIFY] [`model.py`](file:///d:/Model_final/paddy_ai/model.py)

- Add `extract_embedding()` function — uses a **forward hook** on `model.classifier[5]` (the SiLU after `Linear(512, 128)`) to capture the 128-dim embedding without modifying the model architecture
- Add `DualOutputWrapper(nn.Module)` class — wraps the trained model for TFLite export to output both `(logits, embedding)` from a single forward pass. Only used during export, not during training

```python
# The classifier structure for reference:
# [0] Dropout(0.4)
# [1] Linear(in_features, 512)
# [2] SiLU()
# [3] Dropout(0.3)
# [4] Linear(512, 128)
# [5] SiLU()          ← HOOK HERE (128-dim embedding)
# [6] Dropout(0.2)
# [7] Linear(128, NUM_CLASSES)  ← logits
```

---

#### [NEW] [`extract_embeddings.py`](file:///d:/Model_final/paddy_ai/extract_embeddings.py)

New script to run **after training** and **before export**. It:

1. Loads the trained model and full training dataset
2. Passes every training image through the model in eval mode
3. Extracts the 128-dim embedding for each image using the forward hook
4. Computes per-class **mean vectors** (one 128-dim vector per class)
5. Computes a **shared covariance matrix** (128×128) across all classes
6. Inverts the covariance matrix (for Mahalanobis distance formula)
7. Computes Mahalanobis distance for every training sample and picks the **99th percentile** as the auto-calibrated threshold
8. Saves everything to `models/class_stats.json`:
   ```json
   {
     "class_names": ["Bacterial Blight", "Healthy", "Not Leaf"],
     "embedding_dim": 128,
     "class_means": { "Bacterial Blight": [...], "Healthy": [...], "Not Leaf": [...] },
     "precision_matrix": [[...], ...],
     "threshold": 42.5
   }
   ```

Usage: `python extract_embeddings.py` (runs automatically, no args needed)

---

#### [MODIFY] [`predict.py`](file:///d:/Model_final/paddy_ai/predict.py)

Modify the `predict()` function to add two new gates:

- **After model inference (Step 2):** If top prediction is `"Not Leaf"` → return `status: "not_paddy"` with rejection message
- **New Step 3 (Mahalanobis):** Extract embedding, compute Mahalanobis distance to the predicted class mean. If distance > threshold → return `status: "not_paddy"`. This catches images the model is overconfident about but that are actually OOD
- Existing confidence/entropy gates become Step 4 and Step 5

Updated flow:
```
Step 1: HSV color gate             (existing, unchanged)
Step 2: Model inference            (existing)
  2a:   "Not Leaf" class check     (NEW)
Step 3: Mahalanobis distance check (NEW)
Step 4: Confidence ≥ 60%           (existing, renumbered)
Step 5: Entropy OOD check          (existing, renumbered)
```

---

#### [MODIFY] [`export_tflite.py`](file:///d:/Model_final/paddy_ai/export_tflite.py)

- In `step2_export_onnx()`: Wrap model in `DualOutputWrapper` before ONNX export so the TFLite model outputs both `logits` (shape `[1, NUM_CLASSES]`) and `embedding` (shape `[1, 128]`)
- In `main()`: After export, copy `models/class_stats.json` alongside `model.tflite` and `labels.txt`, and print updated instructions to copy 3 files to Android assets

---

#### [MODIFY] [`train.py`](file:///d:/Model_final/paddy_ai/train.py)

- At the end of `train()`, after saving the best model, automatically run embedding extraction (import and call `extract_embeddings.extract_and_save()`) so `class_stats.json` is always generated fresh after training

---

### Android App

---

#### [MODIFY] [`PaddyClassifier.kt`](file:///d:/Model_final/paddy_ai/PaddyCareAndroid/app/src/main/java/com/paddycare/ai/ml/PaddyClassifier.kt)

Major changes:
- Add `CLASS_STATS_FILE = "class_stats.json"` constant
- In `init {}`: Load class statistics (means, precision matrix, threshold) from `assets/class_stats.json` using `org.json.JSONObject`
- Change `classify()` to use `runForMultipleInputsOutputs()` instead of `run()` to get both logits and embedding outputs from the dual-output TFLite model
- Return a new `ClassifyResult` data class containing: `predictions: List<Pair<String, Float>>`, `embedding: FloatArray`
- Add `mahalanobisDistance(embedding, classMean)` function — computes `sqrt((x-μ)ᵀ Σ⁻¹ (x-μ))` using the loaded precision matrix
- Add `isOOD(embedding, predictedClass)` function — returns `true` if Mahalanobis distance to the predicted class exceeds the threshold

---

#### [MODIFY] [`HomeScreen.kt`](file:///d:/Model_final/paddy_ai/PaddyCareAndroid/app/src/main/java/com/paddycare/ai/ui/screens/HomeScreen.kt)

Update the `analyzeImage()` function:
- After classification, check if top prediction is `"Not Leaf"` → return `NotPaddy`
- After that, check `classifier.isOOD(embedding, topPrediction)` → return `NotPaddy`
- Existing confidence/entropy checks remain after these new checks

---

#### [MODIFY] [`DiseaseInfo.kt`](file:///d:/Model_final/paddy_ai/PaddyCareAndroid/app/src/main/java/com/paddycare/ai/data/DiseaseInfo.kt)

- Add `"Not Leaf"` to `DISEASE_BN` map: `"Not Leaf" to "ধানের পাতা নয়"`
- Add `"Not Leaf"` entries to both `TREATMENTS_BN` and `TREATMENTS_EN` with rejection messages (matching Python config)

---

## Verification Plan

### Automated Tests
- `python -c "from model import extract_embedding, DualOutputWrapper; print('OK')"` — verify imports
- `python extract_embeddings.py` — verify stats generation produces valid JSON
- `python predict.py <test_image>` — verify 5-layer pipeline runs end-to-end

### Manual Verification
1. Create a small `dataset/train/Not Leaf/` folder with ~10 test images
2. Run `python train.py` (short run, 2 epochs) to verify training works with 3 classes
3. Run `python extract_embeddings.py` to verify `class_stats.json` is generated
4. Run `python export_tflite.py` to verify dual-output TFLite export
5. Test predictions on: a paddy leaf, a green shirt, a banana leaf — verify correct rejection
6. Build Android app in Android Studio and verify compilation

Image captured
    │
    ▼
┌─────────────────────────────────────┐
│ Layer 1: HSV Color + Texture Gate   │  ← rejects obviously non-plant images
│          (existing, fast, free)     │
└──────────────┬──────────────────────┘
               │ passes
               ▼
┌─────────────────────────────────────┐
│ Layer 2: Model Inference            │  ← model now has a "Not Leaf" class
│          Top prediction = Not Leaf? │     so it can actively say "this isn't
│          → REJECT                   │     a paddy leaf"
└──────────────┬──────────────────────┘
               │ predicts a disease/healthy
               ▼
┌─────────────────────────────────────┐
│ Layer 3: Mahalanobis Distance       │  ← even if the model says "Bacterial
│          Embedding too far from     │     Blight 95%", if the embedding is
│          known leaf embeddings?     │     far from real BLB embeddings
│          → REJECT                   │     → it's a false positive → reject
└──────────────┬──────────────────────┘
               │ embedding is close to known data
               ▼
┌─────────────────────────────────────┐
│ Layer 4: Confidence ≥ 60%           │  ← existing gate
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│ Layer 5: Entropy OOD Check          │  ← existing gate
└──────────────┬──────────────────────┘
               │
               ▼
           ✅ RESULT
