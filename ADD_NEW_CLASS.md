# 🌾 How to Add a New Disease Class — PaddyCare AI

Follow these steps **every time** you add a new paddy disease class.

---

## Step 1 — Prepare the Dataset

Create the new class folders with at least **100–200 images per split** (more = better):

```
dataset/
  train/
    <NewDiseaseName>/    ← add images here (80 % of total)
  val/
    <NewDiseaseName>/    ← add images here (10 % of total)
  test/
    <NewDiseaseName>/    ← add images here (10 % of total)
```

> **Folder name rules:**
> - Use the **exact same name** you will put in `CLASS_NAMES` in `config.py`
> - Capitalise consistently, e.g. `Brown Spot` not `brown_spot`
> - `ImageFolder` discovers classes by sorting folder names alphabetically

---

## Step 2 — Update `config.py`

Open `config.py` and update **two places**:

### 2a. `CLASS_NAMES` list — keep alphabetical order!

```python
CLASS_NAMES = [
    "Bacterial Blight",   # B
    "Blast",              # B  <- example new class
    "Brown Spot",         # B  <- example new class
    "Healthy",            # H
    # -- add new disease names here (alphabetical order!) --
]
```

> ⚠️ **Critical:** The order here must match the alphabetical folder sort order.
> Run `python -c "import os; print(sorted(os.listdir('dataset/train')))"` to verify.

### 2b. `TREATMENTS` dict — add bilingual treatment text

```python
"Brown Spot": {
    "en": (
        "1. Spray Mancozeb or Propiconazole fungicide.\n"
        "2. Apply balanced potassium and silicon fertilizer.\n"
        "3. Avoid water stress during the tillering stage.\n"
        "4. Use resistant varieties.\n"
        "5. Remove and burn infected plant debris."
    ),
    "bn": (
        "১. ম্যানকোজেব বা প্রোপিকোনাজোল ছত্রাকনাশক স্প্রে করুন।\n"
        "২. সুষম পটাশিয়াম ও সিলিকন সার প্রয়োগ করুন।\n"
        "৩. কুশি পর্যায়ে পানির চাপ এড়িয়ে চলুন।\n"
        "৪. প্রতিরোধী জাত ব্যবহার করুন।\n"
        "৫. আক্রান্ত গাছের অবশিষ্ট পুড়িয়ে ফেলুন।"
    ),
},
```

> The `OOD_ENTROPY_THRESHOLD` updates **automatically** — no action needed.

---

## Step 3 — Retrain the Model

Delete the old checkpoint first, then retrain:

```bash
del models\model.pth
python train.py
```

---

## Step 4 — Evaluate

```bash
python evaluate.py
```

Check per-class accuracy. If any class is worse, collect more images for it.

---

## Step 5 — Export to TFLite

```bash
# Recommended — INT8 quantized (~4x smaller)
python export_tflite.py --quantize

# Or float32 if you need maximum compatibility
python export_tflite.py
```

---

## Step 6 — Deploy to Android

```
models/model.tflite  ->  PaddyCareAndroid/app/src/main/assets/model.tflite
models/labels.txt    ->  PaddyCareAndroid/app/src/main/assets/labels.txt
```

---

## OOD Entropy Threshold Reference (auto-calculated)

| Classes (N) | OOD_ENTROPY_THRESHOLD |
|:-----------:|:---------------------:|
| 2           | 0.5199                |
| 4           | 1.0397                |
| 6           | 1.3451                |
| 8           | 1.5596                |
| 10          | 1.7269                |
| 12          | 1.9098                |

---

## Checklist

- [ ] Create dataset/train/<ClassName>/ with 100-200+ images
- [ ] Create dataset/val/<ClassName>/   with 10-20+ images
- [ ] Create dataset/test/<ClassName>/  with 10-20+ images
- [ ] Add <ClassName> to CLASS_NAMES in config.py (alphabetical order!)
- [ ] Add TREATMENTS[<ClassName>]["en"] and ["bn"] in config.py
- [ ] Delete old models/model.pth
- [ ] python train.py
- [ ] python evaluate.py
- [ ] python export_tflite.py --quantize
- [ ] Copy model.tflite + labels.txt to Android assets
- [ ] Build and test the Android app
