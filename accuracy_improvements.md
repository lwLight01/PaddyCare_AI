# Recommendations for Improving Model Accuracy

I've reviewed your current training pipeline (`train.py`), model architecture (`model.py`), and data loaders (`dataset_utils.py`). You have a solid foundation with **EfficientNet-B5** and two-phase transfer learning! Below are the original recommendations, with their current implementation status.

---

### 1. Optimize Data Augmentations (High Impact, Easy)
Your current augmentations in `dataset_utils.py` are very aggressive (heavy ColorJitter, Perspective, Affine, and Random Erasing). If your dataset isn't massive, the model might be struggling to learn because the images are too distorted (causing underfitting).
* **Suggestion**: Tame the custom augmentations and instead use PyTorch's built-in **`RandAugment`** or **`TrivialAugmentWide`**. These are state-of-the-art automated augmentation strategies that apply the perfect amount of distortion without destroying the image features.

### 2. Switch to a Better Optimizer (Moderate Impact, Easy)
You are currently using the standard `Adam` optimizer. For image classification tasks with Convolutional Neural Networks, weight decay doesn't behave perfectly in standard Adam.
* **Suggestion**: Switch to **`AdamW`** (`torch.optim.AdamW`). It decouples weight decay from the gradient update, which consistently leads to better generalization and higher accuracy on validation sets.

### 3. ✅ Upgrade the Model Architecture (High Impact, Moderate Effort) — DONE
Originally `MobileNetV2` was used. It has since been upgraded to **`EfficientNet-B5`**, which provides significantly higher accuracy (~30M params, 456×456 native resolution) while remaining deployable to Android via INT8 TFLite quantization.

### 4. ✅ Implement Two-Phase Transfer Learning (High Impact, Moderate Effort) — DONE
A two-phase training strategy is now implemented:
  * **Phase 1 (Warmup, 10 epochs):** The entire backbone is frozen; only the custom classifier head trains.
  * **Phase 2 (Fine-tuning):** All layers are unfrozen and trained end-to-end with a smaller learning rate.

### 5. ✅ Gradient Accumulation for Larger Effective Batch Sizes (Moderate Impact, Moderate Effort) — DONE
Gradient accumulation is now enabled in `config.py` with `ACCUM_STEPS = 4`. With `BATCH_SIZE = 8`, the effective batch size is **32** (8 × 4), providing smoother convergence without extra GPU memory usage.

### 6. Address Class Imbalance (High Impact, Depends on Dataset)
In agricultural datasets, you usually have significantly more "Healthy" leaves than specific diseases like "Blight". If the model sees mostly healthy leaves, it will become biased and guess "Healthy" too often to inflate its accuracy.
* **Suggestion**: Calculate the class weights (inverse frequency of each class in your dataset) and pass them into your `CrossEntropyLoss(weight=class_weights)`. Alternatively, use a `WeightedRandomSampler` in your DataLoader to ensure every batch has an equal number of healthy and diseased leaves.

---

**Remaining suggestions** (1, 2, and 6 above) can still be implemented for further accuracy gains. The three highest-impact items (architecture upgrade, two-phase training, and gradient accumulation) are already in place.
