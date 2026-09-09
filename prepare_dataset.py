"""
Augmentation types (6 exactly, cycling):
  0 - Horizontal Flip
  1 - Vertical Flip
  2 - Gaussian Noise
  3 - Zoom (centre-crop then resize back)
  4 - Rotate (+/- 30 degrees)
  5 - Color Jitter (brightness / contrast / saturation)

"""

import os
import random
import shutil
import math
import numpy as np
from pathlib import Path
from PIL import Image, ImageEnhance

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

BASE_DIR    = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "dataset"
TRAIN_DIR   = DATASET_DIR / "train"
VAL_DIR     = DATASET_DIR / "val"
TEST_DIR    = DATASET_DIR / "test"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

BLAST_CLASS = "Blast"

VAL_RATIO  = 0.10
TEST_RATIO = 0.10

NUM_AUG_TYPES = 6


def list_images(folder: Path) -> list[Path]:
    if not folder.exists():
        return []
    return [
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    ]


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def count_per_class(split_dir: Path) -> dict:
    result = {}
    if not split_dir.exists():
        return result
    for cls_dir in sorted(split_dir.iterdir()):
        if cls_dir.is_dir():
            result[cls_dir.name] = list_images(cls_dir)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# 6 Augmentation helpers
# ─────────────────────────────────────────────────────────────────────────────

def _horizontal_flip(img: Image.Image) -> Image.Image:
    """Flip image left-right."""
    return img.transpose(Image.FLIP_LEFT_RIGHT)


def _vertical_flip(img: Image.Image) -> Image.Image:
    """Flip image top-bottom."""
    return img.transpose(Image.FLIP_TOP_BOTTOM)


def _gaussian_noise(img: Image.Image, std: float = 18.0) -> Image.Image:
    """Add Gaussian noise to every channel."""
    arr = np.array(img, dtype=np.float32)
    noise = np.random.normal(0, std, arr.shape).astype(np.float32)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def _zoom(img: Image.Image, factor: float = 0.80) -> Image.Image:
    """
    Zoom in by cropping the centre to (factor * W, factor * H)
    then resizing back to the original dimensions.
    """
    w, h = img.size
    cw = int(w * factor)
    ch = int(h * factor)
    left   = (w - cw) // 2
    top    = (h - ch) // 2
    right  = left + cw
    bottom = top  + ch
    return img.crop((left, top, right, bottom)).resize((w, h), Image.BILINEAR)


def _rotate(img: Image.Image, angle: float = 30.0) -> Image.Image:
    """
    Rotate by ±angle degrees.  The sign alternates so successive calls
    produce clockwise and counter-clockwise variants.
    """
    return img.rotate(angle, resample=Image.BILINEAR, expand=False)


def _color_jitter(img: Image.Image) -> Image.Image:
    """
    Randomly adjust brightness, contrast, and saturation
    (colour jitter).
    """
    img = ImageEnhance.Brightness(img).enhance(random.uniform(0.70, 1.40))
    img = ImageEnhance.Contrast(img).enhance(random.uniform(0.70, 1.40))
    img = ImageEnhance.Color(img).enhance(random.uniform(0.60, 1.50))
    return img


# Map index 0-5 to the 6 augmentation functions
_AUG_FNS = [
    _horizontal_flip,   # 0
    _vertical_flip,     # 1
    _gaussian_noise,    # 2
    _zoom,              # 3
    _rotate,            # 4
    _color_jitter,      # 5
]

AUG_NAMES = [
    "Horizontal Flip",
    "Vertical Flip",
    "Gaussian Noise",
    "Zoom",
    "Rotate",
    "Color Jitter",
]


def augment_image(src_path: Path, dst_path: Path, aug_index: int):
    """
    Apply one of the 6 augmentation types (selected by aug_index % 6)
    and save the result.

    For Rotate, the angle alternates between +30 and -30 to ensure
    both directions are generated.
    """
    img = Image.open(src_path).convert("RGB")
    op  = aug_index % NUM_AUG_TYPES

    if op == 4:
        # Alternate sign for rotation
        angle = 30.0 if (aug_index // NUM_AUG_TYPES) % 2 == 0 else -30.0
        result = _rotate(img, angle)
    else:
        result = _AUG_FNS[op](img)

    result.save(dst_path, quality=92)


def fill_with_augmentation(images: list[Path], target: int, cls_dir: Path, class_name: str):
    """Augment images in-place inside cls_dir until count == target."""
    existing_count = len(images)
    if existing_count >= target:
        return
    needed = target - existing_count
    print(f"  [AUG] {class_name}: adding {needed} augmented images "
          f"(have {existing_count}, need {target})")

    src_pool  = images.copy()
    aug_count = 0
    cycle     = 0

    while aug_count < needed:
        random.shuffle(src_pool)
        for src in src_pool:
            if aug_count >= needed:
                break
            stem = src.stem
            ext  = src.suffix
            op   = (cycle * len(src_pool) + aug_count) % NUM_AUG_TYPES
            dst  = cls_dir / f"{stem}_aug_{AUG_NAMES[op].replace(' ','_')}_{aug_count}{ext}"
            augment_image(src, dst, aug_index=(cycle * len(src_pool) + aug_count))
            aug_count += 1
        cycle += 1

    # Summary of what was applied
    counts = {}
    for i in range(needed):
        name = AUG_NAMES[(i) % NUM_AUG_TYPES]
        counts[name] = counts.get(name, 0) + 1
    print(f"       Types used: { {k: v for k, v in counts.items()} }")


def delete_excess(images: list[Path], target: int, class_name: str):
    """Randomly delete images until count == target."""
    if len(images) <= target:
        return
    excess = len(images) - target
    print(f"  [DEL] {class_name}: removing {excess} images "
          f"(have {len(images)}, target {target})")
    to_delete = random.sample(images, excess)
    for p in to_delete:
        p.unlink()


def pool_and_resplit_blast():
    """Collect ALL blast images from all splits and re-split 80/10/10."""
    print("\n[BLAST] Pooling all Blast images from train / val / test ...")
    blast_train = TRAIN_DIR / BLAST_CLASS
    blast_val   = VAL_DIR   / BLAST_CLASS
    blast_test  = TEST_DIR  / BLAST_CLASS

    pool_dir = DATASET_DIR / "_blast_pool"
    ensure_dir(pool_dir)

    moved = 0
    for src_dir in [blast_train, blast_val, blast_test]:
        for img in list_images(src_dir):
            dst = pool_dir / img.name
            if dst.exists():
                stem = img.stem
                ext  = img.suffix
                dst  = pool_dir / f"{stem}_{src_dir.parent.name}{ext}"
            shutil.move(str(img), str(dst))
            moved += 1

    all_blast = list_images(pool_dir)
    random.shuffle(all_blast)
    total = len(all_blast)

    n_val   = max(1, math.floor(total * VAL_RATIO))
    n_test  = max(1, math.floor(total * TEST_RATIO))
    n_train = total - n_val - n_test

    splits = {
        "train": all_blast[:n_train],
        "val":   all_blast[n_train : n_train + n_val],
        "test":  all_blast[n_train + n_val :],
    }

    dst_dirs = {"train": blast_train, "val": blast_val, "test": blast_test}
    for split, imgs in splits.items():
        dst = dst_dirs[split]
        ensure_dir(dst)
        for img in imgs:
            shutil.move(str(img), str(dst / img.name))
        print(f"  [BLAST] {split}: {len(imgs)} images")

    shutil.rmtree(pool_dir)
    print(f"[BLAST] Done. Total pooled: {total}")
    return splits


def balance_split(split_dir: Path, split_name: str, target: int):
    """Delete excess or augment deficient classes in one split."""
    classes = count_per_class(split_dir)
    print(f"\n[{split_name.upper()}] Balancing {len(classes)} classes "
          f"to {target} images each ...")
    for cls_name, imgs in classes.items():
        cls_dir = split_dir / cls_name
        if len(imgs) > target:
            delete_excess(imgs, target, cls_name)
        elif len(imgs) < target:
            fill_with_augmentation(imgs, target, cls_dir, cls_name)
        else:
            print(f"  [OK]  {cls_name}: {len(imgs)} images (already at target)")


def main():
    print("=" * 60)
    print("PaddyCare - Augment Healthy Class (Train Split Only)")
    print("Augmentation Types: Horizontal Flip | Vertical Flip | "
          "Gaussian Noise | Zoom | Rotate | Color Jitter")
    print("=" * 60)

    # ── Count current state ───────────────────────────────────────
    bb_train   = list_images(TRAIN_DIR / "Bacterial Blight")
    hlt_train  = list_images(TRAIN_DIR / "Healthy")

    bb_count  = len(bb_train)
    hlt_count = len(hlt_train)

    print(f"\n[BEFORE]")
    print(f"  Bacterial Blight (train) : {bb_count}")
    print(f"  Healthy          (train) : {hlt_count}")

    if bb_count == 0:
        print("\n[ERROR] No Bacterial Blight images found in dataset/train/")
        return

    # ── Target: match Bacterial Blight count ─────────────────────
    target = bb_count
    print(f"\n  Target for Healthy       : {target}  (match Bacterial Blight)")

    if hlt_count >= target:
        print(f"\n[OK] Healthy already has {hlt_count} images — no augmentation needed.")
    else:
        fill_with_augmentation(
            images     = hlt_train,
            target     = target,
            cls_dir    = TRAIN_DIR / "Healthy",
            class_name = "Healthy",
        )

    # ── Final summary ─────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Final dataset counts:")
    for split_dir, name in [(TRAIN_DIR, "train"), (VAL_DIR, "val"), (TEST_DIR, "test")]:
        print(f"\n  [{name.upper()}]")
        for cls_name in sorted(["Bacterial Blight", "Healthy"]):
            cls_dir = split_dir / cls_name
            count   = len(list_images(cls_dir)) if cls_dir.exists() else 0
            print(f"    {cls_name:<22}: {count}")

    print("\n[DONE] Healthy class augmented. You can now run train.py")
    print("=" * 60)


if __name__ == "__main__":
    main()

