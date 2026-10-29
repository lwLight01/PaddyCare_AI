import os
import random
import hashlib
import inspect
import logging
import torch
import numpy as np
from PIL import Image, ImageOps
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
import torchvision.transforms.functional as TF

from config import (
    TRAIN_DIR, VAL_DIR, TEST_DIR,
    IMAGE_SIZE, BATCH_SIZE, NUM_WORKERS, PIN_MEMORY,
    NORM_MEAN, NORM_STD, SEED, get_active_classes,
)


def exif_loader(path: str) -> Image.Image:

    with Image.open(path) as im:
        return ImageOps.exif_transpose(im).convert("RGB")


class PadToSquare:
    def __call__(self, img):
        w, h = img.size
        max_wh = max(w, h)
        hp = int((max_wh - w) / 2)
        vp = int((max_wh - h) / 2)
        padding = (hp, vp, max_wh - w - hp, max_wh - h - vp)
        return TF.pad(img, padding, 0, 'constant')

train_transform = transforms.Compose([
    PadToSquare(),
    transforms.RandomResizedCrop((IMAGE_SIZE, IMAGE_SIZE), scale=(0.7, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.2),
    transforms.RandomApply([transforms.RandomRotation(degrees=20)], p=0.3),
    transforms.RandomApply([transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.02)], p=0.5),
    transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0))], p=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=NORM_MEAN, std=NORM_STD),
    transforms.RandomErasing(p=0.1, scale=(0.02, 0.08)),  # mild cutout for regularisation
])

val_transform = transforms.Compose([
    PadToSquare(),
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=NORM_MEAN, std=NORM_STD),
])

_IMAGEFOLDER_HAS_ALLOW_EMPTY = "allow_empty" in inspect.signature(datasets.ImageFolder.__init__).parameters


class SafeImageFolder(datasets.ImageFolder):

    def __init__(self, root, transform=None, classes=None, loader=exif_loader, **kwargs):
        self._forced_classes = list(classes) if classes is not None else None
        if _IMAGEFOLDER_HAS_ALLOW_EMPTY:
            kwargs.setdefault("allow_empty", True)
        super().__init__(root, transform=transform, loader=loader, **kwargs)

    def find_classes(self, directory: str):
        if self._forced_classes is not None:
            classes = self._forced_classes
        else:
            classes = get_active_classes(directory)
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx


def subset_per_class(dataset: datasets.ImageFolder, max_per_class: int, seed: int = 0):
    rng = np.random.RandomState(seed)
    targets = np.array(dataset.targets)
    indices = []
    for c in range(len(dataset.classes)):
        idx = np.where(targets == c)[0]
        if len(idx) > max_per_class:
            idx = rng.choice(idx, max_per_class, replace=False)
        indices.extend(sorted(idx.tolist()))
    return Subset(dataset, indices), targets[indices]


def split_fingerprint(dataset) -> str:

    if isinstance(dataset, Subset):
        base, indices = dataset.dataset, list(dataset.indices)
    else:
        base, indices = dataset, range(len(dataset.samples))
    root = os.path.abspath(base.root)
    h = hashlib.sha1()
    for i in sorted(indices):
        path, label = base.samples[i]
        rel = os.path.relpath(path, root).replace(os.sep, "/")
        h.update(f"{rel}|{base.classes[label]}\n".encode("utf-8"))
    return h.hexdigest()[:16]


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_weighted_sampler(targets: np.ndarray, num_classes: int, generator: torch.Generator = None) -> WeightedRandomSampler:
    targets = np.asarray(targets)
    class_counts = np.maximum(np.bincount(targets, minlength=num_classes), 1)
    sample_weights = 1.0 / class_counts[targets]
    return WeightedRandomSampler(
        weights=torch.tensor(sample_weights, dtype=torch.float64),
        num_samples=len(targets),
        replacement=True,
        generator=generator,
    )


def get_dataloaders(train_dir=TRAIN_DIR, val_dir=VAL_DIR, test_dir=TEST_DIR, max_per_class: int = 0,
                    seed: int = SEED):
    train_dataset = SafeImageFolder(root=train_dir, transform=train_transform)
    if len(train_dataset.classes) == 0:
        raise ValueError(f"No valid image classes found in {train_dir}!")
    class_names = train_dataset.classes

    val_dataset = SafeImageFolder(root=val_dir, transform=val_transform, classes=class_names)
    if len(val_dataset) == 0:
        raise ValueError(f"No validation images found in {val_dir} for classes {class_names}!")

    present_val = set(val_dataset.targets)
    missing_val = [c for i, c in enumerate(class_names) if i not in present_val]
    if missing_val:
        logging.warning(f"Classes with NO validation images (metrics for them are undefined): {missing_val}")

    train_ds, train_targets = train_dataset, np.array(train_dataset.targets)
    val_ds = val_dataset
    if max_per_class > 0:
        train_ds, train_targets = subset_per_class(train_dataset, max_per_class)
        val_ds, _ = subset_per_class(val_dataset, max_per_class)

    sampler_gen = torch.Generator().manual_seed(seed)
    loader_gen = torch.Generator().manual_seed(seed + 1)
    sampler = get_weighted_sampler(train_targets, len(class_names), generator=sampler_gen)

    common = dict(num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY and torch.cuda.is_available(),
                  worker_init_fn=seed_worker)
    loaders = {
        "train": DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler, generator=loader_gen, **common),
        "val": DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, **common),
    }

    if os.path.isdir(test_dir):
        test_dataset = SafeImageFolder(root=test_dir, transform=val_transform, classes=class_names)
        if len(test_dataset) > 0:
            loaders["test"] = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, **common)

    logging.info(f"Classes found: {class_names}")
    logging.info(f"Train: {len(train_ds)} | Val: {len(val_ds)} samples")

    return loaders, class_names, train_dataset
