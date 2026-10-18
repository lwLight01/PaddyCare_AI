import os
import logging
import torch
import numpy as np
from PIL import ImageFile
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
import torchvision.transforms.functional as TF

ImageFile.LOAD_TRUNCATED_IMAGES = True
from config import (
    TRAIN_DIR, VAL_DIR, TEST_DIR,
    IMAGE_SIZE, BATCH_SIZE, NUM_WORKERS, PIN_MEMORY, VALID_EXTENSIONS,
    NORM_MEAN, NORM_STD,
)


class PadToSquare:
    """Pads the shorter side of an image with black pixels to make it square, preserving aspect ratio."""
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
    transforms.TrivialAugmentWide(),          # state-of-the-art auto-augmentation
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


class SafeImageFolder(datasets.ImageFolder):
    """
    ImageFolder that
      1. ignores class sub-directories that contain no images (so empty placeholder
         folders never crash training), and
      2. can be forced to use a fixed class list (``classes=``). Val/test sets must
         use the TRAIN class list so label indices always agree, even when a newly
         added class has no val/test images yet.
    """
    def __init__(self, root, transform=None, classes=None, **kwargs):
        self._forced_classes = list(classes) if classes is not None else None
        # Newer torchvision raises on classes with zero images unless allow_empty=True.
        try:
            super().__init__(root, transform=transform, allow_empty=True, **kwargs)
        except TypeError:
            super().__init__(root, transform=transform, **kwargs)

    def find_classes(self, directory: str):
        if self._forced_classes is not None:
            classes = self._forced_classes
        else:
            classes = []
            if os.path.isdir(directory):
                for entry in os.scandir(directory):
                    if entry.is_dir() and any(
                        f.name.lower().endswith(VALID_EXTENSIONS)
                        for f in os.scandir(entry.path) if f.is_file()
                    ):
                        classes.append(entry.name)
            classes.sort()
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx


def subset_per_class(dataset: datasets.ImageFolder, max_per_class: int, seed: int = 0):
    """Return (Subset, targets_of_subset) with at most `max_per_class` images per class."""
    rng = np.random.RandomState(seed)
    targets = np.array(dataset.targets)
    indices = []
    for c in range(len(dataset.classes)):
        idx = np.where(targets == c)[0]
        if len(idx) > max_per_class:
            idx = rng.choice(idx, max_per_class, replace=False)
        indices.extend(sorted(idx.tolist()))
    return Subset(dataset, indices), targets[indices]


def get_weighted_sampler(targets: np.ndarray, num_classes: int) -> WeightedRandomSampler:
    """WeightedRandomSampler so every training batch is class-balanced."""
    targets = np.asarray(targets)
    class_counts = np.maximum(np.bincount(targets, minlength=num_classes), 1)
    sample_weights = 1.0 / class_counts[targets]
    return WeightedRandomSampler(
        weights=torch.tensor(sample_weights, dtype=torch.float64),
        num_samples=len(targets),
        replacement=True,
    )


def get_dataloaders(train_dir=TRAIN_DIR, val_dir=VAL_DIR, test_dir=TEST_DIR, max_per_class: int = 0):
    """
    Returns (loaders, class_names, train_dataset).
    `max_per_class > 0` limits each split to that many images per class (smoke tests).
    """
    train_dataset = SafeImageFolder(root=train_dir, transform=train_transform)
    if len(train_dataset.classes) == 0:
        raise ValueError(f"No valid image classes found in {train_dir}!")
    class_names = train_dataset.classes

    # Val / test are forced onto the TRAIN class list => indices always agree.
    val_dataset = SafeImageFolder(root=val_dir, transform=val_transform, classes=class_names)
    if len(val_dataset) == 0:
        raise ValueError(f"No validation images found in {val_dir} for classes {class_names}!")

    missing_val = [c for i, c in enumerate(class_names) if i not in set(val_dataset.targets)]
    if missing_val:
        logging.warning(f"Classes with NO validation images (metrics for them are undefined): {missing_val}")

    train_ds, train_targets = train_dataset, np.array(train_dataset.targets)
    val_ds = val_dataset
    if max_per_class > 0:
        train_ds, train_targets = subset_per_class(train_dataset, max_per_class)
        val_ds, _ = subset_per_class(val_dataset, max_per_class)

    sampler = get_weighted_sampler(train_targets, len(class_names))

    loaders = {
        "train": DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler,
                            num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY),
        "val": DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY),
    }

    if os.path.isdir(test_dir):
        test_dataset = SafeImageFolder(root=test_dir, transform=val_transform, classes=class_names)
        if len(test_dataset) > 0:
            loaders["test"] = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False,
                                         num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)

    logging.info(f"Classes found: {class_names}")
    logging.info(f"Train: {len(train_ds)} | Val: {len(val_ds)} samples")
    print(f"[dataset] Classes   : {class_names}")
    print(f"[dataset] Train size: {len(train_ds)}")
    print(f"[dataset] Val size  : {len(val_ds)}")

    return loaders, class_names, train_dataset
