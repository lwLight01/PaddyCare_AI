import os
import logging
import torch
import numpy as np
from PIL import ImageFile
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, WeightedRandomSampler
import torchvision.transforms.functional as TF

ImageFile.LOAD_TRUNCATED_IMAGES = True
from config import (
    TRAIN_DIR, VAL_DIR, TEST_DIR,
    IMAGE_SIZE, BATCH_SIZE, NUM_WORKERS, PIN_MEMORY, VALID_EXTENSIONS
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
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
    transforms.RandomErasing(p=0.1, scale=(0.02, 0.08)),  # mild cutout for regularisation
])

val_transform = transforms.Compose([
    PadToSquare(),
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


class SafeImageFolder(datasets.ImageFolder):
    """
    Subclass of ImageFolder that automatically ignores empty class subdirectories.
    Prevents FileNotFoundError when empty placeholder class folders exist.
    """
    def find_classes(self, directory: str):
        classes = []
        if not os.path.isdir(directory):
            return [], {}

        for entry in os.scandir(directory):
            if entry.is_dir():
                has_images = any(
                    f.name.lower().endswith(VALID_EXTENSIONS)
                    for f in os.scandir(entry.path)
                    if f.is_file()
                )
                if has_images:
                    classes.append(entry.name)

        classes.sort()
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx


def get_class_weights(dataset: datasets.ImageFolder) -> torch.Tensor:
    """Compute inverse-frequency class weights."""
    targets = np.array(dataset.targets)
    class_counts = np.bincount(targets, minlength=len(dataset.classes))
    class_counts = np.maximum(class_counts, 1)          # avoid division by zero
    weights = 1.0 / class_counts
    weights = weights / weights.sum() * len(dataset.classes)  # normalise so mean ≈ 1
    return torch.tensor(weights, dtype=torch.float32)


def get_weighted_sampler(dataset: datasets.ImageFolder) -> WeightedRandomSampler:
    """Build a WeightedRandomSampler so every training batch is class-balanced."""
    targets = np.array(dataset.targets)
    class_counts = np.bincount(targets, minlength=len(dataset.classes))
    class_counts = np.maximum(class_counts, 1)
    sample_weights = 1.0 / class_counts[targets]       # per-sample weight
    sampler = WeightedRandomSampler(
        weights=torch.tensor(sample_weights, dtype=torch.float64),
        num_samples=len(dataset),
        replacement=True,
    )
    return sampler


def get_dataloaders(train_dir=TRAIN_DIR, val_dir=VAL_DIR, test_dir=TEST_DIR):
    train_dataset = SafeImageFolder(root=train_dir, transform=train_transform)
    val_dataset   = SafeImageFolder(root=val_dir,   transform=val_transform)

    if len(train_dataset.classes) == 0:
        raise ValueError(f"No valid image classes found in {train_dir}!")

    # WeightedRandomSampler replaces shuffle=True and balances class frequencies
    sampler = get_weighted_sampler(train_dataset)

    loaders = {
        "train": DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            sampler=sampler,
            num_workers=NUM_WORKERS,
            pin_memory=PIN_MEMORY,
        ),
        "val": DataLoader(
            val_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=PIN_MEMORY,
        ),
    }

    if os.path.isdir(test_dir):
        test_dataset = SafeImageFolder(root=test_dir, transform=val_transform)
        if len(test_dataset) > 0:
            loaders["test"] = DataLoader(
                test_dataset,
                batch_size=BATCH_SIZE,
                shuffle=False,
                num_workers=NUM_WORKERS,
                pin_memory=PIN_MEMORY,
            )

    class_names = train_dataset.classes
    logging.info(f"Classes found: {class_names}")
    logging.info(
        f"Train: {len(train_dataset)} | Val: {len(val_dataset)} samples"
    )
    print(f"[dataset] Classes   : {class_names}")
    print(f"[dataset] Train size: {len(train_dataset)}")
    print(f"[dataset] Val size  : {len(val_dataset)}")

    return loaders, class_names, train_dataset
