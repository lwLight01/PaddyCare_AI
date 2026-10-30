"""Verify dataset integrity: check for corrupt/truncated images and cross-split duplicate leakage."""
from __future__ import annotations

import argparse
import hashlib
import os
from collections import defaultdict
import numpy as np
from PIL import Image, ImageOps

from config import TRAIN_DIR, VAL_DIR, TEST_DIR, VALID_EXTENSIONS


def dhash_image(img: Image.Image, hash_size: int = 8) -> int:
    """Compute a 64-bit perceptual difference hash (dHash) using only Pillow + NumPy."""
    gray = ImageOps.exif_transpose(img).convert("L").resize(
        (hash_size + 1, hash_size), Image.Resampling.BILINEAR
    )
    pixels = np.asarray(gray, dtype=np.int16)
    diff = pixels[:, 1:] > pixels[:, :-1]
    bits = 0
    for b in diff.flatten():
        bits = (bits << 1) | int(b)
    return bits


def file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def iter_split_images(split_dir: str):
    if not os.path.isdir(split_dir):
        return
    for cls_name in sorted(os.listdir(split_dir)):
        cls_path = os.path.join(split_dir, cls_name)
        if not os.path.isdir(cls_path):
            continue
        for fname in sorted(os.listdir(cls_path)):
            if fname.lower().endswith(VALID_EXTENSIONS):
                yield cls_name, os.path.join(cls_path, fname)


def verify_dataset(
    splits: dict[str, str] | None = None,
    remove_corrupt: bool = False,
    remove_cross_split_duplicates: bool = False,
) -> dict:
    if splits is None:
        splits = {"train": TRAIN_DIR, "val": VAL_DIR, "test": TEST_DIR}

    corrupt_files: list[tuple[str, str]] = []
    exact_hashes: dict[str, list[tuple[str, str, str]]] = defaultdict(list)
    perceptual_hashes: dict[int, list[tuple[str, str, str]]] = defaultdict(list)
    counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for split_name, split_dir in splits.items():
        for cls_name, path in iter_split_images(split_dir) or []:
            try:
                with Image.open(path) as im:
                    im.verify()
                with Image.open(path) as im:
                    im.load()
                    ph = dhash_image(im)
                sha = file_sha256(path)
                counts[split_name][cls_name] += 1
                exact_hashes[sha].append((split_name, cls_name, path))
                perceptual_hashes[ph].append((split_name, cls_name, path))
            except Exception as e:
                corrupt_files.append((path, str(e)))

    cross_split_exact = {
        h: items for h, items in exact_hashes.items()
        if len({s for s, _, _ in items}) > 1
    }
    cross_split_phash = {
        h: items for h, items in perceptual_hashes.items()
        if len({s for s, _, _ in items}) > 1
    }

    removed = []
    if remove_corrupt:
        for path, _ in corrupt_files:
            if os.path.exists(path):
                os.remove(path)
                removed.append(path)

    if remove_cross_split_duplicates:
        for items in cross_split_phash.values():
            # Keep train copy if present, otherwise first split; remove copies in val/test
            items_sorted = sorted(items, key=lambda x: {"train": 0, "val": 1, "test": 2}.get(x[0], 3))
            for _, _, dup_path in items_sorted[1:]:
                if os.path.exists(dup_path):
                    os.remove(dup_path)
                    removed.append(dup_path)

    return {
        "counts": {k: dict(v) for k, v in counts.items()},
        "corrupt_files": corrupt_files,
        "cross_split_exact_groups": list(cross_split_exact.values()),
        "cross_split_phash_groups": list(cross_split_phash.values()),
        "removed_files": removed,
    }


def main():
    parser = argparse.ArgumentParser(description="Check dataset splits for corrupt files and leakage")
    parser.add_argument("--remove-corrupt", action="store_true", help="Delete unreadable/truncated images")
    parser.add_argument("--remove-cross-split-duplicates", action="store_true",
                        help="Delete val/test images that duplicate a train image")
    args = parser.parse_args()

    report = verify_dataset(
        remove_corrupt=args.remove_corrupt,
        remove_cross_split_duplicates=args.remove_cross_split_duplicates,
    )

    print("=" * 60)
    print("PaddyCare AI — Dataset Verification Report")
    print("=" * 60)
    for split, cls_map in report["counts"].items():
        total = sum(cls_map.values())
        print(f"  {split:5s}: {total:5d} images | {cls_map}")

    print(f"\nCorrupt / truncated files          : {len(report['corrupt_files'])}")
    for path, err in report["corrupt_files"][:10]:
        print(f"  [CORRUPT] {path} ({err})")

    print(f"Cross-split exact duplicate groups : {len(report['cross_split_exact_groups'])}")
    print(f"Cross-split dHash duplicate groups : {len(report['cross_split_phash_groups'])}")
    for grp in report["cross_split_phash_groups"][:10]:
        summary = ", ".join(f"{s}:{os.path.basename(p)}" for s, _, p in grp)
        print(f"  [LEAKAGE] {summary}")

    if report["removed_files"]:
        print(f"\nRemoved {len(report['removed_files'])} files.")
    print("=" * 60)


if __name__ == "__main__":
    main()

