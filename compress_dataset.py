"""
compress_dataset.py
────────────────────────────────────────────────────────────────────────
Re-compress all images in dataset/ to JPEG quality 85 % (keeping the
ORIGINAL pixel dimensions) and write them to dataset_compressed/.

Why quality 85?
  • Visually indistinguishable from the original at the 456-px training
    scale your model uses (EfficientNet-B5 native resolution).
  • Reduces camera-JPEG files (often 5-17 MB each) by 50-70 %.

Usage:
  python compress_dataset.py              # default quality = 85
  python compress_dataset.py --quality 90 # slightly larger files, even safer
  python compress_dataset.py --quality 80 # more aggressive compression
  python compress_dataset.py --dry-run    # only print stats, no files written
"""

import os
import sys
import argparse
import time
from pathlib import Path
from PIL import Image, ImageFile

# Allow truncated images to be opened (some camera JPEGs are edge-case imperfect)
ImageFile.LOAD_TRUNCATED_IMAGES = True


# ── Config ────────────────────────────────────────────────────────────────────
SRC_ROOT = Path(__file__).parent / "dataset"
DST_ROOT = Path(__file__).parent / "dataset_compressed"
SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}


def human_bytes(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if abs(n) < 1024.0:
            return f"{n:.1f} {unit}"
        n /= 1024.0
    return f"{n:.1f} TB"


def compress_image(src: Path, dst: Path, quality: int, dry_run: bool):
    """
    Re-save an image at the target JPEG quality.
    Returns (original_size, compressed_size).
    """
    orig_size = src.stat().st_size

    if dry_run:
        return orig_size, orig_size

    dst.parent.mkdir(parents=True, exist_ok=True)

    try:
        with Image.open(src) as img:
            # Convert palette/RGBA modes to RGB for JPEG compatibility
            if img.mode not in ("RGB",):
                img = img.convert("RGB")

            # Resize image to max 640x640, maintaining aspect ratio
            # Using LANCZOS for high quality downsampling
            img.thumbnail((640, 640), Image.Resampling.LANCZOS)

            # Always save as JPEG regardless of original format
            dst_jpg = dst.with_suffix(".jpg")
            img.save(
                dst_jpg,
                format="JPEG",
                quality=quality,
                optimize=True,
                progressive=True,
            )

        comp_size = dst_jpg.stat().st_size
        return orig_size, comp_size

    except Exception as exc:
        print(f"\n  WARNING: SKIPPED {src.name} — {exc}")
        return orig_size, orig_size


def run(quality: int, dry_run: bool) -> None:
    if not SRC_ROOT.exists():
        print(f"ERROR: Source dataset folder not found: {SRC_ROOT}")
        sys.exit(1)

    # Collect all image files
    all_files = [
        p for p in SRC_ROOT.rglob("*")
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS
    ]

    if not all_files:
        print("ERROR: No image files found in dataset/")
        sys.exit(1)

    total_files = len(all_files)
    print(f"\n{'='*60}")
    print(f"  PaddyCare Dataset Compressor")
    print(f"{'='*60}")
    print(f"  Source      : {SRC_ROOT}")
    print(f"  Destination : {DST_ROOT}")
    print(f"  JPEG quality: {quality}  (original resolution kept)")
    print(f"  Images found: {total_files:,}")
    print(f"  Dry-run     : {'YES - no files will be written' if dry_run else 'No'}")
    print(f"{'='*60}\n")

    total_orig = 0
    total_comp = 0
    start = time.time()

    for idx, src in enumerate(all_files, 1):
        rel = src.relative_to(SRC_ROOT)
        dst = DST_ROOT / rel

        orig_sz, comp_sz = compress_image(src, dst, quality, dry_run)
        total_orig += orig_sz
        total_comp += comp_sz

        pct = idx / total_files * 100
        bar_len = 30
        filled = int(bar_len * idx / total_files)
        bar = "=" * filled + "-" * (bar_len - filled)
        saved = orig_sz - comp_sz
        print(
            f"  [{bar}] {pct:5.1f}%  {idx:>5}/{total_files}  "
            f"{src.parent.name[:15]}/{src.name[:25]:<25}  "
            f"{human_bytes(orig_sz):>9} -> {human_bytes(comp_sz):>9}  "
            f"saved {human_bytes(saved):>8}",
            end="\r",
        )

    elapsed = time.time() - start
    saved_total = total_orig - total_comp
    pct_saved = saved_total / total_orig * 100 if total_orig else 0

    print("\n")
    print(f"{'='*60}")
    print(f"  Done in {elapsed:.1f}s")
    print(f"  Original size : {human_bytes(total_orig)}")
    print(f"  Compressed    : {human_bytes(total_comp)}")
    print(f"  Space saved   : {human_bytes(saved_total)}  ({pct_saved:.1f}%)")
    print(f"  Output folder : {DST_ROOT}")
    print(f"{'='*60}\n")

    if not dry_run:
        print("Next steps:")
        print("  1. Verify a few images look correct in dataset_compressed/")
        print("  2. In config.py, change DATASET_DIR to point to dataset_compressed")
        print("     OR rename dataset_compressed -> dataset after deleting old dataset")
        print("  3. Delete the original 'dataset/' folder to free disk space\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Re-compress PaddyCare dataset images to reduce disk usage."
    )
    parser.add_argument(
        "--quality",
        type=int,
        default=85,
        metavar="Q",
        help="JPEG quality (1-95). Default: 85.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Scan source only; do not write any output files.",
    )
    args = parser.parse_args()

    if not (1 <= args.quality <= 95):
        print("ERROR: Quality must be between 1 and 95.")
        sys.exit(1)

    run(quality=args.quality, dry_run=args.dry_run)
