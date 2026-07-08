"""
scripts/prepare_dataset.py
===========================
Prepares the TrashCan dataset for YOLO-DarkWater training.

Operations:
  1. Scans data/images/train_aug/ for all .jpg images.
  2. Performs an 85/15 stratified-by-class train/val split.
  3. Copies (not moves) images + labels to:
       data/images/train/
       data/images/val/
       data/labels/train/
       data/labels/val/
  4. Fixes data/dataset.yaml with absolute paths for this machine.
  5. Saves a training_config.json with split metadata.

Non-destructive: originals in train_aug/ are preserved.

Usage:
    python scripts/prepare_dataset.py
    python scripts/prepare_dataset.py --val-ratio 0.15 --seed 42
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import yaml


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).parent.parent
DATA_DIR = REPO_ROOT / "data"
IMAGES_DIR = DATA_DIR / "images"
LABELS_DIR = DATA_DIR / "labels"
YAML_PATH = DATA_DIR / "dataset.yaml"


def find_label_file(image_path: Path, labels_root: Path) -> Path | None:
    """Locate the label file corresponding to an image file."""
    # Try same directory structure under labels/
    rel = image_path.relative_to(IMAGES_DIR)
    label_path = labels_root / rel.with_suffix(".txt")
    if label_path.exists():
        return label_path

    # Try replacing 'images' with 'labels' in path
    parts = list(image_path.parts)
    for i, part in enumerate(parts):
        if part == "images":
            parts[i] = "labels"
            break
    alt_path = Path(*parts).with_suffix(".txt")
    if alt_path.exists():
        return alt_path

    return None


def get_primary_class(label_path: Path) -> int:
    """Return the class ID of the first annotation in the label file."""
    try:
        with open(label_path) as f:
            first_line = f.readline().strip()
            if first_line:
                return int(first_line.split()[0])
    except (OSError, ValueError, IndexError):
        pass
    return -1  # unknown


def stratified_split(
    image_paths: list[Path],
    label_paths: list[Path | None],
    val_ratio: float = 0.15,
    seed: int = 42,
) -> tuple[list[int], list[int]]:
    """
    Stratified train/val split by primary class.

    Returns:
        (train_indices, val_indices) — indices into image_paths.
    """
    random.seed(seed)

    # Group by primary class
    class_to_indices: dict[int, list[int]] = defaultdict(list)
    for i, lp in enumerate(label_paths):
        cls = get_primary_class(lp) if lp else -1
        class_to_indices[cls].append(i)

    train_idx, val_idx = [], []
    for cls, indices in class_to_indices.items():
        random.shuffle(indices)
        n_val = max(1, round(len(indices) * val_ratio))
        val_idx.extend(indices[:n_val])
        train_idx.extend(indices[n_val:])

    return sorted(train_idx), sorted(val_idx)


def copy_split(
    indices: list[int],
    image_paths: list[Path],
    label_paths: list[Path | None],
    dest_images: Path,
    dest_labels: Path,
) -> tuple[int, int]:
    """Copy image+label pairs to destination directories. Returns (n_copied, n_no_label)."""
    dest_images.mkdir(parents=True, exist_ok=True)
    dest_labels.mkdir(parents=True, exist_ok=True)

    copied, no_label = 0, 0
    for i in indices:
        img_src = image_paths[i]
        shutil.copy2(img_src, dest_images / img_src.name)

        lp = label_paths[i]
        if lp and lp.exists():
            shutil.copy2(lp, dest_labels / img_src.with_suffix(".txt").name)
        else:
            # Create empty label file for images with no annotations
            (dest_labels / img_src.with_suffix(".txt").name).touch()
            no_label += 1

        copied += 1

    return copied, no_label


def fix_dataset_yaml(train_dir: Path, val_dir: Path, test_dir: Path, nc: int, names: list[str]) -> None:
    """Rewrite dataset.yaml with absolute paths for this machine."""
    config = {
        "path": str(DATA_DIR.resolve()),
        "train": str(train_dir.relative_to(DATA_DIR.resolve())),
        "val": str(val_dir.relative_to(DATA_DIR.resolve())),
        "test": str(test_dir.relative_to(DATA_DIR.resolve())),
        "nc": nc,
        "names": names,
    }
    with open(YAML_PATH, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    print(f"  [OK] Updated {YAML_PATH}")


def save_split_config(
    train_images: list[str],
    val_images: list[str],
    val_ratio: float,
    seed: int,
    output_path: Path,
) -> None:
    """Save split metadata as JSON for reproducibility."""
    config = {
        "val_ratio": val_ratio,
        "seed": seed,
        "n_train": len(train_images),
        "n_val": len(val_images),
        "train_images": [Path(p).name for p in train_images[:20]],
        "val_images": [Path(p).name for p in val_images[:20]],
        "note": "First 20 filenames shown per split for reference.",
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"  [OK] Split metadata saved to {output_path}")


def main(val_ratio: float = 0.15, seed: int = 42, force: bool = False) -> None:
    print("\n" + "=" * 60)
    print("YOLO-DarkWater: Dataset Preparation")
    print("=" * 60)

    # ---- Source directories ----
    source_images = IMAGES_DIR / "train_aug"
    source_labels_candidates = [
        LABELS_DIR / "train_aug",
        LABELS_DIR / "train",
        DATA_DIR / "labels",
    ]

    if not source_images.exists():
        sys.exit(f"ERROR: Source images not found at {source_images}")

    # Find label root
    source_labels = None
    for candidate in source_labels_candidates:
        if candidate.exists():
            source_labels = candidate
            break

    if source_labels is None:
        print("WARNING: No label directory found. Labels will be empty.")

    # ---- Collect images ----
    image_paths = sorted(source_images.glob("*.jpg")) + sorted(source_images.glob("*.png"))
    if not image_paths:
        sys.exit(f"ERROR: No images found in {source_images}")
    print(f"\n  Found {len(image_paths)} images in {source_images}")

    # ---- Find corresponding labels ----
    label_paths = []
    for img in image_paths:
        lp = find_label_file(img, source_labels) if source_labels else None
        label_paths.append(lp)
    n_with_labels = sum(1 for lp in label_paths if lp is not None)
    print(f"  Found labels for {n_with_labels}/{len(image_paths)} images")

    # ---- Destination directories ----
    train_images_dir = IMAGES_DIR / "train"
    val_images_dir = IMAGES_DIR / "val"
    train_labels_dir = LABELS_DIR / "train"
    val_labels_dir = LABELS_DIR / "val"
    test_images_dir = IMAGES_DIR / "test"  # already exists

    # Check if already split
    if train_images_dir.exists() and not force:
        n_existing = len(list(train_images_dir.glob("*.jpg")))
        if n_existing > 0:
            print(f"\n  Train split already exists ({n_existing} images). Use --force to redo.")
            print("  Skipping split. Fixing dataset.yaml only.")
            _fix_yaml_only(train_images_dir, val_images_dir, test_images_dir)
            return

    # ---- Perform split ----
    print(f"\n  Performing {100*(1-val_ratio):.0f}/{100*val_ratio:.0f} train/val split (seed={seed})...")
    train_idx, val_idx = stratified_split(image_paths, label_paths, val_ratio, seed)
    print(f"  Train: {len(train_idx)} images | Val: {len(val_idx)} images")

    # ---- Copy files ----
    print("\n  Copying training images...")
    n_train, n_train_nolabel = copy_split(
        train_idx, image_paths, label_paths, train_images_dir, train_labels_dir
    )

    print("  Copying validation images...")
    n_val, n_val_nolabel = copy_split(
        val_idx, image_paths, label_paths, val_images_dir, val_labels_dir
    )

    print(f"  [OK] Train: {n_train} images ({n_train_nolabel} without labels)")
    print(f"  [OK] Val:   {n_val} images ({n_val_nolabel} without labels)")

    # ---- Fix dataset.yaml ----
    print("\n  Updating dataset.yaml with absolute paths...")
    # Parse existing YAML for class names
    try:
        with open(YAML_PATH) as f:
            existing = yaml.safe_load(f)
        nc = existing.get("nc", 16)
        names = existing.get("names", [f"class_{i}" for i in range(nc)])
    except Exception:
        nc = 16
        names = [f"class_{i}" for i in range(nc)]

    fix_dataset_yaml(train_images_dir, val_images_dir, test_images_dir, nc, names)

    # ---- Save split metadata ----
    save_split_config(
        [str(image_paths[i]) for i in train_idx],
        [str(image_paths[i]) for i in val_idx],
        val_ratio,
        seed,
        REPO_ROOT / "results" / "training_config.json",
    )

    print("\n" + "=" * 60)
    print("Dataset preparation complete!")
    print(f"  Train: {train_images_dir}")
    print(f"  Val:   {val_images_dir}")
    print(f"  Test:  {test_images_dir}")
    print("=" * 60)


def _fix_yaml_only(train_dir: Path, val_dir: Path, test_dir: Path) -> None:
    """Minimal YAML fix when split already exists."""
    try:
        with open(YAML_PATH) as f:
            existing = yaml.safe_load(f)
        nc = existing.get("nc", 16)
        names = existing.get("names", [f"class_{i}" for i in range(nc)])
    except Exception:
        nc, names = 16, [f"class_{i}" for i in range(16)]
    fix_dataset_yaml(train_dir, val_dir, test_dir, nc, names)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare dataset for YOLO-DarkWater training")
    parser.add_argument("--val-ratio", type=float, default=0.15, help="Validation set ratio (default: 0.15)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--force", action="store_true", help="Re-split even if train/ already exists")
    args = parser.parse_args()
    main(val_ratio=args.val_ratio, seed=args.seed, force=args.force)
