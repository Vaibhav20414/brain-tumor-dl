"""
prepare_data.py — Download and split the brain MRI dataset.

Usage:
    python scripts/prepare_data.py --dataset br35h --output data/processed
    python scripts/prepare_data.py --dataset phase2 --output data/processed

Supports:
  - br35h  : Binary detection dataset (yes/no folders)
  - phase2 : Multi-class dataset (glioma/meningioma/no_tumor/pituitary folders)

The script assumes the raw data has already been placed in data/raw/.
It creates stratified train/val/test splits (70/15/15) under the output dir.
"""

import argparse
import random
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

from sklearn.model_selection import train_test_split


SPLIT_RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

BR35H_CLASSES = ["yes", "no"]
PHASE2_CLASSES = ["glioma", "meningioma", "no_tumor", "pituitary"]


def collect_samples(raw_dir: Path, class_names: List[str]) -> Dict[str, List[Path]]:
    """Return {class_name: [image_paths]} from raw_dir."""
    samples = {}
    for cls in class_names:
        cls_dir = raw_dir / cls
        if not cls_dir.exists():
            raise FileNotFoundError(f"Expected class directory not found: {cls_dir}")
        imgs = [p for p in cls_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS]
        print(f"  {cls}: {len(imgs)} images")
        samples[cls] = imgs
    return samples


def split_samples(
    samples: Dict[str, List[Path]],
    seed: int = 42,
) -> Dict[str, Dict[str, List[Path]]]:
    """Stratified split into train/val/test."""
    splits: Dict[str, Dict[str, List[Path]]] = {"train": {}, "val": {}, "test": {}}

    for cls, paths in samples.items():
        random.seed(seed)
        train_paths, temp_paths = train_test_split(paths, test_size=0.30, random_state=seed)
        val_paths, test_paths = train_test_split(temp_paths, test_size=0.50, random_state=seed)
        splits["train"][cls] = train_paths
        splits["val"][cls] = val_paths
        splits["test"][cls] = test_paths

    return splits


def copy_split(
    splits: Dict[str, Dict[str, List[Path]]],
    output_dir: Path,
) -> None:
    """Copy files to output_dir/split/class/ directories."""
    for split_name, class_map in splits.items():
        for cls, paths in class_map.items():
            dest = output_dir / split_name / cls
            dest.mkdir(parents=True, exist_ok=True)
            for src in paths:
                shutil.copy2(src, dest / src.name)
        total = sum(len(v) for v in class_map.values())
        print(f"  {split_name}: {total} images")


def main():
    parser = argparse.ArgumentParser(description="Prepare brain MRI dataset splits.")
    parser.add_argument("--dataset", choices=["br35h", "phase2"], default="br35h",
                        help="Which dataset to prepare.")
    parser.add_argument("--raw", default="data/raw",
                        help="Path to raw data directory (default: data/raw).")
    parser.add_argument("--output", default="data/processed",
                        help="Output directory for splits (default: data/processed).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    args = parser.parse_args()

    raw_dir = Path(args.raw)
    output_dir = Path(args.output)
    class_names = BR35H_CLASSES if args.dataset == "br35h" else PHASE2_CLASSES

    print(f"\nDataset : {args.dataset}")
    print(f"Raw dir : {raw_dir}")
    print(f"Output  : {output_dir}")
    print(f"Classes : {class_names}\n")
    print("Collecting samples...")
    samples = collect_samples(raw_dir, class_names)

    print("\nSplitting (70/15/15 stratified)...")
    splits = split_samples(samples, seed=args.seed)

    print("\nCopying files...")
    copy_split(splits, output_dir)

    print(f"\nDone. Processed dataset written to: {output_dir}")


if __name__ == "__main__":
    main()
