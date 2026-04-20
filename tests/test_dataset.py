"""Tests for BrainMRIDataset and build_dataloaders."""

import os
import tempfile
from pathlib import Path

import pytest
from PIL import Image

from src.data.dataset import BrainMRIDataset
from src.data.transforms import get_transforms


def _make_fake_dataset(root: Path, task: str = "detection") -> None:
    """Create a minimal fake image dataset for testing."""
    if task == "detection":
        classes = {"train": ["yes", "no"], "val": ["yes", "no"], "test": ["yes", "no"]}
    else:
        classes = {
            "train": ["glioma", "meningioma", "no_tumor", "pituitary"],
            "val": ["glioma", "meningioma", "no_tumor", "pituitary"],
            "test": ["glioma", "meningioma", "no_tumor", "pituitary"],
        }

    for split, cls_list in classes.items():
        for cls in cls_list:
            cls_dir = root / split / cls
            cls_dir.mkdir(parents=True, exist_ok=True)
            for i in range(3):
                img = Image.new("RGB", (64, 64), color=(i * 50, i * 30, i * 20))
                img.save(cls_dir / f"img_{i}.jpg")


class TestBrainMRIDataset:
    def test_detection_dataset_length(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _make_fake_dataset(root, task="detection")
            ds = BrainMRIDataset(root=str(root), task="detection", split="train")
            assert len(ds) == 6  # 2 classes × 3 images

    def test_detection_labels(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _make_fake_dataset(root, task="detection")
            ds = BrainMRIDataset(root=str(root), task="detection", split="train")
            labels = {label for _, label in ds.samples}
            assert labels == {0, 1}

    def test_classification_dataset_length(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _make_fake_dataset(root, task="classification")
            ds = BrainMRIDataset(
                root=str(root),
                task="classification",
                split="train",
                class_names=["glioma", "meningioma", "no_tumor", "pituitary"],
            )
            assert len(ds) == 12  # 4 classes × 3 images

    def test_getitem_with_transform(self):
        import torch

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _make_fake_dataset(root, task="detection")
            tf = get_transforms("train", image_size=224)
            ds = BrainMRIDataset(root=str(root), task="detection", split="train", transform=tf)
            img, label = ds[0]
            assert img.shape == (3, 224, 224)
            assert label in {0, 1}

    def test_missing_directory_raises(self):
        with pytest.raises(FileNotFoundError):
            BrainMRIDataset(root="/nonexistent/path", task="detection", split="train")
