"""
evaluate.py — Run full evaluation for Phase 1 or Phase 2.

Usage:
    python scripts/evaluate.py --phase 1 --checkpoint outputs/checkpoints/phase1/phase1_best.pth
    python scripts/evaluate.py --phase 2 --checkpoint outputs/checkpoints/phase2/phase2_best.pth
"""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.dataset import build_dataloaders
from src.data.transforms import get_transforms
from src.models.backbone import build_backbone
from src.models.detection_head import DetectionHead
from src.models.classification_head import ClassificationHead
from src.evaluation.evaluator import Evaluator
from src.utils.checkpoint import load_checkpoint
from src.utils.logger import get_logger


def build_phase1_model(cfg: dict) -> nn.Module:
    backbone = build_backbone(cfg["model"]["backbone"], pretrained=False)
    head = DetectionHead(backbone.out_features, cfg["model"]["dropout"])

    class DetectionModel(nn.Module):
        def __init__(self, b, h):
            super().__init__()
            self.backbone = b
            self.head = h

        def forward(self, x):
            return self.head(self.backbone(x))

    return DetectionModel(backbone, head)


def build_phase2_model(cfg: dict) -> nn.Module:
    backbone = build_backbone(cfg["model"]["backbone"], pretrained=False)
    head = ClassificationHead(backbone.out_features, cfg["num_classes"], cfg["model"]["dropout"])

    class ClassificationModel(nn.Module):
        def __init__(self, b, h):
            super().__init__()
            self.backbone = b
            self.head = h

        def forward(self, x):
            return self.head(self.backbone(x))

    return ClassificationModel(backbone, head)


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained brain tumor model.")
    parser.add_argument("--phase", type=int, required=True, choices=[1, 2])
    parser.add_argument("--checkpoint", required=True, help="Path to .pth checkpoint file.")
    parser.add_argument(
        "--config",
        help="Config YAML path (auto-detected from phase if not provided).",
    )
    args = parser.parse_args()

    if args.config:
        config_path = args.config
    else:
        config_path = f"config/phase{args.phase}_config.yaml"

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    logger = get_logger("evaluate")
    logger.info(f"Evaluating Phase {args.phase} | Checkpoint: {args.checkpoint}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build and load model
    if args.phase == 1:
        model = build_phase1_model(cfg)
        task = "detection"
        class_names = ["no_tumor", "tumor"]
        threshold = cfg["evaluation"]["threshold"]
    else:
        model = build_phase2_model(cfg)
        task = "classification"
        class_names = cfg["class_names"]
        threshold = 0.5

    load_checkpoint(model, args.checkpoint, device=device)
    model.to(device)

    # Data
    val_tf = get_transforms("test", cfg["data"]["image_size"])
    _, _, test_loader = build_dataloaders(
        root=cfg["data"]["root"],
        task=task,
        batch_size=cfg["data"]["batch_size"],
        num_workers=cfg["data"]["num_workers"],
        train_transform=val_tf,
        val_transform=val_tf,
        class_names=class_names if task == "classification" else None,
    )

    # Evaluate
    evaluator = Evaluator(
        device=device,
        task=task,
        threshold=threshold,
        class_names=class_names,
        plot_dir=cfg["output"]["plot_dir"],
        report_dir=cfg["output"]["report_dir"],
    )
    metrics = evaluator.run(model, test_loader)

    print(f"\nEvaluation Results (Phase {args.phase}):")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    main()
