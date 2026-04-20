"""
train_phase2.py — Phase 2 training entry point (multi-class tumor classification).

Usage:
    python scripts/train_phase2.py --config config/phase2_config.yaml

Supports three training modes (configured in phase2_config.yaml):
  Option A: Load Phase 1 backbone → freeze → train classification head only
  Option B: Load Phase 1 backbone → fine-tune end-to-end with lower LR
  Option C: Train from scratch (no Phase 1 weights)
  Option D: Multi-task training (detection + classification heads jointly)
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
from src.models.classification_head import ClassificationHead, BrainTumorModel
from src.models.detection_head import DetectionHead
from src.training.losses import ClassificationLoss, MultiTaskLoss, DetectionLoss
from src.training.trainer import Trainer
from src.utils.checkpoint import load_backbone_weights
from src.utils.logger import get_logger
from src.utils.visualizer import plot_training_curves


def build_model(cfg: dict, device: torch.device) -> nn.Module:
    backbone = build_backbone(name=cfg["model"]["backbone"], pretrained=False)
    num_classes = cfg["num_classes"]
    dropout = cfg["model"]["dropout"]

    if cfg["multitask"].get("enabled", False):
        det_head = DetectionHead(in_features=backbone.out_features, dropout=dropout)
        cls_head = ClassificationHead(in_features=backbone.out_features, num_classes=num_classes, dropout=dropout)
        model = BrainTumorModel(backbone, det_head, cls_head)
    else:
        cls_head = ClassificationHead(in_features=backbone.out_features, num_classes=num_classes, dropout=dropout)

        class ClassificationModel(nn.Module):
            def __init__(self, backbone, head):
                super().__init__()
                self.backbone = backbone
                self.head = head

            def forward(self, x):
                return self.head(self.backbone(x))

        model = ClassificationModel(backbone, cls_head)

    # Load Phase 1 backbone weights if specified
    pretrained_path = cfg["model"].get("pretrained_phase1")
    if pretrained_path and Path(pretrained_path).exists():
        load_backbone_weights(model, pretrained_path, device)
    elif pretrained_path:
        get_logger("train_phase2").warning(
            f"pretrained_phase1 path not found: {pretrained_path}. Training from scratch."
        )

    # Freeze backbone if requested
    if cfg["model"].get("freeze_backbone", False):
        model.backbone.requires_grad_(False)
        get_logger("train_phase2").info("Backbone frozen.")

    return model.to(device)


def main():
    parser = argparse.ArgumentParser(description="Train Phase 2 — Tumor Classification")
    parser.add_argument("--config", default="config/phase2_config.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    logger = get_logger("train_phase2", log_file=f"{cfg['output']['checkpoint_dir']}/train.log")
    logger.info(f"Loaded config: {args.config}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    class_names = cfg["class_names"]

    # Data
    train_tf = get_transforms("train", cfg["data"]["image_size"])
    val_tf = get_transforms("val", cfg["data"]["image_size"])
    train_loader, val_loader, _ = build_dataloaders(
        root=cfg["data"]["root"],
        task="classification",
        batch_size=cfg["data"]["batch_size"],
        num_workers=cfg["data"]["num_workers"],
        train_transform=train_tf,
        val_transform=val_tf,
        class_names=class_names,
    )
    logger.info(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

    # Model
    model = build_model(cfg, device)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Trainable parameters: {trainable:,}")

    # Optimizer — only trainable params
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg["training"]["lr"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg["training"]["epochs"]
    )

    # Loss
    label_smoothing = cfg["training"].get("label_smoothing", 0.0)
    loss_fn = ClassificationLoss(
        num_classes=cfg["num_classes"],
        label_smoothing=label_smoothing,
    )

    # Trainer
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_fn=loss_fn,
        device=device,
        task="classification",
        checkpoint_dir=cfg["output"]["checkpoint_dir"],
        early_stopping_patience=cfg["training"]["early_stopping_patience"],
        class_names=class_names,
    )

    history = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=cfg["training"]["epochs"],
        phase_name="phase2",
    )

    plot_training_curves(history, save_dir=cfg["output"]["plot_dir"], phase_name="phase2")
    logger.info("Phase 2 training complete.")


if __name__ == "__main__":
    main()
