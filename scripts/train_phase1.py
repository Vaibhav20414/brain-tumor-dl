"""
train_phase1.py — Phase 1 training entry point (binary tumor detection).

Usage:
    python scripts/train_phase1.py --config config/phase1_config.yaml
"""

import argparse
import sys
from pathlib import Path

import torch
import yaml

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.dataset import build_dataloaders
from src.data.transforms import get_transforms
from src.models.backbone import build_backbone
from src.models.detection_head import DetectionHead
from src.training.losses import DetectionLoss
from src.training.trainer import Trainer
from src.utils.logger import get_logger
from src.utils.visualizer import plot_training_curves


def build_model(cfg: dict, device: torch.device) -> torch.nn.Module:
    """Assemble backbone + detection head into a sequential model."""
    import torch.nn as nn

    backbone = build_backbone(
        name=cfg["model"]["backbone"],
        pretrained=cfg["model"].get("pretrained", False),
    )
    head = DetectionHead(
        in_features=backbone.out_features,
        dropout=cfg["model"]["dropout"],
    )

    class DetectionModel(nn.Module):
        def __init__(self, backbone, head):
            super().__init__()
            self.backbone = backbone
            self.head = head

        def forward(self, x):
            return self.head(self.backbone(x))

    model = DetectionModel(backbone, head).to(device)
    return model


def main():
    parser = argparse.ArgumentParser(description="Train Phase 1 — Tumor Detection")
    parser.add_argument("--config", default="config/phase1_config.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    logger = get_logger("train_phase1", log_file=f"{cfg['output']['checkpoint_dir']}/train.log")
    logger.info(f"Loaded config: {args.config}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # Data
    train_tf = get_transforms("train", cfg["data"]["image_size"])
    val_tf = get_transforms("val", cfg["data"]["image_size"])
    train_loader, val_loader, _ = build_dataloaders(
        root=cfg["data"]["root"],
        task="detection",
        batch_size=cfg["data"]["batch_size"],
        num_workers=cfg["data"]["num_workers"],
        train_transform=train_tf,
        val_transform=val_tf,
    )
    logger.info(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

    # Model
    model = build_model(cfg, device)
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Optimizer & scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["training"]["lr"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg["training"]["epochs"]
    )

    # Loss
    loss_fn = DetectionLoss()

    # Trainer
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_fn=loss_fn,
        device=device,
        task="detection",
        checkpoint_dir=cfg["output"]["checkpoint_dir"],
        early_stopping_patience=cfg["training"]["early_stopping_patience"],
        threshold=cfg["evaluation"]["threshold"],
    )

    history = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=cfg["training"]["epochs"],
        phase_name="phase1",
    )

    plot_training_curves(history, save_dir=cfg["output"]["plot_dir"], phase_name="phase1")
    logger.info("Phase 1 training complete.")


if __name__ == "__main__":
    main()
