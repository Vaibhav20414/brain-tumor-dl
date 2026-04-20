from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

from src.utils.logger import get_logger


logger = get_logger(__name__)


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    val_score: float,
    save_path: str,
) -> None:
    """
    Save model weights, optimizer state, epoch, and best score.

    Args:
        model:      Model to save.
        optimizer:  Optimizer state to save.
        epoch:      Current epoch number.
        val_score:  Validation score at this checkpoint.
        save_path:  Full file path for the checkpoint (.pth).
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "epoch": epoch,
            "val_score": val_score,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        save_path,
    )
    logger.info(f"Checkpoint saved: {save_path} (epoch={epoch}, val_score={val_score:.4f})")


def load_checkpoint(
    model: nn.Module,
    checkpoint_path: str,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: Optional[torch.device] = None,
) -> dict:
    """
    Load model (and optionally optimizer) weights from a checkpoint.

    Args:
        model:           Model to load weights into.
        checkpoint_path: Path to the .pth checkpoint file.
        optimizer:       Optional optimizer to restore state.
        device:          Target device for map_location.

    Returns:
        The checkpoint dict (contains epoch, val_score, etc.).
    """
    if device is None:
        device = torch.device("cpu")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    epoch = checkpoint.get("epoch", 0)
    val_score = checkpoint.get("val_score", 0.0)
    logger.info(f"Checkpoint loaded: {checkpoint_path} (epoch={epoch}, val_score={val_score:.4f})")
    return checkpoint


def load_backbone_weights(model: nn.Module, checkpoint_path: str, device: torch.device) -> None:
    """
    Load only the backbone weights from a Phase 1 checkpoint into a model.

    Useful for Phase 2 training where only backbone weights should be transferred.

    Args:
        model:           Model with a .backbone attribute.
        checkpoint_path: Path to Phase 1 .pth checkpoint.
        device:          Target device.
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    full_state = checkpoint["model_state_dict"]

    backbone_state = {
        k.replace("backbone.", ""): v
        for k, v in full_state.items()
        if k.startswith("backbone.")
    }

    if not backbone_state:
        logger.warning("No backbone keys found in checkpoint. Loading full state dict.")
        model.load_state_dict(full_state, strict=False)
    else:
        model.backbone.load_state_dict(backbone_state)
        logger.info(f"Backbone weights loaded from {checkpoint_path}")
