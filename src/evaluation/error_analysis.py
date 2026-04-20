from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.utils.logger import get_logger


logger = get_logger(__name__)


def find_error_cases(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    task: str = "detection",
    threshold: float = 0.5,
    save_dir: str = "outputs/plots/errors",
    max_cases: int = 16,
) -> Tuple[List, List]:
    """
    Find false-positive and false-negative samples and save image grids.

    Args:
        model:       Trained model (eval mode).
        dataloader:  DataLoader for the dataset to inspect.
        device:      Compute device.
        task:        "detection" or "classification".
        threshold:   Sigmoid threshold (detection only).
        save_dir:    Directory to save error grids.
        max_cases:   Max images per error grid.

    Returns:
        (fp_indices, fn_indices) — indices of FP and FN samples.
    """
    model.eval()
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    fp_images, fn_images = [], []
    fp_indices, fn_indices = [], []
    global_idx = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            logits = model(images)

            if task == "detection":
                probs = torch.sigmoid(logits).squeeze(1).cpu()
                preds = (probs >= threshold).long()
            else:
                preds = logits.argmax(dim=1).cpu()

            for i in range(len(labels)):
                pred = preds[i].item()
                label = labels[i].item()
                img = images[i].cpu()

                if task == "detection":
                    is_fp = (pred == 1 and label == 0)
                    is_fn = (pred == 0 and label == 1)
                else:
                    is_fp = (pred != label and label == 0)
                    is_fn = (pred != label and label != 0)

                if is_fp and len(fp_images) < max_cases:
                    fp_images.append(img)
                    fp_indices.append(global_idx)
                if is_fn and len(fn_images) < max_cases:
                    fn_images.append(img)
                    fn_indices.append(global_idx)
                global_idx += 1

    _save_image_grid(fp_images, save_dir / "false_positives.png", "False Positives")
    _save_image_grid(fn_images, save_dir / "false_negatives.png", "False Negatives")
    logger.info(f"Error analysis: {len(fp_indices)} FP, {len(fn_indices)} FN saved to {save_dir}")

    return fp_indices, fn_indices


def _denormalize(tensor: torch.Tensor) -> torch.Tensor:
    """Reverse ImageNet normalization for display."""
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    return (tensor * std + mean).clamp(0, 1)


def _save_image_grid(images: List[torch.Tensor], save_path: Path, title: str) -> None:
    if not images:
        logger.info(f"No images to save for: {title}")
        return

    n = len(images)
    cols = min(4, n)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    axes = [axes] if rows == 1 and cols == 1 else axes
    axes = [ax for row in (axes if rows > 1 else [axes]) for ax in (row if cols > 1 else [row])]

    for i, (img, ax) in enumerate(zip(images, axes)):
        ax.imshow(_denormalize(img).permute(1, 2, 0).numpy())
        ax.axis("off")
    for ax in axes[len(images):]:
        ax.axis("off")

    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)
