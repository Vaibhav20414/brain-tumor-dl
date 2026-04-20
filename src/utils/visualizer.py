from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt


def plot_training_curves(
    history: Dict,
    save_dir: str,
    phase_name: str = "phase",
) -> None:
    """
    Plot and save loss and primary metric curves.

    Args:
        history:    Dict with keys train_loss, val_loss, train_metrics, val_metrics.
        save_dir:   Directory to save figures.
        phase_name: Used as prefix in file names.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    epochs = list(range(1, len(history["train_loss"]) + 1))

    # Loss curve
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, history["train_loss"], label="Train Loss")
    ax.plot(epochs, history["val_loss"], label="Val Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(f"{phase_name} — Loss Curve")
    ax.legend()
    plt.tight_layout()
    fig.savefig(save_dir / f"{phase_name}_loss.png", dpi=150)
    plt.close(fig)

    # Primary metric curve (F1 for detection, macro-F1 for classification)
    train_scores = _extract_primary(history["train_metrics"])
    val_scores = _extract_primary(history["val_metrics"])

    if train_scores:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(epochs, train_scores, label="Train Score")
        ax.plot(epochs, val_scores, label="Val Score")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Score")
        ax.set_title(f"{phase_name} — Metric Curve")
        ax.legend()
        plt.tight_layout()
        fig.savefig(save_dir / f"{phase_name}_metrics.png", dpi=150)
        plt.close(fig)


def _extract_primary(metrics_list: List[Dict]) -> List[float]:
    if not metrics_list:
        return []
    sample = metrics_list[0]
    key = "f1_macro" if "f1_macro" in sample else "f1"
    return [m.get(key, 0.0) for m in metrics_list]
