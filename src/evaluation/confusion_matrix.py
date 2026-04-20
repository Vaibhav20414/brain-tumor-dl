from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix


def plot_confusion_matrix(
    y_true: List[int],
    y_pred: List[int],
    labels: List[str],
    save_path: str,
    title: str = "Confusion Matrix",
    normalize: bool = True,
) -> None:
    """
    Compute and save a confusion matrix plot.

    Args:
        y_true:    Ground-truth labels.
        y_pred:    Predicted labels.
        labels:    Class names in label-index order.
        save_path: File path to save the figure (PNG).
        title:     Plot title.
        normalize: If True, normalize by row (true class).
    """
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(labels))))

    if normalize:
        cm_display = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-8)
        fmt = ".2f"
    else:
        cm_display = cm
        fmt = "d"

    fig, ax = plt.subplots(figsize=(max(6, len(labels) * 1.5), max(5, len(labels) * 1.3)))
    sns.heatmap(
        cm_display,
        annot=True,
        fmt=fmt,
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
    )
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("True", fontsize=12)
    ax.set_title(title, fontsize=14)
    plt.tight_layout()

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
