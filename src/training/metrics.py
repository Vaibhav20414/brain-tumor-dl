from typing import Dict, List, Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
)


def compute_metrics(
    preds: List[int],
    labels: List[int],
    task: str = "detection",
    class_names: Optional[List[str]] = None,
    probs: Optional[List[float]] = None,
) -> Dict[str, float]:
    """
    Compute evaluation metrics.

    For detection (binary): accuracy, precision, recall, f1, auc.
    For classification (multi-class): accuracy + per-class and macro precision/recall/f1.

    Args:
        preds:  Predicted class indices.
        labels: Ground-truth class indices.
        task:   "detection" or "classification".
        class_names: Class names for classification reporting.
        probs:  Predicted probabilities (needed for AUC in detection).

    Returns:
        Dict of metric_name → float value.
    """
    preds = np.array(preds)
    labels = np.array(labels)

    metrics: Dict[str, float] = {}

    if task == "detection":
        metrics["accuracy"] = float(accuracy_score(labels, preds))
        metrics["precision"] = float(precision_score(labels, preds, zero_division=0))
        metrics["recall"] = float(recall_score(labels, preds, zero_division=0))
        metrics["f1"] = float(f1_score(labels, preds, zero_division=0))
        if probs is not None:
            try:
                metrics["auc"] = float(roc_auc_score(labels, probs))
            except ValueError:
                metrics["auc"] = 0.0
    else:
        average = "macro"
        metrics["accuracy"] = float(accuracy_score(labels, preds))
        metrics["precision_macro"] = float(precision_score(labels, preds, average=average, zero_division=0))
        metrics["recall_macro"] = float(recall_score(labels, preds, average=average, zero_division=0))
        metrics["f1_macro"] = float(f1_score(labels, preds, average=average, zero_division=0))

        names = class_names if class_names else [str(i) for i in range(len(np.unique(labels)))]
        per_class_f1 = f1_score(labels, preds, average=None, zero_division=0)
        for cls_name, f1 in zip(names, per_class_f1):
            metrics[f"f1_{cls_name}"] = float(f1)

    return metrics


def get_classification_report(
    preds: List[int],
    labels: List[int],
    class_names: Optional[List[str]] = None,
) -> str:
    """Return sklearn classification_report string."""
    return classification_report(labels, preds, target_names=class_names, zero_division=0)
