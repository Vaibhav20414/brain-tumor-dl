import json
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.training.metrics import compute_metrics, get_classification_report
from src.evaluation.confusion_matrix import plot_confusion_matrix
from src.evaluation.error_analysis import find_error_cases
from src.utils.logger import get_logger


class Evaluator:
    """
    Orchestrates full model evaluation on a test set.

    Produces: metrics dict, confusion matrix plot, error analysis images,
    and a text classification report (Phase 2).
    """

    def __init__(
        self,
        device: torch.device,
        task: str = "detection",
        threshold: float = 0.5,
        class_names: Optional[List[str]] = None,
        plot_dir: str = "outputs/plots",
        report_dir: str = "outputs/reports",
    ):
        self.device = device
        self.task = task
        self.threshold = threshold
        self.class_names = class_names or (["no_tumor", "tumor"] if task == "detection" else [])
        self.plot_dir = Path(plot_dir)
        self.report_dir = Path(report_dir)
        self.plot_dir.mkdir(parents=True, exist_ok=True)
        self.report_dir.mkdir(parents=True, exist_ok=True)
        self.logger = get_logger(__name__)

    @torch.no_grad()
    def run(self, model: nn.Module, test_loader: DataLoader) -> Dict:
        """
        Run full evaluation on test_loader.

        Returns:
            metrics dict with all computed scores.
        """
        model.eval()
        model.to(self.device)

        all_preds, all_labels, all_probs = [], [], []

        for images, labels in tqdm(test_loader, desc="Evaluating"):
            images = images.to(self.device)
            logits = model(images)

            if self.task == "detection":
                probs = torch.sigmoid(logits).squeeze(1).cpu().tolist()
                preds = [1 if p >= self.threshold else 0 for p in probs]
            else:
                probs = torch.softmax(logits, dim=1).max(dim=1).values.cpu().tolist()
                preds = logits.argmax(dim=1).cpu().tolist()

            all_preds.extend(preds)
            all_labels.extend(labels.tolist())
            all_probs.extend(probs)

        metrics = compute_metrics(
            all_preds, all_labels,
            task=self.task,
            class_names=self.class_names,
            probs=all_probs,
        )

        self.logger.info(f"Evaluation metrics: {metrics}")
        self._save_metrics(metrics)
        self._save_confusion_matrix(all_labels, all_preds)

        if self.task == "classification":
            report = get_classification_report(all_preds, all_labels, self.class_names)
            report_path = self.report_dir / "classification_report.txt"
            report_path.write_text(report)
            self.logger.info(f"Classification report saved to {report_path}")

        find_error_cases(
            model,
            test_loader,
            self.device,
            task=self.task,
            threshold=self.threshold,
            save_dir=str(self.plot_dir / "errors"),
        )

        return metrics

    def _save_metrics(self, metrics: Dict) -> None:
        path = self.report_dir / "metrics.json"
        with open(path, "w") as f:
            json.dump(metrics, f, indent=2)
        self.logger.info(f"Metrics saved to {path}")

    def _save_confusion_matrix(self, y_true: List[int], y_pred: List[int]) -> None:
        plot_confusion_matrix(
            y_true=y_true,
            y_pred=y_pred,
            labels=self.class_names,
            save_path=str(self.plot_dir / "confusion_matrix.png"),
            title=f"Confusion Matrix — {self.task.capitalize()}",
        )
        self.logger.info(f"Confusion matrix saved to {self.plot_dir / 'confusion_matrix.png'}")
