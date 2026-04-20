import json
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.training.metrics import compute_metrics
from src.utils.checkpoint import save_checkpoint
from src.utils.logger import get_logger


class Trainer:
    """
    Phase-agnostic training loop.

    Receives a model and loss function — has no knowledge of tasks beyond what
    the loss function and config dictate.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler,
        loss_fn: nn.Module,
        device: torch.device,
        task: str = "detection",
        checkpoint_dir: str = "outputs/checkpoints",
        early_stopping_patience: int = 7,
        threshold: float = 0.5,
        class_names=None,
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_fn = loss_fn
        self.device = device
        self.task = task
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.early_stopping_patience = early_stopping_patience
        self.threshold = threshold
        self.class_names = class_names
        self.logger = get_logger(__name__)

        self.best_val_score = -1.0
        self.epochs_without_improvement = 0
        self.history = {"train_loss": [], "val_loss": [], "train_metrics": [], "val_metrics": []}

    def train_epoch(self, dataloader: DataLoader) -> Tuple[float, Dict]:
        self.model.train()
        total_loss = 0.0
        all_preds, all_labels, all_probs = [], [], []

        for images, labels in tqdm(dataloader, desc="Train", leave=False):
            images = images.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()
            logits = self.model(images)
            loss = self.loss_fn(logits, labels)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * images.size(0)
            preds, probs = self._get_preds(logits)
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().tolist())
            all_probs.extend(probs)

        avg_loss = total_loss / len(dataloader.dataset)
        metrics = compute_metrics(all_preds, all_labels, task=self.task, class_names=self.class_names, probs=all_probs)
        return avg_loss, metrics

    @torch.no_grad()
    def val_epoch(self, dataloader: DataLoader) -> Tuple[float, Dict]:
        self.model.eval()
        total_loss = 0.0
        all_preds, all_labels, all_probs = [], [], []

        for images, labels in tqdm(dataloader, desc="Val  ", leave=False):
            images = images.to(self.device)
            labels = labels.to(self.device)

            logits = self.model(images)
            loss = self.loss_fn(logits, labels)

            total_loss += loss.item() * images.size(0)
            preds, probs = self._get_preds(logits)
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().tolist())
            all_probs.extend(probs)

        avg_loss = total_loss / len(dataloader.dataset)
        metrics = compute_metrics(all_preds, all_labels, task=self.task, class_names=self.class_names, probs=all_probs)
        return avg_loss, metrics

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int,
        phase_name: str = "phase",
    ) -> Dict:
        self.logger.info(f"Starting training for {epochs} epochs on {self.device}")

        for epoch in range(1, epochs + 1):
            t0 = time.time()
            train_loss, train_metrics = self.train_epoch(train_loader)
            val_loss, val_metrics = self.val_epoch(val_loader)

            if self.scheduler is not None:
                self.scheduler.step()

            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["train_metrics"].append(train_metrics)
            self.history["val_metrics"].append(val_metrics)

            val_score = self._primary_metric(val_metrics)
            elapsed = time.time() - t0

            self.logger.info(
                f"Epoch {epoch:03d}/{epochs} | "
                f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} | "
                f"val_score={val_score:.4f} | {elapsed:.1f}s"
            )

            if val_score > self.best_val_score:
                self.best_val_score = val_score
                self.epochs_without_improvement = 0
                save_checkpoint(
                    self.model,
                    self.optimizer,
                    epoch,
                    val_score,
                    self.checkpoint_dir / f"{phase_name}_best.pth",
                )
                self.logger.info(f"  [best] New checkpoint saved (score={val_score:.4f})")
            else:
                self.epochs_without_improvement += 1
                if self.epochs_without_improvement >= self.early_stopping_patience:
                    self.logger.info(f"Early stopping triggered after {epoch} epochs.")
                    break

        self._save_history(phase_name)
        return self.history

    def _get_preds(self, logits: torch.Tensor):
        if self.task == "detection":
            probs = torch.sigmoid(logits).squeeze(1).cpu().tolist()
            preds = [1 if p >= self.threshold else 0 for p in probs]
        else:
            probs_tensor = torch.softmax(logits, dim=1)
            preds = logits.argmax(dim=1).cpu().tolist()
            probs = probs_tensor.max(dim=1).values.cpu().tolist()
        return preds, probs

    def _primary_metric(self, metrics: Dict) -> float:
        if self.task == "detection":
            return metrics.get("f1", 0.0)
        return metrics.get("f1_macro", 0.0)

    def _save_history(self, phase_name: str) -> None:
        history_path = self.checkpoint_dir / f"{phase_name}_history.json"
        with open(history_path, "w") as f:
            json.dump(self.history, f, indent=2)
        self.logger.info(f"Training history saved to {history_path}")
