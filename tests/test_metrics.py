"""Tests for compute_metrics and loss functions."""

import torch
import pytest

from src.training.metrics import compute_metrics, get_classification_report
from src.training.losses import DetectionLoss, ClassificationLoss, MultiTaskLoss


class TestComputeMetrics:
    def test_detection_perfect_score(self):
        preds = [1, 0, 1, 0, 1]
        labels = [1, 0, 1, 0, 1]
        m = compute_metrics(preds, labels, task="detection")
        assert m["accuracy"] == 1.0
        assert m["f1"] == 1.0
        assert m["precision"] == 1.0
        assert m["recall"] == 1.0

    def test_detection_all_wrong(self):
        preds = [0, 1, 0, 1]
        labels = [1, 0, 1, 0]
        m = compute_metrics(preds, labels, task="detection")
        assert m["accuracy"] == 0.0
        assert m["recall"] == 0.0

    def test_detection_with_probs_auc(self):
        preds = [1, 0, 1, 0]
        labels = [1, 0, 1, 0]
        probs = [0.9, 0.1, 0.8, 0.2]
        m = compute_metrics(preds, labels, task="detection", probs=probs)
        assert "auc" in m
        assert m["auc"] == pytest.approx(1.0)

    def test_classification_metrics_keys(self):
        preds = [0, 1, 2, 3]
        labels = [0, 1, 2, 3]
        class_names = ["glioma", "meningioma", "no_tumor", "pituitary"]
        m = compute_metrics(preds, labels, task="classification", class_names=class_names)
        assert "accuracy" in m
        assert "f1_macro" in m
        assert "f1_glioma" in m
        assert "f1_pituitary" in m

    def test_classification_report_string(self):
        preds = [0, 1, 0, 1]
        labels = [0, 0, 1, 1]
        report = get_classification_report(preds, labels, class_names=["no", "yes"])
        assert "precision" in report
        assert "recall" in report


class TestLosses:
    def test_detection_loss_shape(self):
        loss_fn = DetectionLoss()
        logits = torch.randn(8, 1)
        labels = torch.randint(0, 2, (8,))
        loss = loss_fn(logits, labels)
        assert loss.ndim == 0
        assert loss.item() > 0

    def test_classification_loss_shape(self):
        loss_fn = ClassificationLoss(num_classes=4)
        logits = torch.randn(8, 4)
        labels = torch.randint(0, 4, (8,))
        loss = loss_fn(logits, labels)
        assert loss.ndim == 0
        assert loss.item() > 0

    def test_classification_loss_label_smoothing(self):
        loss_fn_smooth = ClassificationLoss(num_classes=4, label_smoothing=0.1)
        loss_fn_plain = ClassificationLoss(num_classes=4, label_smoothing=0.0)
        logits = torch.randn(8, 4)
        labels = torch.randint(0, 4, (8,))
        # Smoothed loss should generally be higher than plain on correct predictions
        assert loss_fn_smooth(logits, labels).item() != loss_fn_plain(logits, labels).item()

    def test_multitask_loss(self):
        det_loss = DetectionLoss()
        cls_loss = ClassificationLoss(num_classes=4)
        mt_loss = MultiTaskLoss(det_loss, cls_loss, alpha=0.5, beta=0.5)

        det_logits = torch.randn(8, 1)
        cls_logits = torch.randn(8, 4)
        det_labels = torch.randint(0, 2, (8,))
        cls_labels = torch.randint(0, 4, (8,))

        loss = mt_loss(det_logits, cls_logits, det_labels, cls_labels)
        assert loss.ndim == 0
        assert loss.item() > 0
