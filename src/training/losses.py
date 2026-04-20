import torch
import torch.nn as nn


class DetectionLoss(nn.Module):
    """
    Binary cross-entropy with logits for Phase 1 tumor detection.
    Wraps BCEWithLogitsLoss for a consistent interface.
    """

    def __init__(self, pos_weight: torch.Tensor = None):
        super().__init__()
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = targets.float().unsqueeze(1) if logits.shape[-1] == 1 else targets.float()
        return self.criterion(logits, targets)


class ClassificationLoss(nn.Module):
    """
    Cross-entropy loss for Phase 2 tumor type classification.
    Supports optional label smoothing.
    """

    def __init__(self, num_classes: int = 4, label_smoothing: float = 0.0):
        super().__init__()
        self.criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.criterion(logits, targets.long())


class MultiTaskLoss(nn.Module):
    """
    Weighted sum of detection and classification losses for joint training.

    loss = alpha * detection_loss + beta * classification_loss
    """

    def __init__(
        self,
        detection_loss: DetectionLoss,
        classification_loss: ClassificationLoss,
        alpha: float = 0.5,
        beta: float = 0.5,
    ):
        super().__init__()
        self.detection_loss = detection_loss
        self.classification_loss = classification_loss
        self.alpha = alpha
        self.beta = beta

    def forward(
        self,
        det_logits: torch.Tensor,
        cls_logits: torch.Tensor,
        det_targets: torch.Tensor,
        cls_targets: torch.Tensor,
    ) -> torch.Tensor:
        loss_det = self.detection_loss(det_logits, det_targets)
        loss_cls = self.classification_loss(cls_logits, cls_targets)
        return self.alpha * loss_det + self.beta * loss_cls
