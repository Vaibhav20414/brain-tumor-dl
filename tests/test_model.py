"""Tests for model components: backbone, detection head, classification head."""

import torch
import pytest

from src.models.backbone import CNNBackbone, build_backbone
from src.models.detection_head import DetectionHead
from src.models.classification_head import ClassificationHead, BrainTumorModel


class TestCNNBackbone:
    def test_output_shape(self):
        model = CNNBackbone()
        x = torch.randn(4, 3, 224, 224)
        out = model(x)
        assert out.shape == (4, 512)

    def test_out_features_attr(self):
        model = CNNBackbone()
        assert model.out_features == 512

    def test_build_backbone_cnn(self):
        model = build_backbone("cnn")
        assert model.out_features == 512

    def test_build_backbone_unknown_raises(self):
        with pytest.raises(ValueError):
            build_backbone("unknown_backbone")


class TestDetectionHead:
    def test_output_shape(self):
        head = DetectionHead(in_features=512, dropout=0.4)
        x = torch.randn(8, 512)
        out = head(x)
        assert out.shape == (8, 1)

    def test_no_nan_output(self):
        head = DetectionHead()
        x = torch.randn(4, 512)
        out = head(x)
        assert not torch.isnan(out).any()


class TestClassificationHead:
    def test_output_shape(self):
        head = ClassificationHead(in_features=512, num_classes=4, dropout=0.5)
        x = torch.randn(8, 512)
        out = head(x)
        assert out.shape == (8, 4)

    def test_binary_output(self):
        head = ClassificationHead(in_features=512, num_classes=2)
        x = torch.randn(4, 512)
        out = head(x)
        assert out.shape == (4, 2)


class TestBrainTumorModel:
    def _make_model(self, num_classes=4):
        backbone = CNNBackbone()
        det_head = DetectionHead(backbone.out_features)
        cls_head = ClassificationHead(backbone.out_features, num_classes)
        return BrainTumorModel(backbone, det_head, cls_head)

    def test_detection_forward(self):
        model = self._make_model()
        x = torch.randn(2, 3, 224, 224)
        out = model(x, task="detection")
        assert out.shape == (2, 1)

    def test_classification_forward(self):
        model = self._make_model(num_classes=4)
        x = torch.randn(2, 3, 224, 224)
        out = model(x, task="classification")
        assert out.shape == (2, 4)

    def test_unknown_task_raises(self):
        model = self._make_model()
        x = torch.randn(2, 3, 224, 224)
        with pytest.raises(ValueError):
            model(x, task="unknown")

    def test_freeze_backbone(self):
        model = self._make_model()
        model.freeze_backbone()
        for param in model.backbone.parameters():
            assert not param.requires_grad

    def test_unfreeze_backbone(self):
        model = self._make_model()
        model.freeze_backbone()
        model.unfreeze_backbone()
        for param in model.backbone.parameters():
            assert param.requires_grad
