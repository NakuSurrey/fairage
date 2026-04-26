"""
unit tests for AgeEstimator.

Phase 4: model now outputs (B, NUM_THRESHOLDS) logits instead of a single
float. tests updated to match. predict_age helper is also covered.

these run on CPU with pretrained=False so no model weights are downloaded
during the test run.
"""

from __future__ import annotations

import torch
import pytest

from src.models.age_estimator import AgeEstimator
from src.models.ordinal_loss import NUM_THRESHOLDS


class TestAgeEstimator:
    """end-to-end checks that the model wiring is correct."""

    def test_output_shape(self):
        # forward returns logits, one per threshold per image
        model = AgeEstimator(pretrained=False)
        x = torch.randn(2, 3, 224, 224)
        out = model(x)
        assert out.shape == (2, NUM_THRESHOLDS)

    def test_output_is_finite(self):
        # untrained network should still produce finite values — no NaN/inf
        # leak from the head, which would silently break training
        model = AgeEstimator(pretrained=False)
        x = torch.randn(2, 3, 224, 224)
        out = model(x)
        assert torch.isfinite(out).all().item()

    def test_predict_age_shape(self):
        # predict_age decodes logits to a float age per image
        model = AgeEstimator(pretrained=False)
        x = torch.randn(3, 3, 224, 224)
        ages = model.predict_age(x)
        assert ages.shape == (3,)

    def test_predict_age_in_valid_range(self):
        # decoded age must be between 0 and NUM_THRESHOLDS inclusive
        model = AgeEstimator(pretrained=False)
        x = torch.randn(4, 3, 224, 224)
        ages = model.predict_age(x)
        assert (ages >= 0).all().item()
        assert (ages <= NUM_THRESHOLDS).all().item()

    def test_backbone_swap_to_identity(self):
        # ResNet-50's fc must be replaced with Identity so the backbone
        # outputs the 2048-dim feature vector, not 1000-class logits
        model = AgeEstimator(pretrained=False)
        assert isinstance(model.backbone.fc, torch.nn.Identity)

    def test_head_output_dim(self):
        # final Linear in head must produce NUM_THRESHOLDS outputs
        model = AgeEstimator(pretrained=False)
        last_layer = model.head[-1]
        assert isinstance(last_layer, torch.nn.Linear)
        assert last_layer.out_features == NUM_THRESHOLDS

    def test_gradients_flow(self):
        # full backward pass — confirms no detach / no_grad accidentally
        # left in the path between input and loss
        model = AgeEstimator(pretrained=False)
        x = torch.randn(2, 3, 224, 224)
        out = model(x)
        loss = out.sum()
        loss.backward()
        # at least one backbone param should have a non-zero gradient
        any_grad = any(
            p.grad is not None and p.grad.abs().sum().item() > 0
            for p in model.backbone.parameters()
        )
        assert any_grad
