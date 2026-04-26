"""
age estimator — ResNet-50 backbone with an ordinal regression head.

Phase 3 used a single-float regression head. Phase 4 swaps that for an
ordinal head that outputs num_thresholds (100) logits. each logit answers
"is the true age > k?".

design choice — ordinal regression vs plain regression:
    plain regression collapses every wrong prediction into one scalar
    gradient. ordinal regression turns every age into 100 binary
    yes/no questions, each with its own gradient. the model gets a
    much stronger learning signal per sample, which lowers MAE on age
    estimation in published results (Niu et al. 2016).

design choice — keeping the ResNet-50 backbone:
    the backbone learned good visual features in Phase 3. swapping
    only the head means we keep all that representational learning
    and just change how the final layer interprets it.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet50_Weights

from src.models.ordinal_loss import NUM_THRESHOLDS, ordinal_logits_to_age


class AgeEstimator(nn.Module):
    """
    ResNet-50 backbone + ordinal regression head -> (B, num_thresholds) logits.

    forward returns logits. use ordinal_logits_to_age() in ordinal_loss.py
    to convert logits to predicted ages at eval / inference time.
    """

    def __init__(
        self,
        num_thresholds: int = NUM_THRESHOLDS,
        pretrained: bool = True,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.num_thresholds = num_thresholds

        # load ImageNet-pretrained ResNet-50 — gives a strong starting point
        # for the visual features. weights=None gives a fresh init for tests
        # that need to be fast (no model download).
        weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        backbone = models.resnet50(weights=weights)

        # ResNet-50's final fc layer outputs 1000 ImageNet classes by default.
        # grab the input dim of that fc (2048 for resnet-50) and replace fc
        # with Identity so backbone(x) returns the 2048-dim feature vector.
        in_features = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone

        # ordinal head — small MLP -> num_thresholds logits.
        # dropout helps generalisation since UTKFace is small (~24k images).
        # final layer has no activation — BCEWithLogitsLoss applies sigmoid
        # internally during training.
        self.head = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(512, num_thresholds),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, 3, H, W) image tensor — already normalised by transforms
        returns: (B, num_thresholds) raw logits
        """
        # backbone -> 2048-dim feature vector per image
        features = self.backbone(x)

        # head -> num_thresholds logits per image
        logits = self.head(features)
        return logits

    @torch.no_grad()
    def predict_age(self, x: torch.Tensor) -> torch.Tensor:
        """
        convenience wrapper for inference — runs forward and decodes
        logits to a float tensor of predicted ages.

        @torch.no_grad() turns off autograd here — saves memory and
        speed during inference.
        """
        logits = self.forward(x)
        return ordinal_logits_to_age(logits)
