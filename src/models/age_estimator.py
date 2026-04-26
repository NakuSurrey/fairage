"""
age estimator — ResNet-50 backbone with a regression head.

this is the Phase 3 baseline. Phase 4 swaps the head for ordinal regression
to push the MAE down further.

design choice — regression vs classification:
    treating age as a regression problem (one float output) lets the model
    learn that 25 is closer to 26 than to 60. a 101-way classifier would
    treat all wrong answers as equally wrong, which is the wrong signal.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet50_Weights


class AgeEstimator(nn.Module):
    """
    ResNet-50 backbone + small regression head -> single float age prediction.

    backbone is pretrained on imagenet. only the last block plus the new head
    are trained at full LR — earlier blocks fine-tune at a lower effective rate
    once optimiser is set up.
    """

    def __init__(self, pretrained: bool = True, dropout: float = 0.2):
        super().__init__()

        # load ResNet-50 with imagenet weights — these features already capture
        # edges, textures, face structure. saves us ~80% of training compute.
        weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        self.backbone = models.resnet50(weights=weights)

        # ResNet-50's final layer outputs 2048 features for imagenet's 1000 classes
        # replace it with a small regression head — 2048 -> 256 -> 1 (predicted age)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),  # regularisation — UTKFace is small enough to overfit
            nn.Linear(256, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x — image batch, shape [B, 3, 224, 224], imagenet-normalised.
        returns — predicted ages, shape [B], float.
        """
        # backbone.fc returns [B, 1] — squeeze the last dim so loss math is clean
        return self.backbone(x).squeeze(-1)


def count_parameters(model: nn.Module) -> dict[str, int]:
    """quick sanity helper — prints total vs trainable params."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}
