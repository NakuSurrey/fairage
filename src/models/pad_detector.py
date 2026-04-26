"""
PAD detector — small CNN for binary classification (real vs attack).

design choice — small CNN over ResNet-50:
    NUAA has ~12k images. ResNet-50 has 23M parameters and would overfit
    badly. a small custom CNN with ~500k parameters is the right capacity
    for the data size and gives a much faster training cycle on the HPC.
    matches the original NUAA paper architecture (Tan et al., 2010) which
    used a similarly small network.

design choice — output 2 logits, not 1:
    using CrossEntropyLoss over 2 classes is the standard pattern in
    PyTorch. easy to extend to 3+ classes later (real / print / replay)
    if a richer dataset is added in a future iteration.

architecture (small but proven):
    block 1: Conv(3->32) -> ReLU -> Conv(32->32) -> ReLU -> MaxPool
    block 2: Conv(32->64) -> ReLU -> Conv(64->64) -> ReLU -> MaxPool
    block 3: Conv(64->128) -> ReLU -> Conv(128->128) -> ReLU -> MaxPool
    AdaptiveAvgPool to (1, 1) -> Linear(128 -> 2) for 2-class output
"""

from __future__ import annotations

import torch
import torch.nn as nn


def _conv_block(in_ch: int, out_ch: int) -> nn.Sequential:
    """
    two 3x3 convs with ReLU, followed by a 2x2 max-pool. standard VGG-style
    block — small enough that the whole network has fewer than 1M params.
    padding=1 keeps spatial size the same through the convs; the pool halves it.
    """
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2, stride=2),
    )


class PADDetector(nn.Module):
    """
    small CNN — input 224x224x3 -> 2-class logits.

    forward returns raw logits. CrossEntropyLoss applies softmax internally
    during training. at inference, take argmax for the predicted class or
    softmax for a probability score.
    """

    def __init__(self, num_classes: int = 2, dropout: float = 0.3) -> None:
        super().__init__()

        # feature extractor — three downsampling blocks
        # spatial size at each stage with 224 input:
        #   after block 1: 112x112
        #   after block 2: 56x56
        #   after block 3: 28x28
        self.features = nn.Sequential(
            _conv_block(3, 32),
            _conv_block(32, 64),
            _conv_block(64, 128),
        )

        # global average pool to (128, 1, 1) — makes the head input-size agnostic
        # and keeps the parameter count low (no giant FC layer)
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        # classification head — Linear(128 -> 2) with dropout for regularisation
        # NUAA is small, regularisation matters more than depth here
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, 3, 224, 224) image tensor — already normalised by transforms
        returns: (B, num_classes) raw logits
        """
        x = self.features(x)
        x = self.global_pool(x)
        x = self.classifier(x)
        return x

    @torch.no_grad()
    def predict_class(self, x: torch.Tensor) -> torch.Tensor:
        """
        convenience inference wrapper. returns predicted class index per image.

        @torch.no_grad() turns off autograd here — saves memory and time
        during inference and at API serving.
        """
        logits = self.forward(x)
        return logits.argmax(dim=1)

    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """
        return per-class probabilities. used by the API to expose a
        confidence score alongside the binary decision.
        """
        logits = self.forward(x)
        return torch.softmax(logits, dim=1)


def count_parameters(model: nn.Module) -> dict[str, int]:
    """quick sanity helper — total vs trainable params, used in EDA / README."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}
