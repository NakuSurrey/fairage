"""
ordinal regression loss for age estimation.

idea — instead of predicting one float, the model outputs K-1 logits where
K is the number of age classes (101 — ages 0 to 100, so K-1 = 100).
each logit answers one binary question: "is the true age > k?"

target encoding:
    age 25  ->  [1, 1, 1, ... (25 ones) ..., 0, 0, 0, ... (75 zeros)]
                                      ^ index 24            ^ index 25 onward
    every position k is 1 if age > k, else 0.

at inference, decoded age = number of positions whose sigmoid > 0.5
                          = sum of (sigmoid(logits) > 0.5)

why this beats plain regression:
    every one of the 100 outputs becomes its own training signal. a wrong
    output at position 30 for a true-25 sample has a clear gradient
    ("you said yes, true is no — push down"). plain regression collapses
    all of this into one scalar gradient, which gives the model less to
    learn from per sample.

this matches Niu et al. 2016 ("Ordinal Regression with Multiple Output
CNN for Age Estimation") which is the standard reference for this
approach in age estimation.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# K-1 = number of binary thresholds. ages run 0..100 inclusive, so K = 101.
NUM_THRESHOLDS = 100


def age_to_ordinal_target(ages: torch.Tensor, num_thresholds: int = NUM_THRESHOLDS) -> torch.Tensor:
    """
    convert a batch of integer ages into ordinal target vectors.

    ages shape:    (B,)             — integer ages, 0 <= age <= num_thresholds
    returns shape: (B, num_thresholds) — float tensor of 0s and 1s

    example with num_thresholds=5 and ages = [2, 4, 0]:
        target = [[1, 1, 0, 0, 0],
                  [1, 1, 1, 1, 0],
                  [0, 0, 0, 0, 0]]
    """
    # clamp keeps values inside [0, num_thresholds] so the comparison stays valid
    # even if a stray label sneaks through outside the expected range
    ages = ages.clamp(min=0, max=num_thresholds).long()

    # build a row of threshold indices [0, 1, 2, ..., num_thresholds-1]
    # shape: (1, num_thresholds) — broadcast against ages of shape (B, 1)
    thresholds = torch.arange(num_thresholds, device=ages.device).unsqueeze(0)

    # ages.unsqueeze(1) -> (B, 1)
    # comparison broadcasts to (B, num_thresholds)
    # position k = 1 if age > k, else 0 — exactly what ordinal targets need
    target = (ages.unsqueeze(1) > thresholds).float()
    return target


def ordinal_logits_to_age(logits: torch.Tensor) -> torch.Tensor:
    """
    decode model output (logits) to predicted age.

    logits shape:  (B, num_thresholds)
    returns shape: (B,) — float tensor of predicted ages

    formula: age = sum over k of indicator(sigmoid(logit_k) > 0.5)
    """
    # sigmoid converts logits to probabilities in [0, 1]
    probs = torch.sigmoid(logits)

    # threshold at 0.5 — count how many "is age > k?" answers are yes
    # summing along the threshold axis gives the predicted age as a float
    age = (probs > 0.5).float().sum(dim=1)
    return age


class OrdinalRegressionLoss(nn.Module):
    """
    binary cross entropy with logits, applied per-threshold, averaged over
    thresholds and over the batch.

    using BCEWithLogitsLoss instead of sigmoid + BCE separately because
    the combined version is more numerically stable — it uses the
    log-sum-exp trick internally.

    reduction='mean' averages over all (batch, threshold) entries, which
    keeps the loss magnitude stable regardless of batch size or
    num_thresholds. matches the convention in the Niu et al. paper.
    """

    def __init__(self, num_thresholds: int = NUM_THRESHOLDS) -> None:
        super().__init__()
        self.num_thresholds = num_thresholds
        # built-in loss handles the sigmoid + BCE combo in one stable pass
        self.bce = nn.BCEWithLogitsLoss(reduction="mean")

    def forward(self, logits: torch.Tensor, ages: torch.Tensor) -> torch.Tensor:
        """
        logits: (B, num_thresholds) — raw model output
        ages:   (B,)                — integer true ages
        returns: scalar loss tensor
        """
        # build the binary target matrix on the fly — keeps the dataset
        # simpler (it only needs to return the integer age)
        target = age_to_ordinal_target(ages, self.num_thresholds)

        # standard BCE-with-logits — internal sigmoid keeps gradients stable
        loss = self.bce(logits, target)
        return loss
