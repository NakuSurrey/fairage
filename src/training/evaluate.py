"""
evaluation function used by training and by the bias audit (Phase 5).

returns val loss + val MAE. MAE is the headline metric for age
estimation — interpretable as "average error in years".

Phase 4 change: decodes ordinal logits to predicted age before computing
MAE, so train-time metric and inference-time output line up exactly.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.models.ordinal_loss import ordinal_logits_to_age


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
) -> dict:
    """
    forward pass over the eval loader. no gradients (no_grad decorator).

    returns:
        val_loss — average ordinal loss over the eval set
        val_mae  — mean absolute error in years between decoded ages
                   and true ages
    """
    model.eval()
    running_loss = 0.0
    running_mae = 0.0
    seen = 0

    for images, ages in loader:
        images = images.to(device, non_blocking=True)
        ages = ages.to(device, non_blocking=True)

        # forward only — no backprop, no optimizer step
        logits = model(images)

        # loss is computed on the raw logits (BCEWithLogitsLoss handles sigmoid)
        loss = loss_fn(logits, ages)

        # MAE is computed on the decoded prediction so the metric reflects
        # the actual age the model would output at inference
        pred_ages = ordinal_logits_to_age(logits)
        abs_err = (pred_ages - ages.float()).abs().sum().item()

        bs = images.size(0)
        running_loss += loss.item() * bs
        running_mae += abs_err
        seen += bs

    return {
        "val_loss": running_loss / max(seen, 1),
        "val_mae": running_mae / max(seen, 1),
    }
