"""
evaluation function used by training and by the bias audit (Phase 5).

returns val loss + val MAE + per-sample arrays (preds, targets, genders,
ethnicities) so the bias audit can slice without re-running the model.

Phase 4: decodes ordinal logits to predicted age before computing MAE,
so train-time metric and inference-time output line up exactly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.models.ordinal_loss import ordinal_logits_to_age


@dataclass
class EvalArrays:
    """per-sample arrays kept on CPU for downstream bias slicing."""
    preds: torch.Tensor       # shape (N,), float — decoded predicted ages
    targets: torch.Tensor     # shape (N,), float — true ages
    genders: torch.Tensor     # shape (N,), long
    ethnicities: torch.Tensor # shape (N,), long


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    return_arrays: bool = False,
) -> dict:
    """
    forward pass over the eval loader. no gradients (no_grad decorator).

    args:
        model        — AgeEstimator (or any nn.Module returning logits)
        loader       — DataLoader yielding dicts with image/age/gender/ethnicity
        loss_fn      — OrdinalRegressionLoss
        device       — torch.device
        return_arrays — if True, also return per-sample arrays for bias audit

    returns dict with:
        val_loss — average ordinal loss over the eval set
        val_mae  — mean absolute error in years between decoded ages and true ages
        arrays   — EvalArrays object, only present if return_arrays=True
    """
    model.eval()
    running_loss = 0.0
    running_mae = 0.0
    seen = 0

    all_preds: list[torch.Tensor] = []
    all_targets: list[torch.Tensor] = []
    all_genders: list[torch.Tensor] = []
    all_ethnicities: list[torch.Tensor] = []

    for batch in loader:
        # dataset returns a dict — unpack by key, not tuple position
        images = batch["image"].to(device, non_blocking=True)
        ages = batch["age"].to(device, non_blocking=True)

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

        if return_arrays:
            # pull tensors back to CPU before the next batch — GPU memory stays bounded
            all_preds.append(pred_ages.detach().cpu())
            all_targets.append(ages.detach().cpu().float())
            all_genders.append(batch["gender"].cpu())
            all_ethnicities.append(batch["ethnicity"].cpu())

    out = {
        "val_loss": running_loss / max(seen, 1),
        "val_mae": running_mae / max(seen, 1),
    }

    if return_arrays:
        out["arrays"] = EvalArrays(
            preds=torch.cat(all_preds),
            targets=torch.cat(all_targets),
            genders=torch.cat(all_genders),
            ethnicities=torch.cat(all_ethnicities),
        )

    return out
