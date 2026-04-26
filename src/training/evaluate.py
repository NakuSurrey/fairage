"""
evaluation — runs a model on a dataloader, returns MAE and per-sample records.

used by:
    - training loop, to compute validation MAE every epoch
    - bias audit (Phase 5), to slice predictions by gender/ethnicity/age bucket
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader


@dataclass
class EvalResult:
    mae: float
    rmse: float
    num_samples: int
    # per-sample arrays — used by bias audit, kept on CPU
    preds: torch.Tensor      # shape [N], float
    targets: torch.Tensor    # shape [N], float
    genders: torch.Tensor    # shape [N], long
    ethnicities: torch.Tensor  # shape [N], long


@torch.no_grad()
def evaluate(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> EvalResult:
    """
    run model on every batch in `loader`, no gradients.
    returns aggregate MAE/RMSE plus per-sample arrays for downstream slicing.
    """
    model.eval()  # turn off dropout, freeze batchnorm running stats

    all_preds: list[torch.Tensor] = []
    all_targets: list[torch.Tensor] = []
    all_genders: list[torch.Tensor] = []
    all_ethnicities: list[torch.Tensor] = []

    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        ages = batch["age"].float().to(device, non_blocking=True)

        preds = model(images)

        # pull tensors back to CPU before the next batch — GPU memory stays bounded
        all_preds.append(preds.detach().cpu())
        all_targets.append(ages.detach().cpu())
        all_genders.append(batch["gender"].cpu())
        all_ethnicities.append(batch["ethnicity"].cpu())

    preds = torch.cat(all_preds)
    targets = torch.cat(all_targets)
    genders = torch.cat(all_genders)
    ethnicities = torch.cat(all_ethnicities)

    abs_err = (preds - targets).abs()
    mae = abs_err.mean().item()
    rmse = ((preds - targets) ** 2).mean().sqrt().item()

    return EvalResult(
        mae=mae,
        rmse=rmse,
        num_samples=len(targets),
        preds=preds,
        targets=targets,
        genders=genders,
        ethnicities=ethnicities,
    )
