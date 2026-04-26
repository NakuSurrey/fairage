"""
training loop for the ordinal age estimator.

Phase 3 used MSE loss directly on a single regression output.
Phase 4 uses OrdinalRegressionLoss on 100 binary thresholds, and
decodes logits -> predicted age for the MAE metric.

flow per epoch:
    1. train pass — forward, compute ordinal loss, backprop, step
    2. eval pass  — forward, decode logits to ages, compute MAE
    3. log to W&B if a key is set, else log to local JSON
    4. save best checkpoint by val MAE
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from src.config import (
    BATCH_SIZE,
    DEVICE,
    LEARNING_RATE,
    NUM_EPOCHS,
    NUM_WORKERS,
    SEED,
    WEIGHT_DECAY,
)
from src.data.utkface_dataset import UTKFaceDataset
from src.data.transforms import build_train_transforms, build_eval_transforms
from src.models.age_estimator import AgeEstimator
from src.models.ordinal_loss import OrdinalRegressionLoss, ordinal_logits_to_age
from src.training.evaluate import evaluate_model

# W&B is optional — gracefully degrade to local JSON logs if no key
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


def set_seed(seed: int) -> None:
    """make the run reproducible — seed every random source we touch."""
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> dict:
    """
    one full pass over the training set.

    returns a dict with the running averages — train loss and train MAE
    (decoded from logits, gives a quick sanity check vs the val MAE).
    """
    model.train()
    running_loss = 0.0
    running_mae = 0.0
    seen = 0

    for images, ages in loader:
        images = images.to(device, non_blocking=True)
        ages = ages.to(device, non_blocking=True)

        # forward -> loss -> backprop -> step
        # zero_grad first so old gradients from the last batch don't pile up
        optimizer.zero_grad()
        logits = model(images)
        loss = loss_fn(logits, ages)
        loss.backward()
        optimizer.step()

        # track running stats — multiply by batch size so the final divide
        # by `seen` gives a true per-sample average even on uneven last batch
        bs = images.size(0)
        running_loss += loss.item() * bs

        # decode for MAE — same code path the eval uses
        with torch.no_grad():
            pred_ages = ordinal_logits_to_age(logits)
            running_mae += (pred_ages - ages.float()).abs().sum().item()

        seen += bs

    return {
        "train_loss": running_loss / max(seen, 1),
        "train_mae": running_mae / max(seen, 1),
    }


def main() -> None:
    # CLI args — keep defaults coming from config.py, override per run
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, required=True,
                        help="path to UTKFace folder")
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY)
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS)
    parser.add_argument("--output-dir", type=str, default="artifacts")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--no-pretrained", action="store_true",
                        help="disable ImageNet weights — used by smoke tests")
    parser.add_argument("--wandb-project", type=str, default="fairage")
    args = parser.parse_args()

    # reproducibility — set seed before any random op
    set_seed(args.seed)

    # set up output directory for checkpoints + local JSON log
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # device — GPU if visible, else CPU. SLURM job gets GPU, local tests get CPU.
    device = torch.device(DEVICE)

    # datasets + loaders. UTKFaceDataset returns (image_tensor, age_int).
    # train transforms add augmentation, eval transforms only resize+normalise.
    train_ds = UTKFaceDataset(
        root=args.data_dir,
        split="train",
        transform=build_train_transforms(),
    )
    val_ds = UTKFaceDataset(
        root=args.data_dir,
        split="val",
        transform=build_eval_transforms(),
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    # model + loss + optimizer + scheduler
    model = AgeEstimator(pretrained=not args.no_pretrained).to(device)
    loss_fn = OrdinalRegressionLoss().to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    # W&B init — optional. if no key in env, fall back to local JSON.
    use_wandb = WANDB_AVAILABLE and os.environ.get("WANDB_API_KEY")
    if use_wandb:
        wandb.init(project=args.wandb_project, config=vars(args))

    # training loop — track best val MAE across epochs to decide checkpoints
    best_mae = float("inf")
    history = []

    for epoch in range(1, args.epochs + 1):
        train_stats = train_one_epoch(model, train_loader, loss_fn, optimizer, device)
        val_stats = evaluate_model(model, val_loader, loss_fn, device)
        scheduler.step()

        # combine stats for logging — flat dict makes W&B charts simple
        epoch_stats = {
            "epoch": epoch,
            "lr": optimizer.param_groups[0]["lr"],
            **train_stats,
            **val_stats,
        }
        history.append(epoch_stats)
        print(f"[epoch {epoch}/{args.epochs}] {epoch_stats}")

        if use_wandb:
            wandb.log(epoch_stats)

        # save best checkpoint by val MAE — overwrites previous best
        if val_stats["val_mae"] < best_mae:
            best_mae = val_stats["val_mae"]
            ckpt_path = output_dir / "age_model_best.pt"
            torch.save({
                "model_state": model.state_dict(),
                "epoch": epoch,
                "val_mae": best_mae,
                "args": vars(args),
            }, ckpt_path)

    # always save the final epoch too — useful for resuming and for diffs
    final_path = output_dir / "age_model_final.pt"
    torch.save({
        "model_state": model.state_dict(),
        "epoch": args.epochs,
        "val_mae": history[-1]["val_mae"],
        "args": vars(args),
    }, final_path)

    # local JSON log — written every run, regardless of W&B status
    log_path = output_dir / "training_history.json"
    with log_path.open("w", encoding="utf-8") as f:
        json.dump({
            "config": vars(args),
            "best_val_mae": best_mae,
            "history": history,
        }, f, indent=2)

    if use_wandb:
        wandb.finish()

    print(f"done — best val MAE {best_mae:.3f} years, history saved to {log_path}")


if __name__ == "__main__":
    main()
