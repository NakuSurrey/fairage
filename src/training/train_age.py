"""
training script for the ordinal age estimator.

invoked by SLURM via slurm/train_age.sh on Surrey HPC.
also runnable on a laptop with --epochs 1 for a smoke test.

Phase 4 changes vs Phase 3:
    - OrdinalRegressionLoss replaces MSELoss
    - model output is (B, 100) logits instead of (B,) floats
    - decoded ages used for MAE so train metric matches inference path

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
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

# repo root on sys.path so `src.*` imports work when SLURM runs us
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.config import (
    BATCH_SIZE,
    CHECKPOINTS_DIR,
    LEARNING_RATE,
    NUM_EPOCHS,
    NUM_WORKERS,
    SEED,
    WEIGHT_DECAY,
)
from src.data.utkface_dataset import UTKFaceDataset
from src.data.transforms import get_train_transform, get_inference_transform
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
    """seed every RNG so train/val/test split is the same on every run."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def split_dataset(full_ds: UTKFaceDataset, seed: int) -> tuple[Subset, Subset, Subset]:
    """
    80/10/10 train/val/test split.
    using indices into the same dataset object — saves memory vs three full datasets.
    """
    n = len(full_ds)
    indices = list(range(n))
    rng = random.Random(seed)
    rng.shuffle(indices)

    n_train = int(0.8 * n)
    n_val = int(0.1 * n)

    train_idx = indices[:n_train]
    val_idx = indices[n_train : n_train + n_val]
    test_idx = indices[n_train + n_val :]

    return (Subset(full_ds, train_idx),
            Subset(full_ds, val_idx),
            Subset(full_ds, test_idx))


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> dict:
    """
    one full pass over the training set.
    returns running averages — train loss and train MAE (decoded from logits).
    """
    model.train()
    running_loss = 0.0
    running_mae = 0.0
    seen = 0

    for batch in loader:
        # dataset returns dicts — unpack by key, not by tuple position
        images = batch["image"].to(device, non_blocking=True)
        ages = batch["age"].to(device, non_blocking=True)

        # forward -> loss -> backprop -> step
        optimizer.zero_grad()
        logits = model(images)
        loss = loss_fn(logits, ages)
        loss.backward()
        optimizer.step()

        # multiply by batch size so the final divide gives a true per-sample mean
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default=None,
                        help="path to UTKFace folder (defaults to config.UTKFACE_DIR)")
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY)
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS)
    parser.add_argument("--output-dir", type=str, default=str(CHECKPOINTS_DIR))
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--no-pretrained", action="store_true",
                        help="disable ImageNet weights — used by smoke tests")
    parser.add_argument("--wandb-project", type=str, default="fairage")
    args = parser.parse_args()

    # reproducibility — set seed before any random op
    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # device — GPU if visible, else CPU. SLURM job gets GPU, local tests get CPU.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # one base dataset with no transforms, then split, then wrap each split with
    # the right transform via WrappedSubset below. this keeps the split deterministic
    # while still giving train its augmentations and val its deterministic pipeline.
    base_ds = UTKFaceDataset(root=args.data_dir, transform=None)
    train_subset, val_subset, test_subset = split_dataset(base_ds, args.seed)

    # transforms applied via a thin wrapper so train and val use different pipelines
    # without needing two copies of the underlying dataset
    class _Wrapped(torch.utils.data.Dataset):
        def __init__(self, subset, transform):
            self.subset = subset
            self.transform = transform

        def __len__(self):
            return len(self.subset)

        def __getitem__(self, idx):
            item = self.subset[idx]
            if self.transform is not None:
                item = dict(item)
                item["image"] = self.transform(item["image"])
            return item

    # base_ds returns PIL images (transform=None). re-open via the dataset object
    # so we can apply transforms here. easier path: build two datasets with different
    # transforms, share the same indices.
    train_ds_full = UTKFaceDataset(root=args.data_dir, transform=get_train_transform())
    eval_ds_full = UTKFaceDataset(root=args.data_dir, transform=get_inference_transform())

    train_ds = Subset(train_ds_full, train_subset.indices)
    val_ds = Subset(eval_ds_full, val_subset.indices)
    test_ds = Subset(eval_ds_full, test_subset.indices)

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
    test_loader = DataLoader(
        test_ds,
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

    # W&B init — optional. if no key in env, fall back to local JSON only.
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

        if val_stats["val_mae"] < best_mae:
            best_mae = val_stats["val_mae"]
            ckpt_path = output_dir / "age_model_best.pt"
            torch.save({
                "model_state": model.state_dict(),
                "epoch": epoch,
                "val_mae": best_mae,
                "args": vars(args),
            }, ckpt_path)

    # final test eval — single number used in README
    test_stats = evaluate_model(model, test_loader, loss_fn, device)
    print(f"final test stats: {test_stats}")

    final_path = output_dir / "age_model_final.pt"
    torch.save({
        "model_state": model.state_dict(),
        "epoch": args.epochs,
        "val_mae": history[-1]["val_mae"],
        "test_mae": test_stats["val_mae"],
        "args": vars(args),
    }, final_path)

    log_path = output_dir / "training_history.json"
    with log_path.open("w", encoding="utf-8") as f:
        json.dump({
            "config": vars(args),
            "best_val_mae": best_mae,
            "test_mae": test_stats["val_mae"],
            "history": history,
        }, f, indent=2)

    if use_wandb:
        wandb.finish()

    print(f"done — best val MAE {best_mae:.3f} years, "
          f"test MAE {test_stats['val_mae']:.3f} years, "
          f"history saved to {log_path}")


if __name__ == "__main__":
    main()
