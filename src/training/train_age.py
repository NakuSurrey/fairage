"""
training script for the age estimator.

invoked by SLURM via slurm/train_age.sh on Surrey HPC.
also runnable on a laptop with --epochs 1 for a smoke test.

what it does:
    1. load UTKFace, split 80/10/10 into train/val/test (seeded for reproducibility)
    2. build DataLoaders with the train and inference transforms
    3. train ResNet-50 + regression head, log every epoch to Weights & Biases
    4. save the best checkpoint by validation MAE
    5. run final evaluation on the test set, log MAE and RMSE
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

# repo root on sys.path so `src.*` imports work when SLURM runs us
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.config import (
    BATCH_SIZE,
    CHECKPOINTS_DIR,
    EARLY_STOPPING_PATIENCE,
    LEARNING_RATE,
    NUM_EPOCHS,
    NUM_WORKERS,
    SEED,
    WEIGHT_DECAY,
)
from src.data.transforms import get_inference_transform, get_train_transform
from src.data.utkface_dataset import UTKFaceDataset
from src.models.age_estimator import AgeEstimator, count_parameters
from src.training.evaluate import evaluate


# ---------- reproducibility ----------

def set_seed(seed: int) -> None:
    """seed every RNG so train/val/test split is the same on every run."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------- data splitting ----------

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


# ---------- training loop ----------

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
) -> dict[str, float]:
    """one full pass over the training set. returns mean loss and MAE."""
    model.train()  # enable dropout, batchnorm uses batch stats

    total_loss = 0.0
    total_abs_err = 0.0
    total_samples = 0

    pbar = tqdm(loader, desc=f"epoch {epoch} train", leave=False)
    for batch in pbar:
        images = batch["image"].to(device, non_blocking=True)
        ages = batch["age"].float().to(device, non_blocking=True)

        # forward pass
        preds = model(images)
        loss = criterion(preds, ages)

        # backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # running stats
        bs = images.size(0)
        total_loss += loss.item() * bs
        total_abs_err += (preds - ages).abs().sum().item()
        total_samples += bs

        pbar.set_postfix(loss=f"{loss.item():.3f}")

    return {
        "loss": total_loss / total_samples,
        "mae": total_abs_err / total_samples,
    }


# ---------- main ----------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS,
                        help="override config NUM_EPOCHS — useful for laptop smoke tests")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--no-wandb", action="store_true",
                        help="skip Weights & Biases logging — for offline / laptop runs")
    parser.add_argument("--checkpoint-name", type=str, default="age_best.pt")
    args = parser.parse_args()

    set_seed(SEED)

    # ---------- device ----------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[setup] device = {device}")
    if device.type == "cuda":
        print(f"[setup] gpu = {torch.cuda.get_device_name(0)}")

    # ---------- W&B (optional) ----------
    wandb_run = None
    if not args.no_wandb:
        try:
            import wandb
            wandb_run = wandb.init(
                project=os.environ.get("WANDB_PROJECT", "fairage"),
                entity=os.environ.get("WANDB_ENTITY"),
                config={
                    "epochs": args.epochs,
                    "batch_size": args.batch_size,
                    "lr": args.lr,
                    "weight_decay": WEIGHT_DECAY,
                    "model": "resnet50_regression_head",
                    "phase": "3_baseline",
                },
            )
        except Exception as exc:
            print(f"[wandb] init failed, continuing without logging: {exc}")
            wandb_run = None

    # ---------- data ----------
    print("[data] loading UTKFace…")
    base_ds = UTKFaceDataset(transform=None)
    print(f"[data] {len(base_ds):,} valid samples ({base_ds.skipped_count} skipped)")

    train_subset, val_subset, test_subset = split_dataset(base_ds, SEED)

    # apply the right transform to each split — train gets augmentation, val/test do not
    # Subset wraps the same base_ds, so swapping transforms via attribute works
    train_ds = UTKFaceDataset(transform=get_train_transform())
    eval_ds = UTKFaceDataset(transform=get_inference_transform())

    # rebuild Subsets against the transformed datasets using the same indices
    train_loader = DataLoader(
        Subset(train_ds, train_subset.indices),
        batch_size=args.batch_size, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        Subset(eval_ds, val_subset.indices),
        batch_size=args.batch_size, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=(device.type == "cuda"),
    )
    test_loader = DataLoader(
        Subset(eval_ds, test_subset.indices),
        batch_size=args.batch_size, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=(device.type == "cuda"),
    )
    print(f"[data] train={len(train_loader.dataset)} val={len(val_loader.dataset)} test={len(test_loader.dataset)}")

    # ---------- model ----------
    model = AgeEstimator(pretrained=True).to(device)
    pc = count_parameters(model)
    print(f"[model] params total={pc['total']:,} trainable={pc['trainable']:,}")

    # ---------- loss + optimizer ----------
    # plain MSE for the baseline — Phase 4 swaps in ordinal MAE+MSE loss
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=WEIGHT_DECAY,
    )

    # ---------- training ----------
    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_path = CHECKPOINTS_DIR / args.checkpoint_name

    best_val_mae = float("inf")
    epochs_no_improve = 0
    history: list[dict] = []

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_stats = train_one_epoch(model, train_loader, optimizer, criterion, device, epoch)
        val_result = evaluate(model, val_loader, device)
        epoch_secs = time.time() - t0

        log = {
            "epoch": epoch,
            "train/loss": train_stats["loss"],
            "train/mae": train_stats["mae"],
            "val/mae": val_result.mae,
            "val/rmse": val_result.rmse,
            "epoch_secs": epoch_secs,
        }
        history.append(log)
        print(
            f"[epoch {epoch:02d}/{args.epochs}] "
            f"train_loss={log['train/loss']:.3f} train_mae={log['train/mae']:.2f} "
            f"val_mae={log['val/mae']:.2f} val_rmse={log['val/rmse']:.2f} "
            f"({epoch_secs:.1f}s)"
        )
        if wandb_run is not None:
            wandb_run.log(log)

        # early stopping + best checkpoint by validation MAE
        if val_result.mae < best_val_mae:
            best_val_mae = val_result.mae
            epochs_no_improve = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "val_mae": val_result.mae,
                "val_rmse": val_result.rmse,
                "config": {
                    "lr": args.lr,
                    "batch_size": args.batch_size,
                    "weight_decay": WEIGHT_DECAY,
                },
            }, ckpt_path)
            print(f"  -> new best, saved to {ckpt_path}")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= EARLY_STOPPING_PATIENCE:
                print(f"[early-stop] no improvement for {EARLY_STOPPING_PATIENCE} epochs — stopping")
                break

    # ---------- final test evaluation ----------
    print(f"[test] loading best checkpoint from {ckpt_path}")
    state = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(state["model_state_dict"])
    test_result = evaluate(model, test_loader, device)
    print(f"[test] mae={test_result.mae:.2f} rmse={test_result.rmse:.2f} n={test_result.num_samples}")

    if wandb_run is not None:
        wandb_run.summary["test/mae"] = test_result.mae
        wandb_run.summary["test/rmse"] = test_result.rmse
        wandb_run.finish()

    # write a small json next to the checkpoint — survives wandb being offline
    summary_path = CHECKPOINTS_DIR / args.checkpoint_name.replace(".pt", "_summary.json")
    with open(summary_path, "w") as f:
        json.dump({
            "best_val_mae": best_val_mae,
            "test_mae": test_result.mae,
            "test_rmse": test_result.rmse,
            "epochs_run": len(history),
            "history": history,
        }, f, indent=2)
    print(f"[done] summary written to {summary_path}")


if __name__ == "__main__":
    main()
