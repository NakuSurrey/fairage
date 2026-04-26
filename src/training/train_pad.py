"""
training script for the PAD detector.

invoked by SLURM via slurm/train_pad.sh on Surrey HPC.
also runnable on a laptop with --epochs 1 for a smoke test.

flow per epoch:
    1. train pass — forward, cross entropy, backprop, step
    2. eval pass  — forward, compute accuracy + HTER (half total error rate)
    3. log to W&B if a key is set, else log to local JSON
    4. save best checkpoint by val HTER (lower is better)

why HTER:
    PAD evaluation in the field uses HTER = (FAR + FRR) / 2 where
    FAR is false accept rate (attacks classified as real) and
    FRR is false reject rate (real classified as attack).
    accuracy alone hides class imbalance. HTER does not.
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
from src.data.pad_dataset import PADDataset
from src.data.transforms import get_train_transform, get_inference_transform
from src.models.pad_detector import PADDetector

# W&B is optional — gracefully degrade to local JSON logs if no key
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


def set_seed(seed: int) -> None:
    """seed every RNG so the train/val split is identical run to run."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def stratified_split(labels: list[int], seed: int,
                     val_ratio: float = 0.15, test_ratio: float = 0.15
                     ) -> tuple[list[int], list[int], list[int]]:
    """
    stratified split — preserves the real/attack ratio in every split.
    important when classes are not 50/50, which they often are not in NUAA.
    """
    rng = random.Random(seed)

    # bucket indices by class label so each split gets the same proportion
    buckets: dict[int, list[int]] = {}
    for idx, lbl in enumerate(labels):
        buckets.setdefault(lbl, []).append(idx)

    train_idx: list[int] = []
    val_idx: list[int] = []
    test_idx: list[int] = []

    for lbl, idxs in buckets.items():
        idxs = idxs.copy()
        rng.shuffle(idxs)
        n = len(idxs)
        n_test = int(n * test_ratio)
        n_val = int(n * val_ratio)
        test_idx.extend(idxs[:n_test])
        val_idx.extend(idxs[n_test:n_test + n_val])
        train_idx.extend(idxs[n_test + n_val:])

    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    rng.shuffle(test_idx)
    return train_idx, val_idx, test_idx


def compute_class_weights(labels: list[int], num_classes: int) -> torch.Tensor:
    """
    inverse-frequency class weights for CrossEntropyLoss.
    handles class imbalance without resampling — preserves the natural
    distribution while telling the loss to weight rare classes higher.
    """
    counts = [max(labels.count(c), 1) for c in range(num_classes)]
    total = sum(counts)
    # weight = total / (num_classes * count) — standard sklearn formula
    weights = [total / (num_classes * c) for c in counts]
    return torch.tensor(weights, dtype=torch.float32)


def hter(preds: torch.Tensor, targets: torch.Tensor) -> dict:
    """
    half total error rate.

    FAR = false accept rate  = attacks classified as real / total attacks
    FRR = false reject rate  = real classified as attack / total real
    HTER = (FAR + FRR) / 2

    returns all three for the dashboard / report.
    """
    real_mask = (targets == 0)
    attack_mask = (targets == 1)

    n_real = real_mask.sum().item()
    n_attack = attack_mask.sum().item()

    # FRR — real samples wrongly flagged as attack
    if n_real > 0:
        frr = ((preds[real_mask] == 1).sum().item() / n_real)
    else:
        frr = 0.0

    # FAR — attack samples wrongly flagged as real
    if n_attack > 0:
        far = ((preds[attack_mask] == 0).sum().item() / n_attack)
    else:
        far = 0.0

    return {
        "far": far,
        "frr": frr,
        "hter": (far + frr) / 2.0,
    }


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> dict:
    """one full training pass. returns mean loss and accuracy."""
    model.train()
    running_loss = 0.0
    correct = 0
    seen = 0

    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)

        optimizer.zero_grad()
        logits = model(images)
        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()

        bs = images.size(0)
        running_loss += loss.item() * bs
        with torch.no_grad():
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
        seen += bs

    return {
        "train_loss": running_loss / max(seen, 1),
        "train_acc": correct / max(seen, 1),
    }


@torch.no_grad()
def evaluate_pad(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
) -> dict:
    """eval pass — loss, accuracy, FAR, FRR, HTER."""
    model.eval()
    running_loss = 0.0
    correct = 0
    seen = 0
    all_preds: list[torch.Tensor] = []
    all_targets: list[torch.Tensor] = []

    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)

        logits = model(images)
        loss = loss_fn(logits, labels)

        bs = images.size(0)
        running_loss += loss.item() * bs
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        seen += bs

        all_preds.append(preds.cpu())
        all_targets.append(labels.cpu())

    preds = torch.cat(all_preds)
    targets = torch.cat(all_targets)
    rates = hter(preds, targets)

    return {
        "val_loss": running_loss / max(seen, 1),
        "val_acc": correct / max(seen, 1),
        **{f"val_{k}": v for k, v in rates.items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default=None,
                        help="path to PAD folder (defaults to config.PAD_DIR)")
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY)
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS)
    parser.add_argument("--output-dir", type=str, default=str(CHECKPOINTS_DIR))
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--wandb-project", type=str, default="fairage-pad")
    args = parser.parse_args()

    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # build the dataset once with no transform so we can split deterministically,
    # then build two transformed copies and slice them by index for the loaders
    base_ds = PADDataset(root=args.data_dir, transform=None)
    labels = base_ds.get_label_array()
    counts = base_ds.class_counts()
    print(f"PAD class balance: {counts}")

    train_idx, val_idx, test_idx = stratified_split(labels, args.seed)

    train_ds_full = PADDataset(root=args.data_dir, transform=get_train_transform())
    eval_ds_full = PADDataset(root=args.data_dir, transform=get_inference_transform())
    train_ds = Subset(train_ds_full, train_idx)
    val_ds = Subset(eval_ds_full, val_idx)
    test_ds = Subset(eval_ds_full, test_idx)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers,
                              pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers,
                            pin_memory=torch.cuda.is_available())
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers,
                             pin_memory=torch.cuda.is_available())

    # model + loss + optimizer + scheduler
    model = PADDetector(num_classes=2).to(device)
    class_weights = compute_class_weights(labels, num_classes=2).to(device)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    use_wandb = WANDB_AVAILABLE and os.environ.get("WANDB_API_KEY")
    if use_wandb:
        wandb.init(project=args.wandb_project, config=vars(args))

    # checkpoint by lowest val HTER — that is the operational metric
    best_hter = float("inf")
    history = []

    for epoch in range(1, args.epochs + 1):
        train_stats = train_one_epoch(model, train_loader, loss_fn, optimizer, device)
        val_stats = evaluate_pad(model, val_loader, loss_fn, device)
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

        if val_stats["val_hter"] < best_hter:
            best_hter = val_stats["val_hter"]
            torch.save({
                "model_state": model.state_dict(),
                "epoch": epoch,
                "val_hter": best_hter,
                "args": vars(args),
            }, output_dir / "pad_model_best.pt")

    # final test eval — single set of numbers used in the report
    test_stats = evaluate_pad(model, test_loader, loss_fn, device)
    print(f"final test stats: {test_stats}")

    torch.save({
        "model_state": model.state_dict(),
        "epoch": args.epochs,
        "val_hter": history[-1]["val_hter"],
        "test_hter": test_stats["val_hter"],
        "args": vars(args),
    }, output_dir / "pad_model_final.pt")

    log_path = output_dir / "pad_training_history.json"
    with log_path.open("w", encoding="utf-8") as f:
        json.dump({
            "config": vars(args),
            "best_val_hter": best_hter,
            "test_metrics": test_stats,
            "class_balance": counts,
            "history": history,
        }, f, indent=2)

    if use_wandb:
        wandb.finish()

    print(f"done — best val HTER {best_hter:.4f}, "
          f"test HTER {test_stats['val_hter']:.4f}, "
          f"history saved to {log_path}")


if __name__ == "__main__":
    main()
