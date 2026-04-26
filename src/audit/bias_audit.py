"""
bias audit — slices model predictions by demographic group and reports per-group MAE.

what gets audited:
    - per gender (male / female)
    - per ethnicity (white / black / asian / indian / others)
    - per age bucket (0-12 / 13-19 / 20-35 / 36-55 / 56-100)
    - intersectional gender x ethnicity grid
    - worst gap (overall MAE - worst-group MAE) — single headline number

why this matters:
    age estimation models trained on imbalanced data perform unevenly
    across groups. a model with overall MAE of 4.5 years can hide a
    9-year MAE on Asian children. that is the kind of failure ITL's
    ACCS compliance audit cares about, and the kind hiring panels
    want to see addressed.

output:
    1. bias_report.json — full structured numbers, one entry per group
    2. bias_report.md   — human-readable markdown table
    3. (optional) bias_chart.png — bar chart of per-group MAE
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import torch

from src.config import AGE_BUCKETS, ETHNICITY_LABELS, GENDER_LABELS


# ---------- types ----------


@dataclass
class GroupResult:
    """one row in the bias report — describes performance on one group slice."""
    group_type: str   # "gender" / "ethnicity" / "age_bucket" / "gender_x_ethnicity"
    group_name: str   # e.g. "Female", "Asian", "20-35", "Female_x_Asian"
    n_samples: int
    mae: float
    rmse: float


# ---------- core slicing ----------


def _bucket_index(age: int, buckets: list[tuple[int, int]] = AGE_BUCKETS) -> int:
    """return the index of the bucket that contains `age`, or -1 if none match."""
    for i, (lo, hi) in enumerate(buckets):
        if lo <= age <= hi:
            return i
    return -1


def _mae_rmse(preds: torch.Tensor, targets: torch.Tensor) -> tuple[float, float]:
    """compute MAE and RMSE for a slice. assumes tensors are on CPU."""
    if preds.numel() == 0:
        return float("nan"), float("nan")
    diff = preds.float() - targets.float()
    mae = diff.abs().mean().item()
    rmse = (diff ** 2).mean().sqrt().item()
    return mae, rmse


def compute_per_group_metrics(
    preds: torch.Tensor,
    targets: torch.Tensor,
    genders: torch.Tensor,
    ethnicities: torch.Tensor,
) -> list[GroupResult]:
    """
    slice the eval results by every demographic axis and return one
    GroupResult per group. the audit notebook and the JSON writer both
    consume this list.
    """
    # all four arrays must have the same length — one row per sample
    assert preds.shape == targets.shape == genders.shape == ethnicities.shape, (
        f"shape mismatch: preds={preds.shape}, targets={targets.shape}, "
        f"genders={genders.shape}, ethnicities={ethnicities.shape}"
    )

    results: list[GroupResult] = []

    # ---- by gender ----
    for code, name in GENDER_LABELS.items():
        mask = (genders == code)
        if mask.sum().item() == 0:
            continue
        mae, rmse = _mae_rmse(preds[mask], targets[mask])
        results.append(GroupResult(
            group_type="gender",
            group_name=name,
            n_samples=int(mask.sum().item()),
            mae=mae,
            rmse=rmse,
        ))

    # ---- by ethnicity ----
    for code, name in ETHNICITY_LABELS.items():
        mask = (ethnicities == code)
        if mask.sum().item() == 0:
            continue
        mae, rmse = _mae_rmse(preds[mask], targets[mask])
        results.append(GroupResult(
            group_type="ethnicity",
            group_name=name,
            n_samples=int(mask.sum().item()),
            mae=mae,
            rmse=rmse,
        ))

    # ---- by age bucket ----
    # build a bucket-index tensor once, then mask per bucket
    bucket_idx = torch.tensor(
        [_bucket_index(int(a.item())) for a in targets],
        dtype=torch.long,
    )
    for i, (lo, hi) in enumerate(AGE_BUCKETS):
        mask = (bucket_idx == i)
        if mask.sum().item() == 0:
            continue
        mae, rmse = _mae_rmse(preds[mask], targets[mask])
        results.append(GroupResult(
            group_type="age_bucket",
            group_name=f"{lo}-{hi}",
            n_samples=int(mask.sum().item()),
            mae=mae,
            rmse=rmse,
        ))

    # ---- intersectional: gender x ethnicity ----
    # this is where bias usually hides — overall numbers can look fine
    # while a small intersectional group has a 2x MAE
    for g_code, g_name in GENDER_LABELS.items():
        for e_code, e_name in ETHNICITY_LABELS.items():
            mask = (genders == g_code) & (ethnicities == e_code)
            if mask.sum().item() == 0:
                continue
            mae, rmse = _mae_rmse(preds[mask], targets[mask])
            results.append(GroupResult(
                group_type="gender_x_ethnicity",
                group_name=f"{g_name}_x_{e_name}",
                n_samples=int(mask.sum().item()),
                mae=mae,
                rmse=rmse,
            ))

    return results


# ---------- summary ----------


def overall_metrics(preds: torch.Tensor, targets: torch.Tensor) -> dict:
    """top-line MAE / RMSE / sample count for the whole eval set."""
    mae, rmse = _mae_rmse(preds, targets)
    return {
        "n_samples": int(preds.numel()),
        "mae": mae,
        "rmse": rmse,
    }


def worst_group_gap(overall_mae: float, group_results: list[GroupResult]) -> dict:
    """
    headline fairness number — how much worse is the worst group than the overall?

    only considers groups with at least 30 samples. small groups have noisy MAEs
    that should not drive the headline.
    """
    eligible = [g for g in group_results if g.n_samples >= 30]
    if not eligible:
        return {"worst_group": None, "worst_mae": None, "gap_years": None}

    worst = max(eligible, key=lambda g: g.mae)
    return {
        "worst_group": f"{worst.group_type}:{worst.group_name}",
        "worst_mae": worst.mae,
        "gap_years": worst.mae - overall_mae,
    }


# ---------- writers ----------


def write_json_report(
    overall: dict,
    group_results: list[GroupResult],
    worst_gap: dict,
    output_path: Path,
) -> None:
    """write the structured report — used by the dashboard and the API."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "overall": overall,
        "groups": [asdict(g) for g in group_results],
        "worst_gap": worst_gap,
    }
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def write_markdown_report(
    overall: dict,
    group_results: list[GroupResult],
    worst_gap: dict,
    output_path: Path,
) -> None:
    """
    human-readable markdown table — drops into README or a docs page.

    one section per group_type. samples count and MAE shown side by side
    so a reader can immediately see where small-sample noise might inflate
    or deflate a group MAE.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []
    lines.append("# FairAge — Bias Audit Report\n")
    lines.append(
        f"**Overall:** MAE = {overall['mae']:.2f} years, "
        f"RMSE = {overall['rmse']:.2f} years, "
        f"N = {overall['n_samples']:,}\n"
    )
    if worst_gap.get("worst_group"):
        lines.append(
            f"**Worst-group gap:** {worst_gap['worst_group']} — "
            f"MAE {worst_gap['worst_mae']:.2f} years, "
            f"+{worst_gap['gap_years']:.2f} years vs overall "
            f"(groups with N >= 30 only)\n"
        )

    # group rows by group_type so each demographic axis gets its own table
    types_in_order = ["gender", "ethnicity", "age_bucket", "gender_x_ethnicity"]
    for gtype in types_in_order:
        rows = [g for g in group_results if g.group_type == gtype]
        if not rows:
            continue
        # sort by MAE descending — worst groups float to the top
        rows = sorted(rows, key=lambda g: g.mae, reverse=True)

        title = gtype.replace("_", " ").title()
        lines.append(f"\n## {title}\n")
        lines.append("| Group | N | MAE (years) | RMSE (years) |")
        lines.append("|---|---:|---:|---:|")
        for r in rows:
            lines.append(
                f"| {r.group_name} | {r.n_samples:,} | "
                f"{r.mae:.2f} | {r.rmse:.2f} |"
            )

    with output_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


# ---------- end-to-end driver ----------


def run_bias_audit(
    preds: torch.Tensor,
    targets: torch.Tensor,
    genders: torch.Tensor,
    ethnicities: torch.Tensor,
    output_dir: Path,
) -> dict:
    """
    end-to-end audit. takes per-sample arrays from evaluate_model and writes
    bias_report.json + bias_report.md to output_dir. returns the structured
    payload for in-memory inspection (used by the audit notebook).
    """
    overall = overall_metrics(preds, targets)
    group_results = compute_per_group_metrics(preds, targets, genders, ethnicities)
    worst_gap = worst_group_gap(overall["mae"], group_results)

    write_json_report(overall, group_results, worst_gap, output_dir / "bias_report.json")
    write_markdown_report(overall, group_results, worst_gap, output_dir / "bias_report.md")

    return {
        "overall": overall,
        "groups": [asdict(g) for g in group_results],
        "worst_gap": worst_gap,
    }
