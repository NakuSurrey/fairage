"""
UTKFace dataset class.

reads images from data/utkface/ and parses age, gender, ethnicity from each filename.
filename format: [age]_[gender]_[race]_[date].jpg.chip.jpg
example: 25_0_2_20170104203021098.jpg.chip.jpg
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import torch
from PIL import Image
from torch.utils.data import Dataset

from src.config import AGE_MAX, AGE_MIN, UTKFACE_DIR


@dataclass
class UTKSample:
    """one parsed sample — image path plus its labels."""
    path: Path
    age: int
    gender: int     # 0 = male, 1 = female
    ethnicity: int  # 0 = white, 1 = black, 2 = asian, 3 = indian, 4 = others


def _parse_filename(path: Path) -> Optional[UTKSample]:
    """
    parse one UTKFace filename into a sample.
    returns None if the filename is malformed — some files in the dataset
    are missing fields and would crash training otherwise.
    """
    # filename has dots inside (.jpg.chip.jpg) — split only on underscores
    stem = path.name.split(".")[0]  # drop .jpg.chip.jpg suffix
    parts = stem.split("_")

    # need at least age, gender, race — date is optional but usually present
    if len(parts) < 3:
        return None

    try:
        age = int(parts[0])
        gender = int(parts[1])
        ethnicity = int(parts[2])
    except ValueError:
        # any non-integer field means a corrupt name — skip it
        return None

    # sanity-check ranges before returning
    if not (AGE_MIN <= age <= AGE_MAX):
        return None
    if gender not in (0, 1):
        return None
    if ethnicity not in (0, 1, 2, 3, 4):
        return None

    return UTKSample(path=path, age=age, gender=gender, ethnicity=ethnicity)


class UTKFaceDataset(Dataset):
    """
    PyTorch Dataset for UTKFace age estimation.

    returns dict per item:
        image     — transformed tensor, shape [3, H, W]
        age       — int tensor, scalar
        gender    — int tensor, scalar
        ethnicity — int tensor, scalar
        path      — original file path as string (for debugging / audit)
    """

    def __init__(
        self,
        root: Path | str | None = None,
        transform: Optional[Callable] = None,
        sample_paths: Optional[list[Path]] = None,
    ):
        # allow caller to pass a custom root for tests, otherwise use config default
        self.root = Path(root) if root is not None else UTKFACE_DIR
        self.transform = transform

        if sample_paths is not None:
            # explicit list — used by train/val/test splitting
            paths = [Path(p) for p in sample_paths]
        else:
            # scan the folder for every .jpg
            if not self.root.exists():
                raise FileNotFoundError(
                    f"UTKFace folder not found at {self.root}. "
                    f"see data/README.md for download steps."
                )
            paths = sorted(self.root.glob("*.jpg"))

        # parse every filename, drop ones that fail validation
        self.samples: list[UTKSample] = []
        skipped = 0
        for p in paths:
            sample = _parse_filename(p)
            if sample is None:
                skipped += 1
                continue
            self.samples.append(sample)

        # store skip count for logging — not raised as error, some noise is normal
        self.skipped_count = skipped

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]

        # PIL handles JPEG decode; convert to RGB to drop alpha channel if present
        image = Image.open(sample.path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return {
            "image": image,
            "age": torch.tensor(sample.age, dtype=torch.long),
            "gender": torch.tensor(sample.gender, dtype=torch.long),
            "ethnicity": torch.tensor(sample.ethnicity, dtype=torch.long),
            "path": str(sample.path),
        }

    def get_label_arrays(self) -> dict[str, list[int]]:
        """
        return all labels as plain lists — used by EDA notebook and bias audit
        without having to load images.
        """
        return {
            "age": [s.age for s in self.samples],
            "gender": [s.gender for s in self.samples],
            "ethnicity": [s.ethnicity for s in self.samples],
        }
