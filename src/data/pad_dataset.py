"""
PAD (Presentation Attack Detection) dataset class.

NUAA Imposter Database — chosen in Phase 6 over CelebA-Spoof (~600GB) and
Replay-Attack DB (institutional license required). NUAA is small, free,
and exercises the full PAD pipeline end to end.

binary classification: real face vs print attack.

expected folder layout (either form is accepted):

    data/pad/
    ├── real/         # genuine face photos -> label 0
    └── attack/       # printed photo attacks -> label 1

or the original NUAA folder names:

    data/pad/
    ├── ClientFace/    # genuine -> label 0
    └── ImposterFace/  # attack -> label 1

label encoding:
    0 = real (genuine face)
    1 = attack (presentation attack)

this matches the convention used by every PAD evaluation toolkit so the
HTER / EER metrics computed downstream stay comparable to published numbers.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import torch
from PIL import Image
from torch.utils.data import Dataset

from src.config import PAD_DIR

# accepted folder names. first match wins for each class.
REAL_FOLDER_NAMES = ("real", "ClientFace", "client", "genuine")
ATTACK_FOLDER_NAMES = ("attack", "ImposterFace", "imposter", "spoof", "print")

# valid image extensions in NUAA
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")


@dataclass
class PADSample:
    """one parsed PAD sample — image path plus its binary label."""
    path: Path
    label: int  # 0 = real, 1 = attack


def _resolve_class_folder(root: Path, candidates: tuple[str, ...]) -> Optional[Path]:
    """
    return the first existing subfolder of `root` that matches any name in
    `candidates`. case-insensitive on the folder name.

    returns None if none of the candidates exist — caller decides what to do.
    """
    if not root.exists():
        return None

    # build a lowercase->actual mapping of children once, then look up
    children = {p.name.lower(): p for p in root.iterdir() if p.is_dir()}
    for cand in candidates:
        match = children.get(cand.lower())
        if match is not None:
            return match
    return None


def _collect_images(folder: Path) -> list[Path]:
    """
    return every image file under `folder`, recursively. NUAA nests photos
    inside per-subject subfolders, so a flat glob is not enough.
    """
    images: list[Path] = []
    for ext in IMAGE_EXTENSIONS:
        images.extend(folder.rglob(f"*{ext}"))
    # sort for determinism — same order on every run, every machine
    return sorted(images)


class PADDataset(Dataset):
    """
    PyTorch Dataset for binary PAD (real vs attack).

    returns dict per item:
        image  — transformed tensor, shape [3, H, W]
        label  — long tensor, scalar (0 = real, 1 = attack)
        path   — original file path as string (for debugging)
    """

    def __init__(
        self,
        root: Path | str | None = None,
        transform: Optional[Callable] = None,
        sample_paths: Optional[list[tuple[Path, int]]] = None,
    ):
        # allow caller to pass a custom root for tests, otherwise use config default
        self.root = Path(root) if root is not None else PAD_DIR
        self.transform = transform

        if sample_paths is not None:
            # explicit list — used by train/val splitting once base set is parsed
            self.samples = [PADSample(path=Path(p), label=int(lbl)) for p, lbl in sample_paths]
        else:
            if not self.root.exists():
                raise FileNotFoundError(
                    f"PAD folder not found at {self.root}. "
                    f"see data/README.md for download steps."
                )
            self.samples = self._scan(self.root)

    @staticmethod
    def _scan(root: Path) -> list[PADSample]:
        """
        find the real and attack subfolders, list all images under each,
        produce a flat list of PADSample with the right labels.
        """
        real_dir = _resolve_class_folder(root, REAL_FOLDER_NAMES)
        attack_dir = _resolve_class_folder(root, ATTACK_FOLDER_NAMES)

        if real_dir is None or attack_dir is None:
            raise FileNotFoundError(
                f"PAD folder {root} must contain a real-face subfolder "
                f"(one of: {REAL_FOLDER_NAMES}) and an attack subfolder "
                f"(one of: {ATTACK_FOLDER_NAMES}). see data/README.md."
            )

        samples: list[PADSample] = []
        # label 0 for real images
        for p in _collect_images(real_dir):
            samples.append(PADSample(path=p, label=0))
        # label 1 for attack images
        for p in _collect_images(attack_dir):
            samples.append(PADSample(path=p, label=1))

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]

        # PIL handles JPEG/PNG/BMP decode; convert to RGB to drop alpha if present
        # NUAA contains some grayscale images — convert("RGB") handles those too
        image = Image.open(sample.path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return {
            "image": image,
            "label": torch.tensor(sample.label, dtype=torch.long),
            "path": str(sample.path),
        }

    def get_label_array(self) -> list[int]:
        """
        return all labels as a flat list — used for stratified splitting
        and class-balance reporting without loading any images.
        """
        return [s.label for s in self.samples]

    def class_counts(self) -> dict[str, int]:
        """quick balance check — used by EDA notebook and the training script."""
        labels = self.get_label_array()
        return {
            "real": labels.count(0),
            "attack": labels.count(1),
        }
