"""
unit tests for PAD pipeline.

builds a temporary folder of synthetic PAD images. tests run on CPU,
no GPU needed, no real NUAA dataset required. covers:

- PADDataset folder name resolution (real/attack and ClientFace/ImposterFace)
- PADDataset image scanning, label assignment, dict shape
- PADDetector forward pass, predict_class, predict_proba
- training helpers: stratified_split, compute_class_weights, hter
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from PIL import Image

from src.data.pad_dataset import PADDataset, _resolve_class_folder
from src.data.transforms import get_inference_transform
from src.models.pad_detector import PADDetector, count_parameters
from src.training.train_pad import (
    compute_class_weights,
    hter,
    stratified_split,
)


# ---------- helpers ----------

def _make_image(path: Path, color=(128, 64, 32), size=(180, 180)) -> None:
    """write a small RGB JPEG. content does not matter for these tests."""
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color=color).save(path, "JPEG")


@pytest.fixture
def fake_pad_dir_flat(tmp_path: Path) -> Path:
    """layout with the simple `real/` and `attack/` folder names."""
    root = tmp_path / "pad"
    for i in range(3):
        _make_image(root / "real" / f"r_{i}.jpg")
    for i in range(2):
        _make_image(root / "attack" / f"a_{i}.jpg")
    return root


@pytest.fixture
def fake_pad_dir_nuaa(tmp_path: Path) -> Path:
    """layout with NUAA's native folder names."""
    root = tmp_path / "nuaa"
    for i in range(4):
        _make_image(root / "ClientFace" / "subj_01" / f"c_{i}.jpg")
    for i in range(3):
        _make_image(root / "ImposterFace" / "subj_01" / f"i_{i}.jpg")
    return root


# ---------- folder resolution ----------

class TestFolderResolution:
    def test_finds_real_folder_flat(self, fake_pad_dir_flat: Path):
        match = _resolve_class_folder(fake_pad_dir_flat, ("real", "ClientFace"))
        assert match is not None
        assert match.name == "real"

    def test_finds_clientface_folder(self, fake_pad_dir_nuaa: Path):
        # NUAA-native name should also be found
        match = _resolve_class_folder(fake_pad_dir_nuaa, ("real", "ClientFace"))
        assert match is not None
        assert match.name == "ClientFace"

    def test_returns_none_when_not_found(self, tmp_path: Path):
        match = _resolve_class_folder(tmp_path, ("real", "ClientFace"))
        assert match is None


# ---------- dataset ----------

class TestPADDataset:
    def test_loads_flat_layout(self, fake_pad_dir_flat: Path):
        # 3 real + 2 attack -> 5 samples
        ds = PADDataset(root=fake_pad_dir_flat)
        assert len(ds) == 5
        assert ds.class_counts() == {"real": 3, "attack": 2}

    def test_loads_nuaa_layout(self, fake_pad_dir_nuaa: Path):
        # 4 real + 3 attack -> 7 samples, recursive scan into subj_01/
        ds = PADDataset(root=fake_pad_dir_nuaa)
        assert len(ds) == 7
        assert ds.class_counts() == {"real": 4, "attack": 3}

    def test_missing_folder_raises(self, tmp_path: Path):
        # nothing in the folder -> FileNotFoundError, not silent empty dataset
        empty = tmp_path / "empty_pad"
        empty.mkdir()
        with pytest.raises(FileNotFoundError):
            PADDataset(root=empty)

    def test_root_does_not_exist_raises(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            PADDataset(root=tmp_path / "no_such_dir")

    def test_getitem_returns_dict(self, fake_pad_dir_flat: Path):
        ds = PADDataset(root=fake_pad_dir_flat)
        item = ds[0]
        assert "image" in item
        assert "label" in item
        assert "path" in item

    def test_getitem_with_transform(self, fake_pad_dir_flat: Path):
        ds = PADDataset(root=fake_pad_dir_flat, transform=get_inference_transform())
        item = ds[0]
        # transform produces a tensor of shape (3, 224, 224) with the project IMAGE_SIZE
        assert isinstance(item["image"], torch.Tensor)
        assert item["image"].shape == (3, 224, 224)

    def test_label_dtype_is_long(self, fake_pad_dir_flat: Path):
        ds = PADDataset(root=fake_pad_dir_flat)
        item = ds[0]
        assert item["label"].dtype == torch.long

    def test_get_label_array_length_matches(self, fake_pad_dir_flat: Path):
        ds = PADDataset(root=fake_pad_dir_flat)
        labels = ds.get_label_array()
        assert len(labels) == len(ds)


# ---------- model ----------

class TestPADDetector:
    def test_output_shape(self):
        # default num_classes = 2 -> output shape (B, 2)
        model = PADDetector()
        x = torch.randn(4, 3, 224, 224)
        out = model(x)
        assert out.shape == (4, 2)

    def test_output_is_finite(self):
        # untrained network must still produce finite values
        model = PADDetector()
        x = torch.randn(2, 3, 224, 224)
        out = model(x)
        assert torch.isfinite(out).all().item()

    def test_predict_class_shape_and_values(self):
        # predict_class returns indices in {0, 1}
        model = PADDetector()
        x = torch.randn(4, 3, 224, 224)
        cls = model.predict_class(x)
        assert cls.shape == (4,)
        assert ((cls == 0) | (cls == 1)).all().item()

    def test_predict_proba_sums_to_one(self):
        # softmax over 2 classes -> per-row sum = 1
        model = PADDetector()
        x = torch.randn(3, 3, 224, 224)
        probs = model.predict_proba(x)
        assert probs.shape == (3, 2)
        sums = probs.sum(dim=1)
        assert torch.allclose(sums, torch.ones(3), atol=1e-5)

    def test_param_count_is_small(self):
        # design intent — small CNN, fewer than 1M params for ~12k-sample dataset
        model = PADDetector()
        n = count_parameters(model)
        assert n["total"] < 1_000_000
        assert n["trainable"] == n["total"]  # nothing frozen

    def test_gradients_flow(self):
        # full backward pass — confirms no detach in the network
        model = PADDetector()
        x = torch.randn(2, 3, 224, 224)
        out = model(x).sum()
        out.backward()
        assert any(p.grad is not None and p.grad.abs().sum().item() > 0
                   for p in model.parameters())


# ---------- training helpers ----------

class TestStratifiedSplit:
    def test_preserves_class_ratio(self):
        # 100 real (label 0), 50 attack (label 1) -> 2:1 ratio in every split
        labels = [0] * 100 + [1] * 50
        train, val, test = stratified_split(labels, seed=0,
                                            val_ratio=0.2, test_ratio=0.2)
        # each split should be roughly 2:1 real:attack
        for split in (train, val, test):
            real_count = sum(1 for i in split if labels[i] == 0)
            attack_count = sum(1 for i in split if labels[i] == 1)
            if attack_count > 0:
                ratio = real_count / attack_count
                assert 1.5 <= ratio <= 2.5  # loose bound for small splits

    def test_no_overlap_between_splits(self):
        # an index must appear in only one of train/val/test
        labels = [0, 0, 0, 0, 1, 1, 1, 1, 0, 1] * 10
        train, val, test = stratified_split(labels, seed=0)
        all_indices = train + val + test
        assert len(all_indices) == len(set(all_indices))
        # and every index must be covered
        assert set(all_indices) == set(range(len(labels)))


class TestClassWeights:
    def test_balanced_classes_give_equal_weights(self):
        # 50/50 split -> weights should be equal
        labels = [0] * 50 + [1] * 50
        weights = compute_class_weights(labels, num_classes=2)
        assert torch.allclose(weights, torch.tensor([1.0, 1.0]))

    def test_imbalanced_classes_higher_weight_for_rare(self):
        # 80% class 0, 20% class 1 -> class 1 should weight higher
        labels = [0] * 80 + [1] * 20
        weights = compute_class_weights(labels, num_classes=2)
        assert weights[1].item() > weights[0].item()


class TestHter:
    def test_perfect_predictions(self):
        # preds = targets -> FAR = FRR = HTER = 0
        preds = torch.tensor([0, 0, 1, 1])
        targets = torch.tensor([0, 0, 1, 1])
        rates = hter(preds, targets)
        assert rates["far"] == 0.0
        assert rates["frr"] == 0.0
        assert rates["hter"] == 0.0

    def test_all_real_misclassified(self):
        # every real flagged as attack -> FRR = 1.0, FAR = 0, HTER = 0.5
        preds = torch.tensor([1, 1, 1, 1])
        targets = torch.tensor([0, 0, 1, 1])
        rates = hter(preds, targets)
        assert rates["frr"] == 1.0
        assert rates["far"] == 0.0
        assert rates["hter"] == 0.5

    def test_all_attack_misclassified(self):
        # every attack flagged as real -> FAR = 1.0, FRR = 0, HTER = 0.5
        preds = torch.tensor([0, 0, 0, 0])
        targets = torch.tensor([0, 0, 1, 1])
        rates = hter(preds, targets)
        assert rates["far"] == 1.0
        assert rates["frr"] == 0.0
        assert rates["hter"] == 0.5

    def test_half_each_wrong(self):
        # 1 real wrong (FRR=0.5), 1 attack wrong (FAR=0.5) -> HTER = 0.5
        preds = torch.tensor([0, 1, 0, 1])
        targets = torch.tensor([0, 0, 1, 1])
        rates = hter(preds, targets)
        assert rates["frr"] == 0.5
        assert rates["far"] == 0.5
        assert rates["hter"] == 0.5
