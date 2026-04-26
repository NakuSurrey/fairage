"""
unit tests for src/data/utkface_dataset.py

builds a temporary folder of synthetic JPEGs with realistic UTKFace filenames.
no real dataset needed — these tests run on any machine, including CI.
"""

from pathlib import Path

import pytest
import torch
from PIL import Image

from src.data.transforms import get_inference_transform, get_train_transform
from src.data.utkface_dataset import UTKFaceDataset, _parse_filename


# ---------- helpers ----------

def _make_fake_image(path: Path, size: tuple[int, int] = (200, 200)) -> None:
    """write a small RGB JPEG to disk. content does not matter — only shape."""
    Image.new("RGB", size, color=(128, 64, 32)).save(path, "JPEG")


@pytest.fixture
def fake_utkface_dir(tmp_path: Path) -> Path:
    """
    build a folder that mimics UTKFace layout with 6 valid samples and 2 noisy ones.
    valid filenames cover both genders and three ethnicities for downstream tests.
    """
    folder = tmp_path / "utkface"
    folder.mkdir()

    valid_names = [
        "25_0_2_20170104203021098.jpg.chip.jpg",  # male, asian, age 25
        "30_1_0_20170104203021099.jpg.chip.jpg",  # female, white, 30
        "5_0_1_20170104203021100.jpg.chip.jpg",   # male, black, 5
        "70_1_3_20170104203021101.jpg.chip.jpg",  # female, indian, 70
        "45_0_4_20170104203021102.jpg.chip.jpg",  # male, others, 45
        "18_1_2_20170104203021103.jpg.chip.jpg",  # female, asian, 18
    ]
    for name in valid_names:
        _make_fake_image(folder / name)

    # noisy filenames — should be skipped by the parser
    _make_fake_image(folder / "abc_0_2_20170104.jpg.chip.jpg")  # age not int
    _make_fake_image(folder / "25_5_2_20170104.jpg.chip.jpg")   # gender out of range

    return folder


# ---------- filename parser ----------

class TestParseFilename:
    def test_valid_filename(self):
        sample = _parse_filename(Path("25_0_2_20170104.jpg.chip.jpg"))
        assert sample is not None
        assert sample.age == 25
        assert sample.gender == 0
        assert sample.ethnicity == 2

    def test_non_int_age_returns_none(self):
        assert _parse_filename(Path("abc_0_2_20170104.jpg.chip.jpg")) is None

    def test_gender_out_of_range_returns_none(self):
        assert _parse_filename(Path("25_5_2_20170104.jpg.chip.jpg")) is None

    def test_ethnicity_out_of_range_returns_none(self):
        assert _parse_filename(Path("25_0_9_20170104.jpg.chip.jpg")) is None

    def test_age_above_max_returns_none(self):
        # AGE_MAX is 100 — 200 should be rejected as outlier
        assert _parse_filename(Path("200_0_2_20170104.jpg.chip.jpg")) is None

    def test_too_few_fields_returns_none(self):
        assert _parse_filename(Path("25.jpg.chip.jpg")) is None


# ---------- dataset ----------

class TestUTKFaceDataset:
    def test_loads_only_valid_files(self, fake_utkface_dir: Path):
        ds = UTKFaceDataset(root=fake_utkface_dir)
        # 8 files on disk, 2 are noisy — expect 6 valid samples
        assert len(ds) == 6
        assert ds.skipped_count == 2

    def test_missing_folder_raises(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            UTKFaceDataset(root=tmp_path / "does_not_exist")

    def test_getitem_returns_dict_without_transform(self, fake_utkface_dir: Path):
        ds = UTKFaceDataset(root=fake_utkface_dir)
        item = ds[0]

        assert "image" in item
        assert "age" in item
        assert "gender" in item
        assert "ethnicity" in item
        assert "path" in item

    def test_getitem_with_inference_transform(self, fake_utkface_dir: Path):
        ds = UTKFaceDataset(root=fake_utkface_dir, transform=get_inference_transform())
        item = ds[0]

        # tensor shape must match config IMAGE_SIZE = 224
        assert isinstance(item["image"], torch.Tensor)
        assert item["image"].shape == (3, 224, 224)
        # normalised — values can go negative
        assert item["image"].dtype == torch.float32

    def test_getitem_with_train_transform(self, fake_utkface_dir: Path):
        ds = UTKFaceDataset(root=fake_utkface_dir, transform=get_train_transform())
        item = ds[0]
        assert item["image"].shape == (3, 224, 224)

    def test_label_dtypes_are_long(self, fake_utkface_dir: Path):
        ds = UTKFaceDataset(root=fake_utkface_dir)
        item = ds[0]
        # cross-entropy and embedding lookups need long dtype
        assert item["age"].dtype == torch.long
        assert item["gender"].dtype == torch.long
        assert item["ethnicity"].dtype == torch.long

    def test_get_label_arrays_lengths_match(self, fake_utkface_dir: Path):
        ds = UTKFaceDataset(root=fake_utkface_dir)
        labels = ds.get_label_arrays()
        assert len(labels["age"]) == len(ds)
        assert len(labels["gender"]) == len(ds)
        assert len(labels["ethnicity"]) == len(ds)
