"""
unit tests for src/audit/bias_audit.py

builds synthetic prediction arrays with known errors per group, verifies the
audit produces the expected per-group MAE values. no model needed, no dataset
needed — runs on any machine in under a second.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from src.audit.bias_audit import (
    GroupResult,
    _bucket_index,
    _mae_rmse,
    compute_per_group_metrics,
    overall_metrics,
    run_bias_audit,
    worst_group_gap,
)
from src.config import AGE_BUCKETS, ETHNICITY_LABELS, GENDER_LABELS


class TestBucketIndex:
    """checks the age -> bucket mapping the audit relies on."""

    def test_age_in_first_bucket(self):
        # bucket 0 is (0, 12) — age 5 should land there
        assert _bucket_index(5) == 0

    def test_age_at_bucket_boundary(self):
        # boundary ages must land in the bucket whose hi equals them
        # AGE_BUCKETS[0] is (0, 12) -> 12 belongs to bucket 0, not bucket 1
        assert _bucket_index(12) == 0
        assert _bucket_index(13) == 1

    def test_age_in_last_bucket(self):
        # last bucket is (56, 100) — 99 lands there
        last_idx = len(AGE_BUCKETS) - 1
        assert _bucket_index(99) == last_idx

    def test_age_above_range_returns_minus_one(self):
        # ages above 100 are not in any bucket — sentinel -1
        assert _bucket_index(150) == -1


class TestMaeRmse:
    """direct check of the metric helper."""

    def test_zero_error(self):
        # identical preds and targets -> zero MAE and zero RMSE
        preds = torch.tensor([10.0, 20.0, 30.0])
        targets = torch.tensor([10.0, 20.0, 30.0])
        mae, rmse = _mae_rmse(preds, targets)
        assert mae == 0.0
        assert rmse == 0.0

    def test_constant_error(self):
        # every prediction is 5 years off -> MAE = 5, RMSE = 5
        preds = torch.tensor([15.0, 25.0, 35.0])
        targets = torch.tensor([10.0, 20.0, 30.0])
        mae, rmse = _mae_rmse(preds, targets)
        assert mae == pytest.approx(5.0)
        assert rmse == pytest.approx(5.0)

    def test_empty_returns_nan(self):
        # empty slice -> NaN, not crash. real datasets can have empty groups.
        preds = torch.tensor([])
        targets = torch.tensor([])
        mae, rmse = _mae_rmse(preds, targets)
        # NaN != NaN — explicit isnan check
        import math
        assert math.isnan(mae)
        assert math.isnan(rmse)


class TestComputePerGroupMetrics:
    """end-to-end slicing on a small synthetic batch."""

    def _make_batch(self):
        """
        4 samples:
          - sample 0: male/white/age 25 — pred 25, error 0
          - sample 1: female/black/age 30 — pred 35, error 5
          - sample 2: male/asian/age 50 — pred 50, error 0
          - sample 3: female/white/age 70 — pred 60, error 10
        """
        targets = torch.tensor([25.0, 30.0, 50.0, 70.0])
        preds = torch.tensor([25.0, 35.0, 50.0, 60.0])
        genders = torch.tensor([0, 1, 0, 1])           # male, female, male, female
        ethnicities = torch.tensor([0, 1, 2, 0])       # white, black, asian, white
        return preds, targets, genders, ethnicities

    def test_returns_list_of_group_results(self):
        preds, targets, genders, ethnicities = self._make_batch()
        results = compute_per_group_metrics(preds, targets, genders, ethnicities)
        assert all(isinstance(r, GroupResult) for r in results)
        assert len(results) > 0

    def test_male_group_mae(self):
        # male samples (0 and 2) have errors 0 and 0 -> MAE 0
        preds, targets, genders, ethnicities = self._make_batch()
        results = compute_per_group_metrics(preds, targets, genders, ethnicities)
        male = next(r for r in results if r.group_type == "gender" and r.group_name == "Male")
        assert male.n_samples == 2
        assert male.mae == pytest.approx(0.0)

    def test_female_group_mae(self):
        # female samples (1 and 3) have errors 5 and 10 -> MAE 7.5
        preds, targets, genders, ethnicities = self._make_batch()
        results = compute_per_group_metrics(preds, targets, genders, ethnicities)
        female = next(r for r in results if r.group_type == "gender" and r.group_name == "Female")
        assert female.n_samples == 2
        assert female.mae == pytest.approx(7.5)

    def test_white_ethnicity_mae(self):
        # white samples (0 and 3) have errors 0 and 10 -> MAE 5
        preds, targets, genders, ethnicities = self._make_batch()
        results = compute_per_group_metrics(preds, targets, genders, ethnicities)
        white = next(r for r in results if r.group_type == "ethnicity" and r.group_name == "White")
        assert white.n_samples == 2
        assert white.mae == pytest.approx(5.0)

    def test_age_bucket_assignment(self):
        # age 25 is in (20-35), age 30 also (20-35), age 50 in (36-55), age 70 in (56-100)
        # bucket 20-35 has samples 0,1 with errors 0,5 -> MAE 2.5
        preds, targets, genders, ethnicities = self._make_batch()
        results = compute_per_group_metrics(preds, targets, genders, ethnicities)
        b20_35 = next(
            r for r in results
            if r.group_type == "age_bucket" and r.group_name == "20-35"
        )
        assert b20_35.n_samples == 2
        assert b20_35.mae == pytest.approx(2.5)

    def test_intersectional_groups_present(self):
        # at least one gender_x_ethnicity row should be in the result list
        preds, targets, genders, ethnicities = self._make_batch()
        results = compute_per_group_metrics(preds, targets, genders, ethnicities)
        intersectional = [r for r in results if r.group_type == "gender_x_ethnicity"]
        assert len(intersectional) >= 1

    def test_empty_groups_skipped(self):
        # the batch has no Indian or Others ethnicity — those groups should be absent
        # from results, not zero-filled
        preds, targets, genders, ethnicities = self._make_batch()
        results = compute_per_group_metrics(preds, targets, genders, ethnicities)
        names = {r.group_name for r in results if r.group_type == "ethnicity"}
        assert "Indian" not in names
        assert "Others" not in names


class TestOverallAndWorstGap:
    """top-line metrics and the headline fairness gap."""

    def test_overall_metrics_match_manual_calc(self):
        # errors: 0, 5, 0, 10 -> mean abs = 3.75
        preds = torch.tensor([25.0, 35.0, 50.0, 60.0])
        targets = torch.tensor([25.0, 30.0, 50.0, 70.0])
        out = overall_metrics(preds, targets)
        assert out["n_samples"] == 4
        assert out["mae"] == pytest.approx(3.75)

    def test_worst_group_gap_ignores_small_groups(self):
        # one group has MAE 100 but only 5 samples — should be ignored
        # the threshold is 30 samples
        groups = [
            GroupResult("gender", "Male", n_samples=200, mae=4.0, rmse=5.0),
            GroupResult("gender", "Female", n_samples=200, mae=6.0, rmse=7.0),
            GroupResult("ethnicity", "Tiny", n_samples=5, mae=100.0, rmse=110.0),
        ]
        gap = worst_group_gap(overall_mae=5.0, group_results=groups)
        assert gap["worst_group"] == "gender:Female"
        assert gap["worst_mae"] == pytest.approx(6.0)
        assert gap["gap_years"] == pytest.approx(1.0)

    def test_worst_group_gap_returns_none_when_no_eligible(self):
        # all groups too small -> headline returns None values, no crash
        groups = [
            GroupResult("gender", "Male", n_samples=5, mae=4.0, rmse=5.0),
        ]
        gap = worst_group_gap(overall_mae=5.0, group_results=groups)
        assert gap["worst_group"] is None
        assert gap["worst_mae"] is None
        assert gap["gap_years"] is None


class TestRunBiasAuditEndToEnd:
    """integration test — runs the full audit and checks the files on disk."""

    def test_writes_json_and_markdown(self, tmp_path: Path):
        # bigger synthetic batch — 200 samples, 100 male / 100 female
        torch.manual_seed(0)
        n = 200
        targets = torch.randint(0, 100, (n,)).float()
        # add a 5-year bias only on female samples to make the audit findable
        genders = torch.cat([torch.zeros(100), torch.ones(100)]).long()
        ethnicities = torch.randint(0, 5, (n,))
        preds = targets.clone()
        preds[genders == 1] += 5.0  # female samples are systematically over-predicted

        output = run_bias_audit(preds, targets, genders, ethnicities, tmp_path)

        # file artifacts exist
        assert (tmp_path / "bias_report.json").exists()
        assert (tmp_path / "bias_report.md").exists()

        # JSON parses and has the expected sections
        with (tmp_path / "bias_report.json").open() as f:
            data = json.load(f)
        assert "overall" in data
        assert "groups" in data
        assert "worst_gap" in data

        # the planted female bias should show up in worst-group gap
        assert data["worst_gap"]["worst_group"] is not None
        # female MAE = 5, male MAE = 0, overall MAE = 2.5
        # gap should be ~2.5 years
        assert data["worst_gap"]["gap_years"] == pytest.approx(2.5, abs=0.1)

    def test_returns_payload_in_memory(self, tmp_path: Path):
        # the function returns the same payload it writes — useful in notebooks
        targets = torch.tensor([25.0, 35.0, 50.0])
        preds = torch.tensor([25.0, 35.0, 50.0])
        genders = torch.tensor([0, 1, 0])
        ethnicities = torch.tensor([0, 1, 2])
        result = run_bias_audit(preds, targets, genders, ethnicities, tmp_path)
        assert "overall" in result
        assert "groups" in result
        assert "worst_gap" in result
