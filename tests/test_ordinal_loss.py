"""
unit tests for ordinal regression loss + encoder + decoder.

these run on CPU — no GPU needed, no dataset needed. fast feedback loop
before the full training script ever sees the HPC.
"""

from __future__ import annotations

import torch
import pytest

from src.models.ordinal_loss import (
    NUM_THRESHOLDS,
    OrdinalRegressionLoss,
    age_to_ordinal_target,
    ordinal_logits_to_age,
)


class TestAgeToOrdinalTarget:
    """checks the encoder turns integer ages into the right binary vectors."""

    def test_target_shape(self):
        # batch of 4 ages -> matrix of (4, 100) targets
        ages = torch.tensor([0, 25, 50, 100])
        target = age_to_ordinal_target(ages)
        assert target.shape == (4, NUM_THRESHOLDS)

    def test_target_dtype(self):
        # targets must be float — BCEWithLogitsLoss expects float targets
        ages = torch.tensor([10])
        target = age_to_ordinal_target(ages)
        assert target.dtype == torch.float32

    def test_age_zero_is_all_zeros(self):
        # age 0 means "is age > k?" answers no for every k -> all zeros
        ages = torch.tensor([0])
        target = age_to_ordinal_target(ages)
        assert target.sum().item() == 0.0

    def test_age_25_has_25_ones(self):
        # age 25 -> positions 0..24 are 1 (age > 0, > 1, ..., > 24 all true)
        # positions 25..99 are 0 (age > 25, > 26, ... all false)
        ages = torch.tensor([25])
        target = age_to_ordinal_target(ages)
        assert target.sum().item() == 25.0
        # check the boundary explicitly
        assert target[0, 24].item() == 1.0
        assert target[0, 25].item() == 0.0

    def test_age_100_is_all_ones(self):
        # age 100 -> every "is age > k?" for k in 0..99 is true -> all ones
        ages = torch.tensor([100])
        target = age_to_ordinal_target(ages)
        assert target.sum().item() == float(NUM_THRESHOLDS)

    def test_clamp_above_max(self):
        # an age above num_thresholds gets clamped — should not crash
        # and should produce all ones (every threshold is exceeded)
        ages = torch.tensor([200])
        target = age_to_ordinal_target(ages)
        assert target.sum().item() == float(NUM_THRESHOLDS)

    def test_clamp_below_min(self):
        # a negative age gets clamped to 0 — should produce all zeros
        ages = torch.tensor([-5])
        target = age_to_ordinal_target(ages)
        assert target.sum().item() == 0.0


class TestOrdinalLogitsToAge:
    """checks the decoder reverses the encoder for clean inputs."""

    def test_decode_shape(self):
        # batch of 3 -> (3,) decoded ages
        logits = torch.zeros(3, NUM_THRESHOLDS)
        ages = ordinal_logits_to_age(logits)
        assert ages.shape == (3,)

    def test_decode_all_negative_logits_gives_zero(self):
        # very negative logits -> sigmoid ~ 0 -> no threshold passes -> age 0
        logits = torch.full((1, NUM_THRESHOLDS), -10.0)
        age = ordinal_logits_to_age(logits)
        assert age.item() == 0.0

    def test_decode_all_positive_logits_gives_max(self):
        # very positive logits -> sigmoid ~ 1 -> every threshold passes
        # -> age = NUM_THRESHOLDS = 100
        logits = torch.full((1, NUM_THRESHOLDS), 10.0)
        age = ordinal_logits_to_age(logits)
        assert age.item() == float(NUM_THRESHOLDS)

    def test_decode_round_trip_age_25(self):
        # encode age 25 -> binary target -> turn into "fake logits" by
        # mapping 1 -> +10, 0 -> -10 -> decode should give back 25
        ages = torch.tensor([25])
        target = age_to_ordinal_target(ages)
        fake_logits = (target * 20.0) - 10.0
        decoded = ordinal_logits_to_age(fake_logits)
        assert decoded.item() == 25.0

    def test_decode_round_trip_multiple_ages(self):
        # same idea but for a batch — confirms the decode is per-sample correct
        ages = torch.tensor([0, 5, 50, 99])
        target = age_to_ordinal_target(ages)
        fake_logits = (target * 20.0) - 10.0
        decoded = ordinal_logits_to_age(fake_logits)
        assert torch.equal(decoded, ages.float())


class TestOrdinalRegressionLoss:
    """checks the loss module behaves correctly end to end."""

    def test_loss_is_scalar(self):
        # loss should reduce to a single scalar tensor (mean reduction)
        loss_fn = OrdinalRegressionLoss()
        logits = torch.randn(8, NUM_THRESHOLDS)
        ages = torch.randint(0, 101, (8,))
        loss = loss_fn(logits, ages)
        assert loss.dim() == 0

    def test_loss_is_positive(self):
        # cross-entropy loss is always >= 0
        loss_fn = OrdinalRegressionLoss()
        logits = torch.randn(8, NUM_THRESHOLDS)
        ages = torch.randint(0, 101, (8,))
        loss = loss_fn(logits, ages)
        assert loss.item() >= 0.0

    def test_loss_near_zero_for_perfect_logits(self):
        # if logits exactly match the target encoding (with high magnitude),
        # the BCE loss should be close to zero
        loss_fn = OrdinalRegressionLoss()
        ages = torch.tensor([10, 30, 70])
        target = age_to_ordinal_target(ages)
        # +10 / -10 logits -> sigmoid ~ 1 / ~ 0 -> tiny BCE
        perfect_logits = (target * 20.0) - 10.0
        loss = loss_fn(perfect_logits, ages)
        assert loss.item() < 0.001

    def test_loss_higher_for_wrong_logits(self):
        # flipped logits -> sigmoid says 0 where target is 1 (and vice versa)
        # -> loss should be far higher than the perfect case
        loss_fn = OrdinalRegressionLoss()
        ages = torch.tensor([10, 30, 70])
        target = age_to_ordinal_target(ages)
        wrong_logits = ((1 - target) * 20.0) - 10.0
        loss = loss_fn(wrong_logits, ages)
        assert loss.item() > 5.0

    def test_loss_backward_pass(self):
        # gradients must flow — without this, training would silently no-op
        loss_fn = OrdinalRegressionLoss()
        logits = torch.randn(4, NUM_THRESHOLDS, requires_grad=True)
        ages = torch.tensor([20, 30, 40, 50])
        loss = loss_fn(logits, ages)
        loss.backward()
        assert logits.grad is not None
        assert logits.grad.shape == logits.shape
