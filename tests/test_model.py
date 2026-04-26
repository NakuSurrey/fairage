"""
unit tests for src/models/age_estimator.py
no real data needed — runs on dummy tensors, fast.
"""

import torch

from src.models.age_estimator import AgeEstimator, count_parameters


class TestAgeEstimator:
    def test_forward_pass_shape(self):
        # build with pretrained=False to skip the imagenet weights download in CI
        model = AgeEstimator(pretrained=False)
        model.eval()

        # batch of 4 RGB 224x224 images — matches IMAGE_SIZE in config
        x = torch.randn(4, 3, 224, 224)
        with torch.no_grad():
            out = model(x)

        # one float per image — last dim squeezed away by forward()
        assert out.shape == (4,)
        assert out.dtype == torch.float32

    def test_single_image_works(self):
        # batch size 1 — common path in the API
        model = AgeEstimator(pretrained=False)
        model.eval()
        x = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (1,)

    def test_param_count_is_reasonable(self):
        # ResNet-50 = ~25M params, head adds a tiny amount
        model = AgeEstimator(pretrained=False)
        counts = count_parameters(model)
        assert 20_000_000 < counts["total"] < 30_000_000
        # all params trainable by default
        assert counts["trainable"] == counts["total"]

    def test_train_mode_uses_dropout(self):
        # in train mode, two forward passes on the same input give different outputs
        # because dropout drops different units each time
        model = AgeEstimator(pretrained=False, dropout=0.5)
        model.train()
        x = torch.randn(2, 3, 224, 224)
        torch.manual_seed(1)
        out1 = model(x)
        torch.manual_seed(2)
        out2 = model(x)
        # not strictly guaranteed, but with dropout 0.5 the chance of identical
        # outputs across two seeds is effectively zero
        assert not torch.allclose(out1, out2)
