"""
unit tests for Phase 9.

two test surfaces:
    1. /explain endpoint via FastAPI TestClient + stub engine
    2. streamlit_app helpers — overlay_saliency and the API client
       functions, with `requests` mocked so no real network call

streamlit's UI components are not tested directly — the helpers are
where the logic lives. UI is tested by humans clicking through the demo.
"""

from __future__ import annotations

import io
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from PIL import Image

# whole module skips if FastAPI is missing (engine tests need it)
pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from src.api.inference import PredictionResult, SaliencyResult


# ---------- shared helpers ----------


def _make_jpeg_bytes(size=(100, 100), color=(128, 64, 32)) -> bytes:
    """small in-memory JPEG for upload tests."""
    buf = io.BytesIO()
    Image.new("RGB", size, color=color).save(buf, "JPEG")
    return buf.getvalue()


class StubEngine:
    """
    same shape as src/api/test_api.py's StubEngine, with explain() added.

    real engine needs ONNX files on disk. stub bypasses both predict()
    and explain() so endpoint tests never touch a model.
    """

    def __init__(self, *, age_path: Path, pad_path: Path):
        self.age_model_path = age_path
        self.pad_model_path = pad_path
        self.spoof_threshold = 0.5
        self.next_predict: PredictionResult | None = None
        self.next_explain: SaliencyResult | None = None
        self.calls: list[tuple[str, bytes]] = []

    def predict(self, image_bytes: bytes) -> PredictionResult:
        self.calls.append(("predict", image_bytes))
        if self.next_predict is None:
            raise RuntimeError("StubEngine.next_predict not set in this test")
        return self.next_predict

    def explain(self, image_bytes: bytes, grid: int = 12,
                patch_px: int = 28) -> SaliencyResult:
        self.calls.append(("explain", image_bytes))
        if self.next_explain is None:
            raise RuntimeError("StubEngine.next_explain not set in this test")
        return self.next_explain


def _make_app_with_stub(stub: StubEngine):
    """attach stub engine to the real app's state, return the app."""
    from src.api import main as api_main
    api_main.app.state.engine = stub
    return api_main.app


# ---------- /explain endpoint ----------


class TestExplainEndpoint:
    def test_returns_saliency_map(self, tmp_path: Path):
        # build a stub that returns a 224x224 saliency map
        stub = StubEngine(age_path=tmp_path / "age.onnx",
                          pad_path=tmp_path / "pad.onnx")
        sal_array = np.random.rand(224, 224).astype(np.float32)
        stub.next_explain = SaliencyResult(
            saliency_map=sal_array,
            baseline_age=32.0,
            inference_ms=1234.0,
        )
        app = _make_app_with_stub(stub)

        with TestClient(app) as client:
            app.state.engine = stub
            res = client.post(
                "/explain",
                files={"file": ("face.jpg", _make_jpeg_bytes(), "image/jpeg")},
            )
        assert res.status_code == 200
        body = res.json()
        # saliency_map serialised as nested list, shape preserved
        assert len(body["saliency_map"]) == 224
        assert len(body["saliency_map"][0]) == 224
        assert body["baseline_age"] == 32.0
        assert body["grid"] == 12  # default forwarded
        assert body["image_size"] == 224

    def test_grid_param_forwarded(self, tmp_path: Path):
        # the grid query param should reach engine.explain()
        stub = StubEngine(age_path=tmp_path / "age.onnx",
                          pad_path=tmp_path / "pad.onnx")
        stub.next_explain = SaliencyResult(
            saliency_map=np.zeros((224, 224), dtype=np.float32),
            baseline_age=25.0,
            inference_ms=500.0,
        )
        app = _make_app_with_stub(stub)

        with TestClient(app) as client:
            app.state.engine = stub
            res = client.post(
                "/explain",
                params={"grid": 8},
                files={"file": ("face.jpg", _make_jpeg_bytes(), "image/jpeg")},
            )
        assert res.status_code == 200
        # the response should echo the grid we sent, not the default
        assert res.json()["grid"] == 8

    def test_rejects_invalid_grid(self, tmp_path: Path):
        stub = StubEngine(age_path=tmp_path / "age.onnx",
                          pad_path=tmp_path / "pad.onnx")
        app = _make_app_with_stub(stub)

        with TestClient(app) as client:
            app.state.engine = stub
            # grid=1 below the min — rejected
            res = client.post(
                "/explain",
                params={"grid": 1},
                files={"file": ("f.jpg", _make_jpeg_bytes(), "image/jpeg")},
            )
        assert res.status_code == 400
        assert "grid" in res.json()["detail"].lower()

    def test_rejects_invalid_patch_px(self, tmp_path: Path):
        stub = StubEngine(age_path=tmp_path / "age.onnx",
                          pad_path=tmp_path / "pad.onnx")
        app = _make_app_with_stub(stub)

        with TestClient(app) as client:
            app.state.engine = stub
            res = client.post(
                "/explain",
                params={"patch_px": 200},  # above max 128
                files={"file": ("f.jpg", _make_jpeg_bytes(), "image/jpeg")},
            )
        assert res.status_code == 400
        assert "patch_px" in res.json()["detail"].lower()

    def test_rejects_non_image(self, tmp_path: Path):
        stub = StubEngine(age_path=tmp_path / "age.onnx",
                          pad_path=tmp_path / "pad.onnx")
        app = _make_app_with_stub(stub)

        with TestClient(app) as client:
            app.state.engine = stub
            res = client.post(
                "/explain",
                files={"file": ("d.txt", b"hello", "text/plain")},
            )
        assert res.status_code == 400

    def test_returns_503_when_engine_missing(self):
        from src.api import main as api_main
        api_main.app.state.engine = None
        with TestClient(api_main.app) as client:
            api_main.app.state.engine = None
            res = client.post(
                "/explain",
                files={"file": ("f.jpg", _make_jpeg_bytes(), "image/jpeg")},
            )
        assert res.status_code == 503


# ---------- engine.explain() saliency math ----------


class TestSaliencyMath:
    """check the occlusion math directly without the API layer."""

    def test_saliency_map_normalised_to_unit_range(self, tmp_path: Path):
        # build an actual ONNX model + run engine.explain through it
        # to confirm the output stays inside [0, 1]
        ort = pytest.importorskip("onnxruntime")
        torch = pytest.importorskip("torch")

        # tiny stub model producing 100 logits — same shape contract as
        # AgeEstimator. exporting both as fp32 ONNX so the engine has
        # something real to load.
        import torch.nn as nn

        class TinyAge(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(3, 8, 3, padding=1)
                self.pool = nn.AdaptiveAvgPool2d(1)
                self.fc = nn.Linear(8, 100)

            def forward(self, x):
                x = torch.relu(self.conv(x))
                x = self.pool(x).flatten(1)
                return self.fc(x)

        class TinyPad(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(3, 4, 3, padding=1)
                self.pool = nn.AdaptiveAvgPool2d(1)
                self.fc = nn.Linear(4, 2)

            def forward(self, x):
                x = torch.relu(self.conv(x))
                x = self.pool(x).flatten(1)
                return self.fc(x)

        def _export(m, path):
            kwargs = dict(
                input_names=["input"],
                output_names=["logits"],
                dynamic_axes={"input": {0: "b"}, "logits": {0: "b"}},
                opset_version=17,
                do_constant_folding=True,
            )
            import inspect
            if "dynamo" in inspect.signature(torch.onnx.export).parameters:
                kwargs["dynamo"] = False
            torch.onnx.export(m.eval(), torch.randn(1, 3, 224, 224),
                              str(path), **kwargs)

        age_path = tmp_path / "age.onnx"
        pad_path = tmp_path / "pad.onnx"
        _export(TinyAge(), age_path)
        _export(TinyPad(), pad_path)

        from src.api.inference import InferenceEngine
        engine = InferenceEngine(age_model_path=age_path,
                                 pad_model_path=pad_path)

        result = engine.explain(_make_jpeg_bytes(), grid=4, patch_px=20)
        assert result.saliency_map.shape == (224, 224)
        assert result.saliency_map.min() >= 0.0
        assert result.saliency_map.max() <= 1.0
        assert result.inference_ms > 0


# ---------- streamlit helpers ----------


class TestSaliencyOverlay:
    def test_overlay_returns_pil_image(self):
        from streamlit_app.app import overlay_saliency
        img = Image.new("RGB", (50, 50), color=(100, 100, 100))
        sal = [[0.5] * 32 for _ in range(32)]
        out = overlay_saliency(img, sal)
        assert isinstance(out, Image.Image)
        # output matches saliency resolution (heatmap pixel-aligned)
        assert out.size == (32, 32)

    def test_overlay_red_increases_with_saliency(self):
        from streamlit_app.app import overlay_saliency
        img = Image.new("RGB", (32, 32), color=(50, 50, 50))

        # zero saliency -> red channel near image red value
        zero_sal = [[0.0] * 32 for _ in range(32)]
        zero_arr = np.asarray(overlay_saliency(img, zero_sal))

        # max saliency -> red channel pulled up by the heatmap
        full_sal = [[1.0] * 32 for _ in range(32)]
        full_arr = np.asarray(overlay_saliency(img, full_sal))

        assert full_arr[:, :, 0].mean() > zero_arr[:, :, 0].mean()


class TestAPIClient:
    """mock requests so no real HTTP hits the network during tests."""

    def test_estimate_age_posts_image(self):
        from streamlit_app import app as st_app
        with patch.object(st_app.requests, "post") as mock_post:
            mock_post.return_value.status_code = 200
            mock_post.return_value.json.return_value = {"is_spoof": False}
            mock_post.return_value.raise_for_status = lambda: None
            out = st_app.call_estimate_age(b"jpegbytes", "x.jpg")

        assert mock_post.called
        url = mock_post.call_args[0][0]
        assert "/estimate-age" in url
        assert out["is_spoof"] is False

    def test_explain_forwards_grid(self):
        from streamlit_app import app as st_app
        with patch.object(st_app.requests, "post") as mock_post:
            mock_post.return_value.status_code = 200
            mock_post.return_value.json.return_value = {"saliency_map": []}
            mock_post.return_value.raise_for_status = lambda: None
            st_app.call_explain(b"jpegbytes", grid=8)

        # grid should be in params
        params = mock_post.call_args.kwargs.get("params", {})
        assert params.get("grid") == 8

    def test_health_returns_none_on_connection_error(self):
        from streamlit_app import app as st_app
        import requests as _r
        with patch.object(st_app.requests, "get",
                          side_effect=_r.ConnectionError):
            assert st_app.call_health() is None

    def test_bias_report_returns_none_on_404(self):
        from streamlit_app import app as st_app
        with patch.object(st_app.requests, "get") as mock_get:
            mock_get.return_value.status_code = 404
            assert st_app.call_bias_report() is None
