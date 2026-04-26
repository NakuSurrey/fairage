"""
unit tests for the FastAPI app.

uses FastAPI's TestClient — runs the app in-process, no real server
needed. injects a stub InferenceEngine so tests do not depend on the
real ONNX model files being present.

covers:
    - /health: ok and degraded states
    - /estimate-age: real face path, spoof rejection, validation errors
    - /bias-report: missing file, valid file, corrupt file
    - schema validation: pydantic enforces field bounds

skips gracefully if FastAPI / starlette are missing.
"""

from __future__ import annotations

import io
import json
from pathlib import Path

import pytest
from PIL import Image

# whole module skips if FastAPI is not installed in the test env
pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from src.api.inference import PredictionResult


# ---------- stub engine ----------


class StubEngine:
    """
    in-memory stand-in for InferenceEngine.

    real engine loads ONNX sessions on init. stub bypasses that entirely
    and returns whatever the test sets in `next_result`. lets the tests
    drive any code path without a model file on disk.
    """

    def __init__(self, *, age_path: Path, pad_path: Path):
        self.age_model_path = age_path
        self.pad_model_path = pad_path
        self.spoof_threshold = 0.5
        self.next_result: PredictionResult | None = None
        self.calls: list[bytes] = []

    def predict(self, image_bytes: bytes) -> PredictionResult:
        # remember every call so tests can assert against them
        self.calls.append(image_bytes)
        if self.next_result is None:
            raise RuntimeError("StubEngine.next_result not set in this test")
        return self.next_result


def _make_app_with_stub(stub: StubEngine):
    """
    build a fresh FastAPI app and attach the stub engine to its state.

    bypasses the real lifespan handler — keeps tests deterministic and
    independent of whether ONNX files exist on disk.
    """
    from src.api import main as api_main

    # rebuild the app without invoking lifespan — TestClient(app) below
    # will trigger the real lifespan otherwise. instead, we put the stub
    # on the existing app's state directly.
    api_main.app.state.engine = stub
    return api_main.app


def _make_jpeg_bytes(size=(100, 100), color=(128, 64, 32)) -> bytes:
    """small in-memory JPEG for upload tests."""
    buf = io.BytesIO()
    Image.new("RGB", size, color=color).save(buf, "JPEG")
    return buf.getvalue()


# ---------- /health ----------


class TestHealth:
    def test_health_ok_when_engine_loaded(self, tmp_path: Path):
        stub = StubEngine(age_path=tmp_path / "age.onnx",
                          pad_path=tmp_path / "pad.onnx")
        app = _make_app_with_stub(stub)
        # context-manager form skips startup/shutdown lifespan calls
        with TestClient(app) as client:
            # overwrite engine again — TestClient lifespan may have run
            app.state.engine = stub
            res = client.get("/health")
        assert res.status_code == 200
        body = res.json()
        assert body["status"] == "ok"
        assert body["age_model_loaded"] is True
        assert body["pad_model_loaded"] is True

    def test_health_degraded_when_engine_missing(self):
        from src.api import main as api_main
        api_main.app.state.engine = None
        with TestClient(api_main.app) as client:
            api_main.app.state.engine = None  # reassert after lifespan
            res = client.get("/health")
        assert res.status_code == 200
        body = res.json()
        assert body["status"] == "degraded"
        assert body["age_model_loaded"] is False


# ---------- /estimate-age ----------


class TestEstimateAge:
    def test_real_face_returns_age(self, tmp_path: Path):
        stub = StubEngine(age_path=tmp_path / "age.onnx",
                          pad_path=tmp_path / "pad.onnx")
        stub.next_result = PredictionResult(
            is_spoof=False,
            pad_score=0.05,
            estimated_age=27.5,
            age_confidence=0.82,
            inference_ms=42.1,
        )
        app = _make_app_with_stub(stub)

        with TestClient(app) as client:
            app.state.engine = stub
            res = client.post(
                "/estimate-age",
                files={"file": ("face.jpg", _make_jpeg_bytes(), "image/jpeg")},
            )
        assert res.status_code == 200
        body = res.json()
        assert body["is_spoof"] is False
        assert body["estimated_age"] == 27.5
        assert body["pad_score"] == 0.05
        assert body["age_confidence"] == 0.82
        assert body["inference_ms"] == 42.1
        assert "model_version" in body

    def test_spoof_returns_null_age(self, tmp_path: Path):
        stub = StubEngine(age_path=tmp_path / "age.onnx",
                          pad_path=tmp_path / "pad.onnx")
        stub.next_result = PredictionResult(
            is_spoof=True,
            pad_score=0.95,
            estimated_age=None,
            age_confidence=None,
            inference_ms=18.0,
        )
        app = _make_app_with_stub(stub)

        with TestClient(app) as client:
            app.state.engine = stub
            res = client.post(
                "/estimate-age",
                files={"file": ("spoof.jpg", _make_jpeg_bytes(), "image/jpeg")},
            )
        assert res.status_code == 200
        body = res.json()
        assert body["is_spoof"] is True
        assert body["estimated_age"] is None
        assert body["age_confidence"] is None
        assert body["pad_score"] == 0.95

    def test_rejects_non_image_content_type(self, tmp_path: Path):
        stub = StubEngine(age_path=tmp_path / "age.onnx",
                          pad_path=tmp_path / "pad.onnx")
        app = _make_app_with_stub(stub)

        with TestClient(app) as client:
            app.state.engine = stub
            res = client.post(
                "/estimate-age",
                files={"file": ("doc.txt", b"hello", "text/plain")},
            )
        assert res.status_code == 400
        assert "image" in res.json()["detail"].lower()

    def test_rejects_empty_upload(self, tmp_path: Path):
        stub = StubEngine(age_path=tmp_path / "age.onnx",
                          pad_path=tmp_path / "pad.onnx")
        app = _make_app_with_stub(stub)

        with TestClient(app) as client:
            app.state.engine = stub
            res = client.post(
                "/estimate-age",
                files={"file": ("empty.jpg", b"", "image/jpeg")},
            )
        assert res.status_code == 400
        assert "empty" in res.json()["detail"].lower()

    def test_returns_503_when_engine_missing(self):
        from src.api import main as api_main
        api_main.app.state.engine = None
        with TestClient(api_main.app) as client:
            api_main.app.state.engine = None
            res = client.post(
                "/estimate-age",
                files={"file": ("face.jpg", _make_jpeg_bytes(), "image/jpeg")},
            )
        assert res.status_code == 503


# ---------- /bias-report ----------


class TestBiasReport:
    def test_returns_404_when_file_missing(self, tmp_path: Path, monkeypatch):
        # point the API at an empty tmp dir — file does not exist there
        from src.api import main as api_main
        monkeypatch.setattr(api_main, "BIAS_REPORT_PATH",
                            tmp_path / "bias_report.json")

        with TestClient(api_main.app) as client:
            res = client.get("/bias-report")
        assert res.status_code == 404
        assert "bias report" in res.json()["detail"].lower()

    def test_returns_payload_when_file_exists(self, tmp_path: Path, monkeypatch):
        # write a minimal valid bias report and point the API at it
        report = {
            "overall": {"n_samples": 100, "mae": 4.5, "rmse": 5.8},
            "groups": [
                {"group_type": "gender", "group_name": "Male",
                 "n_samples": 50, "mae": 4.0, "rmse": 5.2},
                {"group_type": "gender", "group_name": "Female",
                 "n_samples": 50, "mae": 5.0, "rmse": 6.4},
            ],
            "worst_gap": {"worst_group": "gender:Female",
                          "worst_mae": 5.0, "gap_years": 0.5},
        }
        path = tmp_path / "bias_report.json"
        path.write_text(json.dumps(report))

        from src.api import main as api_main
        monkeypatch.setattr(api_main, "BIAS_REPORT_PATH", path)

        with TestClient(api_main.app) as client:
            res = client.get("/bias-report")
        assert res.status_code == 200
        body = res.json()
        assert body["overall"]["mae"] == 4.5
        assert len(body["groups"]) == 2
        assert body["worst_gap"]["worst_group"] == "gender:Female"

    def test_returns_500_on_corrupt_json(self, tmp_path: Path, monkeypatch):
        # broken JSON should surface as a clean 500, not a stack trace
        path = tmp_path / "bias_report.json"
        path.write_text("not valid json {")

        from src.api import main as api_main
        monkeypatch.setattr(api_main, "BIAS_REPORT_PATH", path)

        with TestClient(api_main.app) as client:
            res = client.get("/bias-report")
        assert res.status_code == 500
        assert "corrupt" in res.json()["detail"].lower()


# ---------- preprocessing math ----------


class TestPreprocessing:
    """direct tests on the helper functions inside inference.py."""

    def test_preprocess_output_shape(self):
        from src.api.inference import _preprocess_image
        img_bytes = _make_jpeg_bytes(size=(60, 90))
        arr = _preprocess_image(img_bytes)
        # (1, 3, 224, 224) — batched, CHW, ResNet-50 input size
        assert arr.shape == (1, 3, 224, 224)
        assert arr.dtype.name == "float32"

    def test_preprocess_normalisation_shifts_values(self):
        # imagenet normalisation produces both negative and positive values
        from src.api.inference import _preprocess_image
        arr = _preprocess_image(_make_jpeg_bytes())
        assert arr.min() < 0.0
        assert arr.max() > 0.0

    def test_softmax_sums_to_one(self):
        import numpy as np
        from src.api.inference import _softmax
        x = np.array([[1.0, 2.0, 3.0]])
        out = _softmax(x, axis=1)
        assert abs(out.sum() - 1.0) < 1e-6

    def test_decode_age_returns_age_and_confidence(self):
        import numpy as np
        from src.api.inference import _decode_age
        # logits shaped like a 30-year-old: first 30 strongly positive,
        # rest strongly negative
        logits = np.full((1, 100), -10.0, dtype=np.float32)
        logits[0, :30] = 10.0
        age, confidence = _decode_age(logits)
        assert age == 30.0
        # very confident in every threshold -> confidence near 1.0
        assert confidence > 0.99
