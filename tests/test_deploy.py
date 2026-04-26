"""
unit tests for the deploy pipeline.

uses a tiny stub model — not the real AgeEstimator with ResNet-50 — so
tests stay fast on CPU and do not download ImageNet weights. the export
+ quantize + benchmark code paths are agnostic to model architecture, so
the stub exercises the same pipeline.

skips gracefully if onnxruntime is not installed in the test env.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

# every test in this file needs onnxruntime — skip the whole module if it is missing
ort = pytest.importorskip("onnxruntime", reason="onnxruntime not installed")

from src.deploy.benchmark import benchmark_onnx_model, run_full_benchmark
from src.deploy.quantize_onnx import quantize_onnx_model


# ---------- stub model ----------


class TinyModel(nn.Module):
    """
    minimal stand-in for AgeEstimator. same input/output shape contract
    (3x224x224 in, 100 logits out) so the export + quantize + benchmark
    code paths are exercised without loading a 23M-param ResNet-50.
    """

    def __init__(self, num_logits: int = 100):
        super().__init__()
        # pool to (1,1) so the linear layer has a fixed input dim regardless
        # of input image size
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(3, num_logits)

    def forward(self, x):
        x = self.pool(x)
        x = self.flatten(x)
        return self.fc(x)


# ---------- helpers ----------


def _export_tiny_to_onnx(path: Path, image_size: int = 64) -> None:
    """export the stub model to ONNX in the same way export_onnx.py does."""
    model = TinyModel().eval()
    dummy = torch.randn(1, 3, image_size, image_size)
    path.parent.mkdir(parents=True, exist_ok=True)

    # same legacy-exporter pin used in src/deploy/export_onnx.py — keeps the
    # tests aligned with what the real export does at production time
    export_kwargs = dict(
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch_size"}, "logits": {0: "batch_size"}},
        opset_version=17,
        do_constant_folding=True,
    )
    import inspect
    if "dynamo" in inspect.signature(torch.onnx.export).parameters:
        export_kwargs["dynamo"] = False
    torch.onnx.export(model, dummy, str(path), **export_kwargs)


# ---------- export tests ----------


class TestExportPath:
    """exercises the export helper indirectly via the same code path."""

    def test_onnx_file_is_created(self, tmp_path: Path):
        out = tmp_path / "tiny.onnx"
        _export_tiny_to_onnx(out)
        assert out.exists()
        assert out.stat().st_size > 0

    def test_onnx_runs_in_onnxruntime(self, tmp_path: Path):
        # smoke check — the exported file must load and run without error
        out = tmp_path / "tiny.onnx"
        _export_tiny_to_onnx(out, image_size=64)
        sess = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
        sample = np.random.randn(1, 3, 64, 64).astype(np.float32)
        outputs = sess.run(None, {"input": sample})
        assert outputs[0].shape == (1, 100)

    def test_dynamic_batch_size(self, tmp_path: Path):
        # dynamic_axes config means the same model accepts different batch sizes
        out = tmp_path / "tiny.onnx"
        _export_tiny_to_onnx(out, image_size=64)
        sess = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
        for batch in (1, 4, 8):
            sample = np.random.randn(batch, 3, 64, 64).astype(np.float32)
            outputs = sess.run(None, {"input": sample})
            assert outputs[0].shape == (batch, 100)


# ---------- quantization tests ----------


class TestQuantization:
    def test_int8_file_is_created(self, tmp_path: Path):
        # export -> quantize -> the int8 file must exist on disk
        fp32 = tmp_path / "tiny.onnx"
        int8 = tmp_path / "tiny_int8.onnx"
        _export_tiny_to_onnx(fp32)
        info = quantize_onnx_model(fp32, int8)
        assert int8.exists()
        assert int8.stat().st_size > 0
        assert info["int8_path"] == str(int8)

    def test_int8_model_runs(self, tmp_path: Path):
        # quantized model must still load and produce 100 logits
        fp32 = tmp_path / "tiny.onnx"
        int8 = tmp_path / "tiny_int8.onnx"
        _export_tiny_to_onnx(fp32, image_size=64)
        quantize_onnx_model(fp32, int8)

        sess = ort.InferenceSession(str(int8), providers=["CPUExecutionProvider"])
        sample = np.random.randn(1, 3, 64, 64).astype(np.float32)
        outputs = sess.run(None, {"input": sample})
        assert outputs[0].shape == (1, 100)

    def test_quantize_returns_size_info(self, tmp_path: Path):
        # the info dict has the fields the README and benchmark consume
        fp32 = tmp_path / "tiny.onnx"
        int8 = tmp_path / "tiny_int8.onnx"
        _export_tiny_to_onnx(fp32)
        info = quantize_onnx_model(fp32, int8)

        assert "fp32_size_mb" in info
        assert "int8_size_mb" in info
        assert "size_reduction_pct" in info
        # for a tiny model the reduction may be small or even slightly negative
        # because of overhead — only check the field type, not magnitude
        assert isinstance(info["fp32_size_mb"], (int, float))
        assert isinstance(info["int8_size_mb"], (int, float))


# ---------- benchmark tests ----------


class TestBenchmark:
    def test_benchmark_returns_expected_keys(self, tmp_path: Path):
        # the structure of the result dict is what the JSON writer relies on
        fp32 = tmp_path / "tiny.onnx"
        _export_tiny_to_onnx(fp32, image_size=64)

        result = benchmark_onnx_model(fp32, num_runs=20, num_warmup=2,
                                      image_size=64)
        assert "model_path" in result
        assert "model_size_mb" in result
        assert "latency_ms" in result
        for k in ("mean", "p50", "p95", "p99", "min", "max", "stddev"):
            assert k in result["latency_ms"]

    def test_benchmark_latencies_are_positive(self, tmp_path: Path):
        # negative or zero latency would mean the timing loop is broken
        fp32 = tmp_path / "tiny.onnx"
        _export_tiny_to_onnx(fp32, image_size=64)

        result = benchmark_onnx_model(fp32, num_runs=20, num_warmup=2,
                                      image_size=64)
        for k in ("mean", "p50", "p95", "p99", "min", "max"):
            assert result["latency_ms"][k] > 0

    def test_benchmark_percentile_ordering(self, tmp_path: Path):
        # p50 <= p95 <= p99 by definition. catches a bug in the percentile call.
        fp32 = tmp_path / "tiny.onnx"
        _export_tiny_to_onnx(fp32, image_size=64)

        result = benchmark_onnx_model(fp32, num_runs=50, num_warmup=2,
                                      image_size=64)
        lat = result["latency_ms"]
        assert lat["p50"] <= lat["p95"]
        assert lat["p95"] <= lat["p99"]

    def test_full_benchmark_writes_json(self, tmp_path: Path):
        # end-to-end: export, quantize, benchmark, JSON file on disk with the
        # right top-level shape
        fp32 = tmp_path / "tiny.onnx"
        int8 = tmp_path / "tiny_int8.onnx"
        report_path = tmp_path / "benchmark_results.json"

        _export_tiny_to_onnx(fp32, image_size=64)
        quantize_onnx_model(fp32, int8)

        report = run_full_benchmark(fp32, int8, report_path, num_runs=20,
                                    image_size=64)

        assert report_path.exists()

        with report_path.open() as f:
            on_disk = json.load(f)
        assert "fp32" in on_disk
        assert "int8" in on_disk
        assert "summary" in on_disk
        assert "speedup_vs_fp32" in on_disk["summary"]
        assert "meets_latency_target" in on_disk["summary"]
