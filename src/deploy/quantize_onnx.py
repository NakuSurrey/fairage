"""
quantize the exported ONNX model to INT8.

quantization replaces every float32 weight with an int8 approximation.
this gives:
    - smaller model file (~4x smaller — float32 is 4 bytes, int8 is 1 byte)
    - faster CPU inference (int8 ops use vectorized CPU instructions)
    - tiny accuracy drop (typically under 1% for a well-trained model)

we use DYNAMIC quantization, not static:

    dynamic — quantize weights ahead of time, quantize activations on the
              fly during inference. no calibration data needed.

    static  — quantize weights AND activations ahead of time. needs a
              calibration dataset to figure out activation ranges. faster
              at runtime but more complex pipeline.

dynamic is the right choice here:
    - the project goal is sub-200ms CPU latency, not absolute peak speed
    - dynamic quantization gives most of the speedup with none of the
      calibration setup. saves a phase of work.
    - matches what most production deployments actually do for transformer
      and CNN models served via onnxruntime

two extra steps the bare quantize_dynamic call does NOT do:
    1. preprocess the FP32 graph (shape inference, optimisation) — without
       this the quantizer can leave Conv ops as ConvInteger at opset 10,
       which has no CPU kernel in some onnxruntime builds. ConvInteger
       becomes QLinearConv after preprocess + per-channel.
    2. per_channel=True — emits QLinearConv (universally supported on CPU)
       instead of the picky ConvInteger pattern. small accuracy bonus too.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

# repo root on sys.path so `src.*` imports work
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.config import EXPORTS_DIR


def quantize_onnx_model(
    fp32_path: Path,
    int8_path: Path,
) -> dict:
    """
    convert a float32 ONNX model into an INT8 quantized ONNX model.

    args:
        fp32_path — input float32 ONNX file (from export_onnx.py)
        int8_path — where to save the int8 quantized file

    returns dict with file size info — used by the benchmark step and the
    README to show the size reduction.
    """
    try:
        from onnxruntime.quantization import QuantType, quantize_dynamic
        from onnxruntime.quantization.shape_inference import quant_pre_process
    except ImportError as e:
        raise RuntimeError(
            "onnxruntime quantization tools are required. "
            "install with: pip install onnxruntime"
        ) from e

    int8_path.parent.mkdir(parents=True, exist_ok=True)

    # step 1 — preprocess the fp32 graph. runs symbolic shape inference and
    # graph optimisation. without this, dynamic quantization can fall back
    # to ConvInteger nodes at opset 10 inside an opset 17 graph, and some
    # onnxruntime CPU builds have no kernel for that combination.
    # the warning "Please consider to run pre-processing before quantization"
    # comes from skipping this exact step.
    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as tmp:
        preprocessed = Path(tmp.name)

    try:
        quant_pre_process(
            input_model=str(fp32_path),
            output_model_path=str(preprocessed),
            skip_optimization=False,
            skip_onnx_shape=False,
            skip_symbolic_shape=False,
        )

        # step 2 — quantize the preprocessed graph.
        # weight_type=QInt8 is the standard signed 8-bit choice.
        # per_channel=True emits QLinearConv (universally supported on CPU)
        # instead of ConvInteger (picky kernel coverage). small accuracy bump
        # comes with it for free.
        quantize_dynamic(
            model_input=str(preprocessed),
            model_output=str(int8_path),
            weight_type=QuantType.QUInt8,
            per_channel=True,
        )
    finally:
        # always delete the temp file, even if quantization failed
        preprocessed.unlink(missing_ok=True)

    # report file sizes — useful as a sanity check and a README number
    fp32_mb = fp32_path.stat().st_size / (1024 * 1024)
    int8_mb = int8_path.stat().st_size / (1024 * 1024)
    reduction = (1 - int8_mb / fp32_mb) * 100 if fp32_mb > 0 else 0.0

    info = {
        "fp32_path": str(fp32_path),
        "int8_path": str(int8_path),
        "fp32_size_mb": round(fp32_mb, 2),
        "int8_size_mb": round(int8_mb, 2),
        "size_reduction_pct": round(reduction, 1),
    }

    print(f"fp32: {fp32_mb:.1f} MB -> int8: {int8_mb:.1f} MB "
          f"({reduction:.0f}% smaller)")
    return info


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fp32",
        type=str,
        default=str(EXPORTS_DIR / "age_model.onnx"),
        help="path to the float32 ONNX file produced by export_onnx.py",
    )
    parser.add_argument(
        "--int8",
        type=str,
        default=str(EXPORTS_DIR / "age_model_int8.onnx"),
        help="where to save the INT8 quantized ONNX file",
    )
    args = parser.parse_args()

    quantize_onnx_model(
        fp32_path=Path(args.fp32),
        int8_path=Path(args.int8),
    )


if __name__ == "__main__":
    main()