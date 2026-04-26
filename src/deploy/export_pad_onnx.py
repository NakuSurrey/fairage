"""
PyTorch -> ONNX export for the trained PAD detector.

mirrors src/deploy/export_onnx.py for the age model, but for the small
PAD CNN. same flow:
    1. load best checkpoint from artifacts/checkpoints/pad_model_best.pt
    2. trace with a dummy input
    3. write the ONNX file
    4. parity-check torch vs onnxruntime output

after this runs, the int8 quantization + benchmark pipeline from Phase 7
also works on the PAD model — same code path, different file paths.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

# repo root on sys.path so `src.*` imports work no matter where this is run from
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.config import CHECKPOINTS_DIR, EXPORTS_DIR, IMAGE_SIZE
from src.models.pad_detector import PADDetector


def export_pad_model_to_onnx(
    checkpoint_path: Path,
    output_path: Path,
    image_size: int = IMAGE_SIZE,
    opset_version: int = 17,
) -> None:
    """
    convert a trained PyTorch PAD model into an ONNX file.

    same arg shape as export_age_model_to_onnx — keeps the deploy
    pipeline symmetric across the two models.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # build a fresh model — checkpoint provides the trained weights
    model = PADDetector()
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    dummy = torch.randn(1, 3, image_size, image_size)

    # same legacy-exporter pin as the age export — torch 2.4.1 default,
    # forward-compatible with newer versions via the dynamo=False kwarg
    export_kwargs = dict(
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={
            "input":  {0: "batch_size"},
            "logits": {0: "batch_size"},
        },
        opset_version=opset_version,
        do_constant_folding=True,
    )
    import inspect
    if "dynamo" in inspect.signature(torch.onnx.export).parameters:
        export_kwargs["dynamo"] = False

    torch.onnx.export(model, dummy, str(output_path), **export_kwargs)

    # parity check — torch and onnxruntime output must match within tolerance
    _verify_parity(model, output_path, dummy)
    print(f"exported and verified: {output_path}")


def _verify_parity(
    torch_model: torch.nn.Module,
    onnx_path: Path,
    sample_input: torch.Tensor,
    tolerance: float = 1e-4,
) -> None:
    """run sample input through both runtimes, raise if outputs disagree."""
    try:
        import onnxruntime as ort
    except ImportError as e:
        raise RuntimeError(
            "onnxruntime is required for export verification. "
            "install with: pip install onnxruntime"
        ) from e

    with torch.no_grad():
        torch_out = torch_model(sample_input).cpu().numpy()

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    ort_out = sess.run(["logits"], {"input": sample_input.cpu().numpy()})[0]

    max_diff = float(np.abs(torch_out - ort_out).max())
    if max_diff > tolerance:
        raise RuntimeError(
            f"torch vs onnx parity check failed: max abs diff = {max_diff:.2e}, "
            f"tolerance = {tolerance:.2e}. exported model would not match "
            f"the trained model at inference."
        )

    print(f"parity check passed: max abs diff = {max_diff:.2e} (tolerance {tolerance})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=str(CHECKPOINTS_DIR / "pad_model_best.pt"),
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(EXPORTS_DIR / "pad_model.onnx"),
    )
    parser.add_argument("--opset", type=int, default=17)
    args = parser.parse_args()

    export_pad_model_to_onnx(
        checkpoint_path=Path(args.checkpoint),
        output_path=Path(args.output),
        opset_version=args.opset,
    )


if __name__ == "__main__":
    main()
