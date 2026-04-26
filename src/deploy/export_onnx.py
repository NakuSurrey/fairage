"""
PyTorch -> ONNX export for the trained age estimator.

ONNX is an open model format. exporting from PyTorch to ONNX lets the
model run anywhere that has an ONNX runtime — Linux server, Windows
laptop, even mobile — without needing PyTorch installed.

flow:
    1. load the best checkpoint from artifacts/checkpoints/age_model_best.pt
    2. trace the model with a dummy input to capture every op
    3. write the ONNX file
    4. run a parity check — same input through PyTorch and through ONNX,
       confirm the outputs match within float tolerance

why parity matters:
    a silent shape or op mismatch during export makes the ONNX model
    produce nonsense at inference, but the API would still return a 200.
    catching this here means we never ship a broken model.
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
from src.models.age_estimator import AgeEstimator


def export_age_model_to_onnx(
    checkpoint_path: Path,
    output_path: Path,
    image_size: int = IMAGE_SIZE,
    opset_version: int = 17,
) -> None:
    """
    convert a trained PyTorch age model into an ONNX file.

    args:
        checkpoint_path — path to torch .pt file written by train_age.py
        output_path     — where to save the .onnx file
        image_size      — input image side length (224 for ResNet-50)
        opset_version   — ONNX opset, 17 is current stable, supported by
                          onnxruntime 1.19+

    writes the ONNX model to output_path. raises RuntimeError if the
    parity check between torch and onnxruntime fails.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # build a fresh model with no pretrained weights — checkpoint contains
    # the trained ones we actually want
    model = AgeEstimator(pretrained=False)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state"])

    # eval mode turns off dropout and freezes batch norm running stats —
    # required for a deterministic export
    model.eval()

    # dummy input — random tensor of the same shape as a real preprocessed image.
    # ONNX export traces the model by running this through it, so the shapes
    # and dtypes here become the exported model's expected input.
    dummy = torch.randn(1, 3, image_size, image_size)

    # dynamic_axes lets the deployed model accept any batch size at runtime
    # without re-exporting. dim 0 (batch) is the only one we want flexible.
    #
    # use the legacy TorchScript-based exporter for stability — torch 2.4.1
    # uses it by default, torch 2.5+ defaults to the new dynamo exporter
    # which has rougher edges with onnxruntime quantization. passing
    # dynamo=False explicitly pins behaviour across torch versions.
    export_kwargs = dict(
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={
            "input":  {0: "batch_size"},
            "logits": {0: "batch_size"},
        },
        opset_version=opset_version,
        do_constant_folding=True,  # fold constant ops at export time, smaller graph
    )
    # dynamo kwarg only exists in torch 2.5+; older versions use the legacy
    # exporter unconditionally so adding it would crash. detect and pass safely.
    import inspect
    if "dynamo" in inspect.signature(torch.onnx.export).parameters:
        export_kwargs["dynamo"] = False

    torch.onnx.export(model, dummy, str(output_path), **export_kwargs)

    # parity check — ONNX output must match PyTorch output within float tolerance
    _verify_parity(model, output_path, dummy)
    print(f"exported and verified: {output_path}")


def _verify_parity(
    torch_model: torch.nn.Module,
    onnx_path: Path,
    sample_input: torch.Tensor,
    tolerance: float = 1e-4,
) -> None:
    """
    run the same input through the torch model and through onnxruntime.
    raise if outputs differ by more than `tolerance` per element.

    1e-4 is loose enough to accept normal floating-point drift between
    torch and onnxruntime (different op kernels), tight enough to catch
    a real bug like a missing layer.
    """
    # import here so the rest of the file works even if onnxruntime is missing
    # at edit time. real export needs it; failing here gives a clear message.
    try:
        import onnxruntime as ort
    except ImportError as e:
        raise RuntimeError(
            "onnxruntime is required for export verification. "
            "install with: pip install onnxruntime"
        ) from e

    # torch output — no_grad keeps memory low and matches inference mode
    with torch.no_grad():
        torch_out = torch_model(sample_input).cpu().numpy()

    # onnxruntime output — wrap the file in a session, run with the same input
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    ort_out = sess.run(["logits"], {"input": sample_input.cpu().numpy()})[0]

    # max absolute element-wise difference — tighter and more interpretable than allclose
    max_diff = float(np.abs(torch_out - ort_out).max())
    if max_diff > tolerance:
        raise RuntimeError(
            f"torch vs onnx parity check failed: max abs diff = {max_diff:.2e}, "
            f"tolerance = {tolerance:.2e}. the exported model would not match "
            f"the trained model at inference."
        )

    print(f"parity check passed: max abs diff = {max_diff:.2e} (tolerance {tolerance})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=str(CHECKPOINTS_DIR / "age_model_best.pt"),
        help="path to the trained torch checkpoint",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(EXPORTS_DIR / "age_model.onnx"),
        help="where to save the exported ONNX file",
    )
    parser.add_argument("--opset", type=int, default=17)
    args = parser.parse_args()

    export_age_model_to_onnx(
        checkpoint_path=Path(args.checkpoint),
        output_path=Path(args.output),
        opset_version=args.opset,
    )


if __name__ == "__main__":
    main()
