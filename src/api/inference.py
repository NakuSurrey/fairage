"""
inference layer for the FairAge API.

loads both quantized ONNX models on startup and exposes:
    - predict()  — full pipeline: PAD first, then age (Phase 8)
    - explain()  — occlusion saliency map for the age model (Phase 9)

design choice — eager loading on startup:
    every request gets a pre-warmed onnxruntime session. zero cold-start
    penalty at user time. session memory is small (~30 MB total for
    int8-quantized age + PAD models on Hetzner).

design choice — single InferenceEngine class:
    holds both sessions in one place. main.py instantiates this once
    on startup and stashes it on app.state. tests can swap in a stub
    by injecting a different engine instance.

design choice — occlusion saliency, not gradient-based:
    the production model runs through onnxruntime, not torch. occlusion
    works with any backend — slide a grey patch across the image, watch
    how the predicted age shifts, the size of the shift at each location
    is the saliency value for that pixel region. slower than Captum's
    IntegratedGradients (50-80 forward passes per image vs 1) but
    backend-agnostic, and explainability is opt-in so the latency cost
    is acceptable.
"""

from __future__ import annotations

import io
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

from src.config import (
    EXPORTS_DIR,
    IMAGE_SIZE,
    IMAGENET_MEAN,
    IMAGENET_STD,
)

# default model paths — production picks the int8-quantized files.
# these can be overridden via env vars or constructor args.
DEFAULT_AGE_MODEL_PATH = EXPORTS_DIR / "age_model_int8.onnx"
DEFAULT_PAD_MODEL_PATH = EXPORTS_DIR / "pad_model_int8.onnx"

# spoof threshold — PAD model outputs P(attack). above this, refuse.
# 0.5 is the natural decision boundary; production might tune lower
# (more false rejects, fewer false accepts) if the operational risk
# of accepting a spoof is high. 0.5 is the documented default.
DEFAULT_SPOOF_THRESHOLD = 0.5

# saliency defaults — chosen so a single explain() call finishes in
# roughly 1-2 seconds on a 4-vCPU box. larger grids give finer maps
# at higher latency cost.
DEFAULT_SALIENCY_GRID = 12   # 12x12 = 144 forward passes per image
DEFAULT_SALIENCY_PATCH_PX = 28  # patch covers ~12% of one side


@dataclass
class PredictionResult:
    """structured output of the full inference pipeline.

    one of two shapes:
      - spoof rejected: pad_score above threshold, age fields are None
      - real face:      both pad_score and age fields populated
    """
    is_spoof: bool
    pad_score: float                # P(attack) from the PAD model
    estimated_age: Optional[float]  # None if rejected as spoof
    age_confidence: Optional[float] # None if rejected as spoof
    inference_ms: float             # wall-clock time end-to-end


@dataclass
class SaliencyResult:
    """
    output of the explain() call.

    saliency_map shape is (H, W) — one float per pixel region. values
    are in [0, 1] after normalisation, where 1 means "covering this
    region changes the prediction most" and 0 means "no effect".

    baseline_age is what the model predicts on the unmasked image.
    used by the UI to caption the heatmap.
    """
    saliency_map: np.ndarray   # (H, W), float32, normalised [0, 1]
    baseline_age: float
    inference_ms: float


def _preprocess_image(image_bytes: bytes, image_size: int = IMAGE_SIZE) -> np.ndarray:
    """
    bytes -> normalised float32 tensor of shape (1, 3, H, W).

    runs the same transform pipeline as the deterministic eval transform
    defined in src/data/transforms.py — that lineup is critical so the
    served model sees the exact distribution it was trained against.

    why we re-implement here instead of importing the torch transform:
        the API container does not install torch. keeping inference
        pure-numpy + PIL means a much smaller Docker image and faster
        cold start. the math is identical.
    """
    # PIL handles every common format (JPEG, PNG, BMP). RGB conversion
    # drops alpha channels and forces 3 channels for the model.
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    # match torchvision Resize((H, W)) — bilinear is the torch default
    img = img.resize((image_size, image_size), Image.BILINEAR)

    # to float32 array in [0, 1], shape (H, W, 3)
    arr = np.asarray(img, dtype=np.float32) / 255.0

    # imagenet normalisation — same constants used in src/data/transforms.py
    mean = np.array(IMAGENET_MEAN, dtype=np.float32)
    std = np.array(IMAGENET_STD, dtype=np.float32)
    arr = (arr - mean) / std

    # HWC -> CHW, then add batch dim -> (1, 3, H, W)
    arr = np.transpose(arr, (2, 0, 1))
    arr = np.expand_dims(arr, axis=0)
    return arr


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """numerically stable softmax — subtracts max before exp."""
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    """sigmoid for ordinal age decoding."""
    return 1.0 / (1.0 + np.exp(-x))


def _decode_age(age_logits: np.ndarray) -> tuple[float, float]:
    """
    convert ordinal logits to a predicted age + confidence score.

    age_logits shape: (1, num_thresholds)

    age formula: count thresholds whose sigmoid > 0.5
    confidence:  mean sigmoid magnitude of the binary decisions —
                 1.0 means every threshold is decisively predicted,
                 0.5 means the model is hovering near the boundary
                 on every threshold.
    """
    probs = _sigmoid(age_logits[0])
    age = float((probs > 0.5).sum())

    # confidence: how far each prob is from the 0.5 boundary, averaged.
    # rescaled to [0, 1] where 1 = always confident, 0 = always uncertain.
    distance_from_boundary = np.abs(probs - 0.5) * 2.0
    confidence = float(distance_from_boundary.mean())
    return age, confidence


def _predicted_age_from_tensor(session, tensor: np.ndarray) -> float:
    """
    helper used by both predict() and explain() — runs the age session
    on a preprocessed tensor and decodes to a float age.
    """
    age_logits = session.run(None, {"input": tensor})[0]
    age, _ = _decode_age(age_logits)
    return age


class InferenceEngine:
    """
    holds both ONNX runtime sessions and exposes a single predict() call.

    one instance lives on app.state.engine for the life of the FastAPI
    process. eager-loaded in the lifespan handler — no lazy imports,
    no cold-start surprise.
    """

    def __init__(
        self,
        age_model_path: Path | str = DEFAULT_AGE_MODEL_PATH,
        pad_model_path: Path | str = DEFAULT_PAD_MODEL_PATH,
        spoof_threshold: float = DEFAULT_SPOOF_THRESHOLD,
    ) -> None:
        # import here so the module is importable even if onnxruntime is
        # missing in some test contexts. real serving needs it.
        try:
            import onnxruntime as ort
        except ImportError as e:
            raise RuntimeError(
                "onnxruntime is required to serve the API. "
                "install with: pip install onnxruntime"
            ) from e

        self.age_model_path = Path(age_model_path)
        self.pad_model_path = Path(pad_model_path)
        self.spoof_threshold = spoof_threshold

        if not self.age_model_path.exists():
            raise FileNotFoundError(
                f"age model not found at {self.age_model_path}. "
                f"run `python -m src.deploy.export_onnx` and "
                f"`python -m src.deploy.quantize_onnx` first."
            )
        if not self.pad_model_path.exists():
            raise FileNotFoundError(
                f"PAD model not found at {self.pad_model_path}. "
                f"run `python -m src.deploy.export_pad_onnx` and "
                f"`python -m src.deploy.quantize_onnx` first."
            )

        # CPU-only runtime — production target is a 4-vCPU Hetzner box
        # with no GPU. CPUExecutionProvider gives consistent latency.
        self._age_session = ort.InferenceSession(
            str(self.age_model_path),
            providers=["CPUExecutionProvider"],
        )
        self._pad_session = ort.InferenceSession(
            str(self.pad_model_path),
            providers=["CPUExecutionProvider"],
        )

    def predict(self, image_bytes: bytes) -> PredictionResult:
        """
        run the full inference pipeline on one image.

        PAD runs first. if the attack score is above the spoof threshold,
        the age model is skipped and the result reports a rejection —
        same response shape, just with None for age fields.
        """
        # perf_counter is monotonic and microsecond-resolution
        start = time.perf_counter()

        # one preprocessing pass — both models accept the same shape
        tensor = _preprocess_image(image_bytes)

        # PAD first — gates whether age estimation runs at all
        pad_logits = self._pad_session.run(None, {"input": tensor})[0]
        pad_probs = _softmax(pad_logits, axis=1)
        pad_score = float(pad_probs[0, 1])  # index 1 = "attack" class

        if pad_score > self.spoof_threshold:
            # spoof short-circuit — don't waste cycles on the age model,
            # don't return a misleading age estimate for a fake input
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            return PredictionResult(
                is_spoof=True,
                pad_score=pad_score,
                estimated_age=None,
                age_confidence=None,
                inference_ms=round(elapsed_ms, 2),
            )

        # real face -> run age model, decode logits, package result
        age_logits = self._age_session.run(None, {"input": tensor})[0]
        age, confidence = _decode_age(age_logits)

        elapsed_ms = (time.perf_counter() - start) * 1000.0
        return PredictionResult(
            is_spoof=False,
            pad_score=pad_score,
            estimated_age=age,
            age_confidence=confidence,
            inference_ms=round(elapsed_ms, 2),
        )

    def explain(
        self,
        image_bytes: bytes,
        grid: int = DEFAULT_SALIENCY_GRID,
        patch_px: int = DEFAULT_SALIENCY_PATCH_PX,
    ) -> SaliencyResult:
        """
        compute an occlusion-based saliency heatmap for the age model.

        the algorithm:
            1. run the model on the unmasked image -> baseline age
            2. for each cell in a `grid x grid` overlay:
                a. mask that cell with a grey patch
                b. run the model -> masked age
                c. store |masked_age - baseline_age| at that cell
            3. normalise the deltas to [0, 1]

        why occlusion and not gradients:
            the served model is a quantized ONNX file. ONNX Runtime does
            not give gradients. occlusion is purely forward-pass — works
            with any model format, any backend. matches Microsoft's
            interpretability approach for non-torch deployments.

        latency: roughly grid*grid forward passes. with grid=12 that is
            144 passes — about 1-2 seconds on a 4-vCPU Hetzner box for the
            int8-quantized model. acceptable for an opt-in /explain endpoint.

        args:
            image_bytes — raw upload bytes
            grid        — grid size (12 -> 144 occlusion cells)
            patch_px    — side length of the grey patch in pixels
        """
        start = time.perf_counter()

        # baseline tensor and baseline age — computed once, reused below
        tensor = _preprocess_image(image_bytes)
        baseline_age = _predicted_age_from_tensor(self._age_session, tensor)

        # the saliency map is laid out at the input resolution so the UI
        # can overlay it directly on the resized image. start as zeros.
        h, w = IMAGE_SIZE, IMAGE_SIZE
        saliency = np.zeros((h, w), dtype=np.float32)

        # cell stride — covers the whole image in `grid` steps along each axis
        stride_h = max(1, h // grid)
        stride_w = max(1, w // grid)

        # the grey patch — neutral value in normalised space. since the
        # tensor is already imagenet-normalised, a "grey" patch in pixel
        # space is roughly zero in normalised space. using zero keeps the
        # math simple and the visualisation interpretable.
        patch_value = 0.0

        # half_patch is how far the patch extends from the cell centre.
        # patches at the image edge are clipped automatically by the
        # numpy slice — cells near the corner cover fewer pixels than
        # cells near the centre, which matches what occlusion saliency
        # should do (no padding artefacts).
        half = patch_px // 2

        # main occlusion loop — one forward pass per (row, col) cell.
        # iterating over flat cell coords keeps the loop tight; could
        # be batched into one big tensor of shape (grid*grid, 3, H, W)
        # but that uses ~6x more memory and the speedup is small.
        for row in range(grid):
            for col in range(grid):
                cy = row * stride_h + stride_h // 2
                cx = col * stride_w + stride_w // 2

                # numpy slicing handles edge-clipping automatically
                y_lo, y_hi = max(0, cy - half), min(h, cy + half)
                x_lo, x_hi = max(0, cx - half), min(w, cx + half)

                # work on a copy — must not mutate the baseline tensor
                masked = tensor.copy()
                masked[:, :, y_lo:y_hi, x_lo:x_hi] = patch_value

                masked_age = _predicted_age_from_tensor(self._age_session, masked)
                delta = abs(masked_age - baseline_age)

                # paint the delta over the cell's pixel region. when cells
                # overlap (large patch_px relative to stride), max-merge
                # so the most-impactful cell wins.
                saliency[y_lo:y_hi, x_lo:x_hi] = np.maximum(
                    saliency[y_lo:y_hi, x_lo:x_hi], delta
                )

        # normalise to [0, 1] for visualisation. if the image has zero
        # variation in predictions, return all zeros instead of dividing
        # by zero.
        max_val = float(saliency.max())
        if max_val > 0:
            saliency = saliency / max_val

        elapsed_ms = (time.perf_counter() - start) * 1000.0
        return SaliencyResult(
            saliency_map=saliency.astype(np.float32),
            baseline_age=baseline_age,
            inference_ms=round(elapsed_ms, 2),
        )
