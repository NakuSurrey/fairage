"""
inference layer for the FairAge API.

loads both quantized ONNX models on startup and exposes a single
predict() function that runs PAD first, then age estimation.

design choice — eager loading on startup:
    every request gets a pre-warmed onnxruntime session. zero cold-start
    penalty at user time. session memory is small (~30 MB total for
    int8-quantized age + PAD models on Hetzner).

design choice — single InferenceEngine class:
    holds both sessions in one place. main.py instantiates this once
    on startup and stashes it on app.state. tests can swap in a stub
    by injecting a different engine instance.

flow per request:
    1. raw bytes -> PIL image -> normalised float tensor [1, 3, 224, 224]
    2. PAD session runs -> softmax -> probability of attack
    3. if attack score above threshold, return refusal (no age computed)
    4. age session runs -> sigmoid -> sum -> predicted age
    5. response packaged into a typed dict the schemas layer wraps
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
