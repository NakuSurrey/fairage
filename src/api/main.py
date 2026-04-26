"""
FastAPI app for FairAge.

endpoints:
    GET  /health        — quick ok check, used by load balancer + Docker
    POST /estimate-age  — accepts image bytes, returns age + PAD score
    POST /explain       — accepts image bytes, returns saliency heatmap
    GET  /bias-report   — returns the precomputed bias audit JSON

design choice — eager model loading via the lifespan context manager:
    the lifespan function runs once on startup. it builds the
    InferenceEngine, which loads both ONNX sessions into memory.
    the engine is stashed on app.state so every request handler can
    pull it back out. on shutdown the engine drops out of scope and
    the sessions are garbage-collected. this is the textbook pattern
    in modern FastAPI (lifespan replaces the older startup/shutdown
    event hooks since FastAPI 0.93+).

design choice — engine on app.state, not a module global:
    keeps test isolation clean. tests can build a TestClient with
    a fresh app and inject a stub engine. a module-level singleton
    would leak state between tests.
"""

from __future__ import annotations

import json
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

from fastapi import FastAPI, File, HTTPException, Request, UploadFile

from src.api.inference import (
    DEFAULT_SALIENCY_GRID,
    DEFAULT_SALIENCY_PATCH_PX,
    InferenceEngine,
)
from src.api.schemas import (
    BiasGroup,
    BiasOverall,
    BiasReportResponse,
    BiasWorstGap,
    EstimateAgeResponse,
    ExplainResponse,
    HealthResponse,
)
from src.config import ARTIFACTS_DIR, IMAGE_SIZE

# how big a request body the API will accept. images normally arrive
# under 1 MB after JPEG compression — 10 MB is a generous ceiling that
# stops resource-exhaustion uploads cold.
MAX_UPLOAD_BYTES = 10 * 1024 * 1024

# the model_version string returned in every response. comes from an
# env var so each deployment can stamp its build without code changes.
# falls back to "dev" for local runs.
MODEL_VERSION = os.environ.get("MODEL_VERSION", "dev")

# bias report path — written by src/audit/bias_audit.py during Phase 5
BIAS_REPORT_PATH = ARTIFACTS_DIR / "bias_report.json"

logger = logging.getLogger("fairage.api")
logging.basicConfig(level=logging.INFO)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """
    runs once on startup, once on shutdown. the recommended way to
    set up shared resources in modern FastAPI.

    eager model load happens here — both ONNX sessions are warmed
    before the first user request can land.
    """
    logger.info("startup — eager-loading ONNX inference engines")
    try:
        engine = InferenceEngine()
        app.state.engine = engine
        logger.info(
            "engine ready — age=%s, pad=%s, model_version=%s",
            engine.age_model_path.name,
            engine.pad_model_path.name,
            MODEL_VERSION,
        )
    except FileNotFoundError as e:
        # the API will still come up, but /estimate-age will return 503.
        # this lets /health respond honestly instead of crashing the pod.
        logger.error("engine init failed: %s", e)
        app.state.engine = None

    # yield hands control back to FastAPI; everything after the yield
    # runs on shutdown
    yield

    logger.info("shutdown — releasing inference engine")
    app.state.engine = None


app = FastAPI(
    title="FairAge",
    description="Bias-audited age estimation with presentation attack detection",
    version=MODEL_VERSION,
    lifespan=lifespan,
)


def _get_engine(request: Request) -> InferenceEngine:
    """small helper — pulls the engine off app.state, raises 503 if absent.

    why not a global: keeps tests isolated. tests build their own app
    and put their own engine on it.
    """
    engine = getattr(request.app.state, "engine", None)
    if engine is None:
        raise HTTPException(
            status_code=503,
            detail="inference engine unavailable — model files not loaded",
        )
    return engine


async def _read_image_bytes(file: UploadFile) -> bytes:
    """
    shared upload validation used by /estimate-age and /explain.

    enforces:
        - content-type starts with image/
        - body is non-empty
        - body is within MAX_UPLOAD_BYTES

    returns the raw bytes ready for engine.predict() or engine.explain().
    """
    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail=f"expected an image upload, got content-type {file.content_type}",
        )

    body = await file.read()
    if len(body) == 0:
        raise HTTPException(status_code=400, detail="empty upload")
    if len(body) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"upload too large: {len(body)} bytes (max {MAX_UPLOAD_BYTES})",
        )
    return body


# ---------- /health ----------


@app.get("/health", response_model=HealthResponse, tags=["meta"])
async def health(request: Request) -> HealthResponse:
    """
    light-weight liveness check for the load balancer and Docker.
    deliberately avoids running any model inference — checking is_alive,
    not is_correct. the bias-report and estimate-age endpoints test
    correctness end-to-end.
    """
    engine = getattr(request.app.state, "engine", None)
    return HealthResponse(
        status="ok" if engine is not None else "degraded",
        age_model_loaded=engine is not None,
        pad_model_loaded=engine is not None,
    )


# ---------- /estimate-age ----------


@app.post(
    "/estimate-age",
    response_model=EstimateAgeResponse,
    tags=["inference"],
)
async def estimate_age(
    request: Request,
    file: UploadFile = File(..., description="face image (JPEG/PNG)"),
) -> EstimateAgeResponse:
    """
    accept a face image, return predicted age + PAD score.

    flow:
        1. validate content type — must look like an image
        2. read body, enforce size cap (MAX_UPLOAD_BYTES)
        3. hand bytes to engine.predict()
        4. wrap PredictionResult into a typed response
    """
    engine = _get_engine(request)
    body = await _read_image_bytes(file)

    # the engine catches PIL decode errors and surfaces them as ValueError —
    # convert to 400 here so the client sees a clean error message
    try:
        result = engine.predict(body)
    except (ValueError, OSError) as e:
        raise HTTPException(
            status_code=400,
            detail=f"could not decode image: {e}",
        )

    return EstimateAgeResponse(
        is_spoof=result.is_spoof,
        pad_score=round(result.pad_score, 4),
        estimated_age=(round(result.estimated_age, 2)
                       if result.estimated_age is not None else None),
        age_confidence=(round(result.age_confidence, 4)
                        if result.age_confidence is not None else None),
        inference_ms=result.inference_ms,
        model_version=MODEL_VERSION,
    )


# ---------- /explain ----------


@app.post(
    "/explain",
    response_model=ExplainResponse,
    tags=["inference"],
)
async def explain(
    request: Request,
    file: UploadFile = File(..., description="face image (JPEG/PNG)"),
    grid: int = DEFAULT_SALIENCY_GRID,
    patch_px: int = DEFAULT_SALIENCY_PATCH_PX,
) -> ExplainResponse:
    """
    accept a face image, return an occlusion saliency heatmap.

    the heatmap shows which pixel regions most influence the predicted
    age. opt-in endpoint — the main /estimate-age path stays fast and
    explanations are computed only when explicitly requested.

    query params:
        grid     — saliency grid size, 2-64. default 12 (= 144 cells).
        patch_px — occlusion patch side in pixels, 4-128. default 28.

    response: ExplainResponse with the saliency_map as a nested list
    of floats normalised to [0, 1]. consumers render as a heatmap.
    """
    engine = _get_engine(request)

    # validate query params — pydantic does not validate path/query
    # ranges automatically the way it does response fields
    if not (2 <= grid <= 64):
        raise HTTPException(status_code=400,
                            detail="grid must be between 2 and 64")
    if not (4 <= patch_px <= 128):
        raise HTTPException(status_code=400,
                            detail="patch_px must be between 4 and 128")

    body = await _read_image_bytes(file)

    try:
        result = engine.explain(body, grid=grid, patch_px=patch_px)
    except (ValueError, OSError) as e:
        raise HTTPException(
            status_code=400,
            detail=f"could not decode image: {e}",
        )

    # convert numpy array to nested list — pydantic does not serialise
    # numpy arrays directly, but plain lists of floats are JSON-native
    return ExplainResponse(
        saliency_map=result.saliency_map.tolist(),
        baseline_age=round(result.baseline_age, 2),
        inference_ms=result.inference_ms,
        grid=grid,
        image_size=IMAGE_SIZE,
    )


# ---------- /bias-report ----------


@app.get(
    "/bias-report",
    response_model=BiasReportResponse,
    tags=["audit"],
)
async def bias_report() -> BiasReportResponse:
    """
    return the precomputed bias audit from artifacts/bias_report.json.

    the audit is regenerated offline (Phase 5 notebook). serving it
    through the API gives the dashboard and any external client a
    typed view of the latest fairness numbers without exposing the
    file system.
    """
    if not BIAS_REPORT_PATH.exists():
        raise HTTPException(
            status_code=404,
            detail=(
                "bias report not yet generated. "
                "run notebooks/04_bias_audit.ipynb to produce "
                f"{BIAS_REPORT_PATH.name}."
            ),
        )

    try:
        with BIAS_REPORT_PATH.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        raise HTTPException(
            status_code=500,
            detail=f"bias report is corrupt: {e}",
        )

    # pydantic validates the shape — if the audit format ever drifts,
    # this call surfaces it loudly as a 500 instead of returning broken
    # JSON to the client
    return BiasReportResponse(
        overall=BiasOverall(**data["overall"]),
        groups=[BiasGroup(**g) for g in data["groups"]],
        worst_gap=BiasWorstGap(**data["worst_gap"]),
    )
