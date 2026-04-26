"""
FairAge — Streamlit demo.

what this app does:
    - takes an image upload from the user
    - calls the FastAPI /estimate-age endpoint -> shows age + PAD score
    - on demand, calls /explain -> overlays a saliency heatmap
    - displays the latest bias audit alongside

design choice — the demo does not run any model itself:
    every prediction goes through the FastAPI service. one source of
    truth, one set of model files, two containers (API + UI). lets the
    Streamlit container stay small (no torch, no onnxruntime).

design choice — API URL via env var with a sensible default:
    Phase 10 docker-compose runs the API at http://api:8003 inside the
    compose network. local dev uses http://127.0.0.1:8003. one env var
    covers both.
"""

from __future__ import annotations

import io
import os

import numpy as np
import requests
import streamlit as st
from PIL import Image

# the API URL — overridable via env so docker-compose can point at the
# internal service name. local dev falls back to localhost.
API_URL = os.environ.get("FAIRAGE_API_URL", "http://127.0.0.1:8003")

# request timeout — cap so a hung API does not lock the UI forever.
# /explain runs ~1-2s; 30s is a generous ceiling for large grids.
DEFAULT_TIMEOUT_S = 30


# ---------- API helpers ----------


def call_estimate_age(image_bytes: bytes, filename: str = "upload.jpg") -> dict:
    """
    POST the image to /estimate-age, return the parsed JSON response.

    raises requests.HTTPError on non-2xx — the UI catches that and shows
    a friendly error instead of crashing.
    """
    files = {"file": (filename, image_bytes, "image/jpeg")}
    response = requests.post(
        f"{API_URL}/estimate-age",
        files=files,
        timeout=DEFAULT_TIMEOUT_S,
    )
    response.raise_for_status()
    return response.json()


def call_explain(image_bytes: bytes, grid: int = 12,
                 filename: str = "upload.jpg") -> dict:
    """
    POST the image to /explain, return the parsed JSON response.

    grid query param is forwarded to the API. larger grid = sharper
    heatmap but slower — UI exposes a slider for power users.
    """
    files = {"file": (filename, image_bytes, "image/jpeg")}
    response = requests.post(
        f"{API_URL}/explain",
        files=files,
        params={"grid": grid},
        timeout=DEFAULT_TIMEOUT_S,
    )
    response.raise_for_status()
    return response.json()


def call_bias_report() -> dict | None:
    """
    GET /bias-report. returns None if the report file is not yet on
    disk (the API responds 404 in that case) — UI hides the section.
    """
    response = requests.get(f"{API_URL}/bias-report", timeout=DEFAULT_TIMEOUT_S)
    if response.status_code == 404:
        return None
    response.raise_for_status()
    return response.json()


def call_health() -> dict | None:
    """
    GET /health. used by the sidebar to show whether the API is up
    and the models are loaded. returns None on connection error so
    the UI can show "API down" without traceback.
    """
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        response.raise_for_status()
        return response.json()
    except requests.RequestException:
        return None


# ---------- saliency overlay ----------


def overlay_saliency(image: Image.Image, saliency_map: list[list[float]]) -> Image.Image:
    """
    blend a heatmap of the saliency map onto the original image.

    flow:
        1. convert nested list -> numpy array shape (H, W)
        2. apply a colormap (red = high saliency, dark = low)
        3. resize to match the displayed image
        4. alpha-blend onto the original at 50% strength

    no matplotlib needed — kept pure-PIL/numpy so the Streamlit container
    stays lean.
    """
    sal = np.array(saliency_map, dtype=np.float32)
    h, w = sal.shape

    # match the original aspect ratio when overlaying. resize image to
    # match the saliency resolution rather than the other way round —
    # keeps the heatmap pixel-aligned.
    img_resized = image.convert("RGB").resize((w, h), Image.BILINEAR)
    img_arr = np.asarray(img_resized, dtype=np.float32)

    # build an RGB heatmap. red channel = saliency, green = 0,
    # blue = inverse saliency. simple but readable; matches the
    # convention of every "where the model looks" visualisation.
    heatmap = np.zeros((h, w, 3), dtype=np.float32)
    heatmap[:, :, 0] = sal * 255.0          # red rises with saliency
    heatmap[:, :, 2] = (1.0 - sal) * 80.0   # subtle blue tint elsewhere

    # alpha blend — 0.5 mixes the heatmap and image equally so both stay
    # readable. higher alpha (0.7) makes the heatmap dominate; lower
    # (0.3) makes the original dominate. 0.5 is the standard default.
    alpha = 0.5
    blended = (1 - alpha) * img_arr + alpha * heatmap
    blended = np.clip(blended, 0, 255).astype(np.uint8)
    return Image.fromarray(blended)


# ---------- streamlit UI ----------


def main() -> None:
    st.set_page_config(
        page_title="FairAge — Demo",
        page_icon="🧑",
        layout="wide",
    )
    st.title("FairAge")
    st.caption(
        "Bias-audited age estimation with presentation attack detection. "
        "Upload a face image; the API returns predicted age, a spoof "
        "score, and an optional saliency heatmap showing where the model "
        "is looking."
    )

    # ---- sidebar: API status + advanced controls ----
    with st.sidebar:
        st.subheader("API status")
        health = call_health()
        if health is None:
            st.error(f"API unreachable at {API_URL}")
        elif health.get("status") == "ok":
            st.success(f"API live at {API_URL}")
        else:
            st.warning(f"API degraded — models not loaded")

        st.subheader("Saliency settings")
        grid_size = st.slider(
            "Grid size",
            min_value=4, max_value=24, value=12, step=2,
            help="Larger grid -> sharper heatmap, slower. 12 is a good default.",
        )
        st.caption("Saliency is opt-in. Click the explain button below the "
                   "prediction to compute it.")

    # ---- main area: upload + results ----
    uploaded = st.file_uploader(
        "Upload a face image",
        type=["jpg", "jpeg", "png", "bmp"],
        help="JPEG or PNG, under 10 MB. UTKFace-style faces work best — "
             "frontal, head-and-shoulders, decent lighting.",
    )

    if uploaded is None:
        st.info("Upload an image above to see a prediction.")
        return

    image_bytes = uploaded.getvalue()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    # two columns: image on the left, prediction on the right
    col_img, col_pred = st.columns([1, 1])
    with col_img:
        st.image(image, caption="Uploaded image", use_container_width=True)

    with col_pred:
        st.subheader("Prediction")
        try:
            pred = call_estimate_age(image_bytes, filename=uploaded.name)
        except requests.RequestException as e:
            st.error(f"prediction failed: {e}")
            return

        if pred["is_spoof"]:
            st.error(
                f"⚠ Possible spoof — PAD score {pred['pad_score']:.2f} "
                f"(threshold 0.5). No age estimated."
            )
        else:
            st.metric(
                label="Estimated age",
                value=f"{pred['estimated_age']:.0f} years",
            )
            st.write(
                f"**PAD score:** {pred['pad_score']:.2f} "
                f"(low = real face)"
            )
            st.write(
                f"**Confidence:** {pred['age_confidence']:.2f}"
            )
            st.write(
                f"**Latency:** {pred['inference_ms']:.0f} ms · "
                f"**Model:** {pred['model_version']}"
            )

    # ---- saliency on demand ----
    # only offered when the input passed the spoof check; explaining a
    # rejected spoof is not informative
    if not pred.get("is_spoof"):
        st.divider()
        st.subheader("Where is the model looking?")
        st.caption(
            "Click below to compute an occlusion saliency map. Slides a "
            "grey patch across the image, tracks how the predicted age "
            "shifts, paints the magnitude. Backend-agnostic — works with "
            "the int8 ONNX model directly."
        )
        if st.button("Compute saliency"):
            with st.spinner("Sliding the patch — ~1-2 seconds for a 12x12 grid"):
                try:
                    explain = call_explain(
                        image_bytes,
                        grid=grid_size,
                        filename=uploaded.name,
                    )
                except requests.RequestException as e:
                    st.error(f"explain failed: {e}")
                    return

            heatmap_img = overlay_saliency(image, explain["saliency_map"])
            col_a, col_b = st.columns([1, 1])
            with col_a:
                st.image(image, caption="Original",
                         use_container_width=True)
            with col_b:
                st.image(heatmap_img, caption="Saliency overlay (red = high)",
                         use_container_width=True)
            st.caption(
                f"Baseline age: {explain['baseline_age']:.0f} years · "
                f"grid {explain['grid']}x{explain['grid']} · "
                f"latency {explain['inference_ms']:.0f} ms"
            )

    # ---- bias report block ----
    st.divider()
    st.subheader("Latest bias audit")
    bias = call_bias_report()
    if bias is None:
        st.info(
            "Bias report not yet generated. Run "
            "`notebooks/04_bias_audit.ipynb` after training to produce it."
        )
    else:
        overall = bias["overall"]
        worst = bias["worst_gap"]
        col_o, col_w = st.columns([1, 1])
        with col_o:
            st.metric("Overall MAE",
                      f"{overall['mae']:.2f} years",
                      help=f"N = {overall['n_samples']:,}")
        with col_w:
            if worst.get("worst_group"):
                st.metric(
                    "Worst-group gap",
                    f"+{worst['gap_years']:.2f} years",
                    help=f"{worst['worst_group']} — "
                         f"MAE {worst['worst_mae']:.2f}",
                )
            else:
                st.write("No eligible groups (need N≥30 per group)")

        # render every group as one row so the recruiter sees the full
        # bias picture, not just the headline numbers
        with st.expander("Per-group breakdown"):
            for grp in bias["groups"]:
                st.write(
                    f"**{grp['group_type']}: {grp['group_name']}** — "
                    f"MAE {grp['mae']:.2f} · RMSE {grp['rmse']:.2f} · "
                    f"N = {grp['n_samples']:,}"
                )


if __name__ == "__main__":
    main()
