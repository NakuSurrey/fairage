# FairAge

Bias-audited age estimation with presentation attack detection. Trained on UTKFace, served as a quantized ONNX model behind a FastAPI service, deployed live on Hetzner.

[![Python](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.4-EE4C2C.svg)](https://pytorch.org)
[![ONNX Runtime](https://img.shields.io/badge/ONNX_Runtime-1.19-005CED.svg)](https://onnxruntime.ai)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688.svg)](https://fastapi.tiangolo.com)
[![Tests](https://img.shields.io/badge/tests-118_passing-brightgreen.svg)](#testing)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

> **Live demo:** http://46.225.208.197/fairage-demo/
> **API docs:** http://46.225.208.197/fairage-api/docs
> **Health:** http://46.225.208.197/fairage-api/health
>
> _Hosted on a 4 vCPU Hetzner box with stub-trained models — predictions are not real ages yet. The serving stack, ONNX pipeline, and bias-audit endpoints all work end-to-end. Real models land after the Surrey HPC training run._

## What it does

Upload a face image. The system returns three things:

- **Estimated age** in years, with a confidence score
- **Presentation Attack score** — probability the image is a printed photo or screen replay rather than a live face
- **Saliency heatmap** showing which pixel regions most influenced the age prediction

A `/bias-report` endpoint exposes the precomputed per-group MAE table — overall vs split by gender, ethnicity, age band, and intersectional cells. The fairness numbers are visible to anyone running the service, not buried in a notebook.

## Why it exists

Age estimation systems are deployed in retail age-gating, online safety, and identity verification. Most published models report a single MAE number that hides large per-group gaps. FairAge ships the bias audit as a first-class artifact alongside the model, treats spoof attempts as a distinct adversarial input class, and runs end to end on a 4-vCPU server with sub-200ms inference latency. The goal: a working reference for what an age estimation product should look like before it touches a real user.

## Architecture

```
┌──────────────────┐                      ┌────────────────────────┐
│  Streamlit demo  │ ── HTTP/JSON ──────► │  FastAPI service       │
│  (image upload)  │                      │   ▸ /estimate-age      │
└──────────────────┘                      │   ▸ /explain           │
                                          │   ▸ /bias-report       │
                                          │   ▸ /health            │
                                          └──────────┬─────────────┘
                                                     │  in-process
                                                     ▼
                                          ┌────────────────────────┐
                                          │ ONNX Runtime sessions  │
                                          │  (eager-loaded on      │
                                          │   container startup)   │
                                          │                        │
                                          │  PAD model  ──┐        │
                                          │               ▼        │
                                          │  spoof? ─ yes ─ refuse │
                                          │   │                    │
                                          │   no                   │
                                          │   ▼                    │
                                          │  Age model  → logits   │
                                          │            → decoded   │
                                          └────────────────────────┘

Build-time pipeline:
  UTKFace + NUAA  ──►  Surrey HPC GPU training  ──►  ONNX export
                                                          │
                                                          ▼
                                                  INT8 quantization
                                                          │
                                                          ▼
                                                  Docker image  ──►  Hetzner
```

## What is in the repo

```
src/
├── data/         — UTKFace + NUAA datasets, transforms
├── models/       — ResNet-50 + ordinal head (age), small CNN (PAD), ordinal loss
├── training/     — train_age.py, train_pad.py, evaluate.py
├── audit/        — per-group MAE bias audit, JSON + Markdown reports
├── deploy/       — PyTorch -> ONNX export, INT8 quantization, CPU benchmark
└── api/          — FastAPI app, ONNX inference engine, schemas

streamlit_app/    — Streamlit demo UI (talks to API only, loads no models)
slurm/            — SLURM batch scripts for Surrey HPC training jobs
deploy/           — docker-compose, nginx config, Hetzner deploy guide
notebooks/        — EDA, smoke baseline, bias audit, PAD EDA
tests/            — 118 unit + integration tests across every module
artifacts/        — bias_report.json, benchmark_results.json (the proofs)
docs/             — ARCHITECTURE.md, DECISIONS.md, ACCS_NOTES.md
```

## Tech stack

| Layer | Choice | Why this over alternatives |
|---|---|---|
| Training | PyTorch 2.4 + AdamW + CosineAnnealingLR | Standard production stack. Surrey HPC modules support it. |
| Backbone | ResNet-50 pretrained on ImageNet | Strong visual prior; fine-tuning beats from-scratch on ~24k samples |
| Loss head | Ordinal regression (Niu et al. 2016) | 100 binary "age > k?" outputs give richer gradient than plain regression |
| PAD model | Small custom CNN (~500k params) | NUAA is small; ResNet-50 would overfit. Matches the data scale. |
| Bias metric | MAE per group + worst-group gap (N≥30) | Industry standard fairness audit; small-group floor prevents noise inflation |
| Export | PyTorch → ONNX → ONNX Runtime INT8 | Microsoft's production pattern; one runtime works on every server platform |
| Quantization | Dynamic INT8 | No calibration set needed; 4× smaller, 1–3× faster on CPU |
| Serving | FastAPI + Uvicorn, 1 worker | Eager model load on startup → zero cold-start at user time |
| Deployment | Docker Compose + Nginx reverse proxy | 127.0.0.1 binding for defense in depth |
| Demo | Streamlit | One-file Python UI; no JS toolchain |

## Performance

- **Overall test MAE:** _stub models in production right now — real MAE published after the Surrey HPC training run_
- **Inference latency:** sub-200ms p99 on a 4-vCPU CPU (int8 quantized model)
- **Model size:** ~25 MB int8 (down from ~95 MB float32)
- **Bias audit:** see [artifacts/bias_report.md](artifacts/bias_report.md) for the full per-group breakdown

The exact numbers are regenerated by running `python -m src.deploy.benchmark` and `notebooks/04_bias_audit.ipynb`. Both write JSON to `artifacts/`, both are tracked in git.

## How to run locally

```bash
# 1. set up the environment
git clone https://github.com/NakuSurrey/fairage.git
cd fairage
python -m venv venv
source venv/Scripts/activate     # on Windows; "source venv/bin/activate" on macOS/Linux
pip install -r requirements.txt
pip install -r requirements-train.txt

# 2. download the datasets — see data/README.md for sources and folder layout

# 3. run the test suite — should be 118 tests passing
pytest tests/ -v

# 4. (optional) run training. needs a GPU. on a laptop, use a tiny epoch budget.
python -m src.training.train_age --epochs 1 --batch-size 16 --no-pretrained

# 5. (optional) export to ONNX, quantize, benchmark
python -m src.deploy.export_onnx
python -m src.deploy.quantize_onnx
python -m src.deploy.benchmark

# 6. run the API locally
uvicorn src.api.main:app --reload --port 8003

# 7. run the Streamlit demo (in another terminal)
streamlit run streamlit_app/app.py
```

The full Surrey HPC training flow is documented in [`slurm/`](slurm/) and uses SLURM batch jobs.
The full Hetzner deploy flow is documented in [`deploy/README.md`](deploy/README.md).

## Testing

118 unit + integration tests, run on CPU, no GPU needed, no real datasets needed. Synthetic data generated in `tmp_path` keeps every test reproducible on a clean machine.

```bash
pytest tests/ -v
```

Coverage:

- Dataset parsers + filename validation (UTKFace, NUAA)
- Ordinal regression encoder, decoder, loss math + gradient flow
- Age estimator forward pass, head shape, ImageNet weight handling
- PAD detector forward pass, softmax probabilities, gradient flow
- Stratified train/val/test splitting, class-weighted loss helpers, HTER metric
- Bias audit grouping, worst-group gap, end-to-end JSON + Markdown writers
- ONNX export, INT8 quantization, CPU latency benchmark percentiles
- FastAPI endpoints — health, estimate-age, explain, bias-report
- Streamlit helpers — saliency overlay rendering, API client wrappers

## Key engineering decisions

A few that affected real architecture choices, recorded in [`docs/DECISIONS.md`](docs/DECISIONS.md):

- **Ordinal regression over plain regression** — every age becomes 100 binary yes/no questions, each gets its own gradient signal. Standard approach in modern age estimation (Niu et al. 2016).
- **ONNX Runtime over TFLite** — TFLite is for phones, ONNX Runtime is for servers. The deployment target is a 4-vCPU Hetzner box, not a phone. ONNX Runtime is also the standard production runtime at Microsoft.
- **Eager model loading on startup** — every user request hits a pre-warmed ONNX session. No cold-start penalty at user time. Container takes 3 seconds longer to come up at deploy time, which happens once.
- **Occlusion saliency over Captum/IntegratedGradients** — the served model is INT8 ONNX, no gradients available. Occlusion is forward-pass only, works with any backend, matches Microsoft's interpretability approach for non-PyTorch deployments.
- **127.0.0.1 port binding behind Nginx** — defense in depth. The previous host on this server had a cryptominer incident from a `0.0.0.0:5432` postgres binding. FairAge does not repeat that mistake.
- **NUAA over CelebA-Spoof for PAD** — CelebA-Spoof is 600 GB. NUAA is 600 MB. Pipeline ships end to end on the smaller dataset; CelebA-Spoof can be swapped in later without code changes.

## Compliance

[`COMPLIANCE.md`](COMPLIANCE.md) covers alignment with ACCS 1:2020 (age estimation technologies), ACCS 2:2021 (data protection), and GDPR Article 22 (explainability for automated decisions). Bias audit, on-device-only inference (no image storage), and the saliency endpoint feed into those.

## What I learned

- Ordinal regression turns a hard regression problem into 100 easier binary problems, and the gradient signal is genuinely better
- INT8 quantization gives a 4× model size reduction with a sub-1% accuracy hit on a well-trained CNN — almost always worth it for CPU serving
- Eager model loading is one of those textbook patterns that costs almost nothing to implement and saves the first user a real, measurable wait
- Occlusion saliency is slower than gradient-based methods but works with any model format. Trade-off worth taking when the model is already deployed.
- A 127.0.0.1 port binding is the cheapest piece of security hardening that exists. Pair it with a reverse proxy and the public attack surface drops to one process you actually want to harden.

## License

MIT — see [LICENSE](LICENSE).
