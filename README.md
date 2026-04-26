# FairAge

Edge-deployed, bias-audited age estimation system with presentation attack detection.

> Stub README — full version arrives in Phase 10 with live demo URL, badges, architecture diagram, and key decisions.

## Status

Build in progress. Phase 1 — project skeleton complete.

## What this project will be

A facial age estimation model trained on UTKFace, audited for demographic bias, exported to a quantised TFLite model that runs on CPU under 200ms. Wrapped in a FastAPI service with a Streamlit demo. Includes a presentation attack detector to reject printed photos and replays.

## Stack at a glance

- PyTorch — training (ResNet-50 backbone, ordinal regression head)
- Surrey HPC + SLURM — GPU training
- ONNX + TFLite — edge model export
- FastAPI — inference API
- Streamlit — demo dashboard
- Docker + Nginx on Hetzner — deployment
- Weights & Biases — training run logging

## Run locally

Setup steps land in Phase 10. The `.env.example` file lists every variable the project will need.

## License

MIT — see `LICENSE`.
