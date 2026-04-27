# Architecture

Long-form architecture explanation for FairAge. The README has the headline diagram; this file is the interview crib sheet — every layer, every boundary, every reason.

## The three machines

FairAge involves three physical machines, each with a single role.

| Machine | Role | What runs there | What does NOT run there |
|---|---|---|---|
| Windows laptop | Code editing, git, light testing | VS Code, Git Bash, the venv, pytest | training, real inference, public traffic |
| Surrey HPC (Eureka2) | GPU training only | SLURM jobs, conda env, W&B logging | API serving, demo UI, public traffic |
| Hetzner server | Production serving | Docker, Nginx, fairage-api, fairage-demo | training (no GPU), code editing |

Code edits never happen on the HPC or the Hetzner box. Both pull from GitHub, never push. This keeps git history clean and prevents the classic "I edited the file on the server and now the local copy is out of date" problem.

## Request lifecycle — what happens when a user uploads a face

Step-by-step walk through one request. Numbered so any step can be audited.

```
1. user opens http://<server>/fairage-demo/ in a browser
2. nginx on port 80 receives the GET, matches the /fairage-demo/ location block
3. nginx forwards to 127.0.0.1:8503 (Streamlit container)
4. Streamlit returns the upload-form HTML
5. user picks an image, clicks upload
6. Streamlit POSTs the bytes to FAIRAGE_API_URL=http://fairage-api:8003/estimate-age
   (over the internal docker network — never leaves the host)
7. FastAPI receives the upload in src/api/main.py
8. Pydantic validates the request shape (file present, content type, size)
9. main.py calls engine.predict(image_bytes) where engine is the eager-loaded
   InferenceEngine on app.state.engine
10. engine._preprocess_image runs PIL decode -> resize -> imagenet normalise
    -> CHW -> (1, 3, 224, 224) float32 numpy array
11. engine._pad_session.run(...) — ONNX Runtime, CPU provider, returns logits
12. softmax + index 1 = P(attack)
13. if P(attack) > 0.5: return PredictionResult(is_spoof=True, ...) — short-circuit
14. otherwise engine._age_session.run(...) returns ordinal logits
15. _decode_age applies sigmoid, sums (>0.5) for predicted age, computes confidence
16. PredictionResult(is_spoof=False, estimated_age=..., confidence=...) returned
17. main.py wraps result in EstimateAgeResponse Pydantic model -> JSON
18. FastAPI returns 200 to Streamlit
19. Streamlit renders the metrics in the right column of the demo page
20. websocket update pushes the rendered DOM back through nginx to the user's browser
```

Total wall-clock time on the Hetzner box (4 vCPU, no GPU): ~150 ms p99 for steps 7–18. Steps 1–6 and 19–20 are network + browser, not measured here.

## Training pipeline — what happens before any of that

Training runs once per model release, on Surrey HPC.

```
1. user pushes code from laptop to GitHub
2. user SSHes into HPC login node
3. user runs: cd ~/fairage && git pull origin main
4. user runs: sbatch slurm/train_age.sh
5. SLURM queues the job. Output goes to slurm-<jobid>.out and slurm-<jobid>.err
6. SLURM allocates a GPU node (a100 partition, 1 GPU, 32 GB RAM, 4 hours)
7. The job script:
   a. loads CUDA/12.2.2 and Anaconda3 modules
   b. activates the `fairage` conda env
   c. runs `python -m src.training.train_age`
8. train_age.py:
   a. seeds every random source for reproducibility
   b. builds train + val DataLoaders from UTKFaceDataset
   c. instantiates AgeEstimator (ResNet-50 + ordinal head)
   d. wraps in OrdinalRegressionLoss
   e. AdamW + CosineAnnealingLR
   f. for each epoch: train pass, eval pass (decoded MAE), W&B log,
      save best checkpoint by val MAE
9. best checkpoint -> artifacts/checkpoints/age_best.pt on the HPC home dir
10. user scp's the checkpoint back to the laptop
11. user runs `python -m src.deploy.export_onnx` on laptop -> .onnx
12. user runs `python -m src.deploy.quantize_onnx` -> _int8.onnx
13. user runs `python -m src.deploy.benchmark` -> latency JSON
14. user scp's the int8.onnx files to the Hetzner box
15. user runs docker compose restart fairage-api on Hetzner
```

Training is decoupled from serving on purpose. The Hetzner box never sees torch, never sees a GPU, never re-trains anything. It only loads two pre-quantized ONNX files at container startup.

## ONNX export pipeline

The conversion from PyTorch checkpoint to served INT8 ONNX is three scripts, each a single responsibility.

```
artifacts/checkpoints/age_best.pt    (PyTorch state dict, ~95 MB)
              │
              ▼
   src/deploy/export_onnx.py
     - rebuilds AgeEstimator
     - loads state_dict
     - torch.onnx.export with opset 17, dynamic batch axis
     - dynamo=False pinned via inspect.signature for cross-version stability
              │
              ▼
artifacts/exports/age_model.onnx     (ONNX float32, ~95 MB)
              │
              ▼
   src/deploy/quantize_onnx.py
     - onnxruntime.quantization.quantize_dynamic
     - QInt8 weight type
     - no calibration set needed (dynamic = activation quant at runtime)
              │
              ▼
artifacts/exports/age_model_int8.onnx (ONNX INT8, ~25 MB)
              │
              ▼
   src/deploy/benchmark.py
     - 100 forward passes, p50 + p95 + p99 latency
     - writes artifacts/benchmark_results.json
              │
              ▼
       scp to Hetzner
```

The PAD model follows the same pipeline with `src/deploy/export_pad_onnx.py` instead of `export_onnx.py`. Same quantize and benchmark scripts work for both.

## Deployment topology — Hetzner box

```
                    Public internet
                          │
                          ▼ port 80
              ┌──────────────────────┐
              │  Nginx (host)        │
              │  - serves other apps │
              │  - /fairage-api/*    │──► 127.0.0.1:8003
              │  - /fairage-demo/*   │──► 127.0.0.1:8503
              └──────────────────────┘
                          │
            ┌─────────────┴──────────────┐
            ▼                            ▼
   ┌─────────────────┐          ┌─────────────────┐
   │ fairage-api     │          │ fairage-demo    │
   │  (Docker)       │◄────────►│  (Docker)       │
   │                 │ internal │                 │
   │  127.0.0.1:8003 │ network  │  127.0.0.1:8503 │
   │  FastAPI        │          │  Streamlit      │
   │  ONNX Runtime   │          │  no models      │
   │  read-only mnt: │          │                 │
   │  ../artifacts/  │          │                 │
   └─────────────────┘          └─────────────────┘
```

Key constraints:

- **Both containers bind 127.0.0.1, never 0.0.0.0.** A direct attack on port 8003 or 8503 from the internet cannot land. The only public surface is Nginx on port 80.
- **The API container is read-only on the model files.** Even a code-execution exploit inside the container cannot tamper with the served model.
- **`tmpfs /tmp:noexec,nosuid,size=64M`.** Blocks the `/tmp/<binary>` execution drop pattern that hit `nhs-db` in April 2026.
- **`restart: on-failure:3`.** If the container crashes 3 times, Docker stops restarting it. `always` would force-multiply a malware payload (kill -> restart -> re-exec).
- **Resource limits.** API gets 1 CPU, 1 GB RAM. Demo gets 0.5 CPU, 512 MB. A compromised container cannot eat the host.

## Security model

The threat model FairAge defends against:

| Threat | Mitigation |
|---|---|
| Direct port attack from internet | 127.0.0.1 binding — port 8003/8503 not reachable externally |
| Malware execution inside container | tmpfs /tmp noexec, read-only model mount, non-root user |
| Force-multiplied malware via restart | restart: on-failure:3, NOT always |
| Resource exhaustion attack | CPU + memory limits per container |
| Model tampering | Read-only mount of artifacts/exports/ |
| Image bytes leaking via logs | API logs status + latency only, not request bodies |
| Image bytes leaking via disk | Stateless handlers, tmpfs noexec, no disk writes during request |
| Credentials in git | .env in .gitignore, .env.example with placeholders only, pre-push grep |
| Outbound exfiltration | No outbound HTTP at request time; only manual apt updates |

What FairAge does NOT defend against:

- Compromise of the Nginx host itself (out of scope; standard server hardening applies)
- Compromise of the developer's laptop (out of scope; standard endpoint security)
- Compromise of GitHub credentials (out of scope; rely on GitHub's 2FA)
- Side-channel timing attacks on the inference endpoint (acknowledged, not mitigated)

## Performance budget

The CV claim is "sub-200ms p99 inference latency on a 4-vCPU CPU". Where the 200ms goes:

| Step | Time budget | Actual (typical) |
|---|---|---|
| FastAPI request parse + Pydantic validate | 5 ms | ~3 ms |
| PIL decode + resize | 15 ms | ~10 ms |
| numpy preprocessing (normalise, transpose) | 5 ms | ~2 ms |
| PAD ONNX forward pass | 30 ms | ~20 ms |
| Age ONNX forward pass | 80 ms | ~60 ms |
| Decode logits + build response | 5 ms | ~3 ms |
| FastAPI serialise + return | 10 ms | ~5 ms |
| **Total** | **150 ms** | **~100 ms** |

The 200ms claim is p99, not p50. The headroom (50ms) covers cold cache, thread contention, and the tail of the latency distribution.

The `/explain` endpoint is opt-in and runs ~144 forward passes (one per occlusion grid cell). Latency: 1–2 seconds. This is acceptable because the user clicks a button to invoke it; it is not on the hot path.

## What this architecture optimises for

In priority order:

1. **Recruiter trust.** Every choice — eager loading, quantized ONNX, bias audit endpoint, saliency endpoint — looks like a production engineer made it.
2. **Reproducibility.** Every test runs on synthetic data in `tmp_path`. Every random source is seeded. Training is one SLURM script, not a notebook.
3. **Auditability.** Bias report is a JSON file in git. Decisions are recorded in `docs/DECISIONS.md`. Compliance gaps are explicit in `COMPLIANCE.md`.
4. **Security defence in depth.** 127.0.0.1 + tmpfs + read-only + restart limits + resource caps. Each one alone would not stop the NHS incident; together they would have caught it at three different layers.
5. **Operational simplicity.** Two containers, one Nginx config, one docker-compose. Update flow is `git pull && docker compose up -d --build`. Anything more complex would be the wrong solution for a single-recruiter demo.

## What this architecture explicitly does not optimise for

- **Latency below 50ms.** The Hetzner box is a single 4-vCPU machine. Sub-50ms would need a GPU, batching, or a faster runtime. None of those are needed for the recruiter demo.
- **Horizontal scaling.** One API container, one demo container. This will not handle thousands of concurrent users. It does not need to.
- **Multi-region failover.** One server. If Hetzner goes down, the demo goes down. Acceptable for a portfolio piece.
- **Live retraining.** The model is static between releases. There is no online learning loop. Adding one would mean changing the threat model significantly and is out of scope.

This is the architecture for a working, honest, recruiter-credible reference implementation. It is not the architecture for a planet-scale deployment, and it does not pretend to be.
