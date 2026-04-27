# Decisions

Canonical record of every meaningful technical decision made during the build. Two kinds of entries: self-decisions (taken silently when the project goal or reference files clearly answered the question) and Decision Points (where Claude stopped and asked, because no clear answer existed in context).

This file is the interview crib. For every decision below the answer to "why this and not the alternative" is one click away.

## Self-decisions (Rule B7 Step 4)

These decisions were taken without stopping the build, because the project goal or a reference file clearly pointed to one option. Each one was logged in the chat with the format `Chose [X] over [Y] — [reason]. Say 'change' if you want different.`

### Phase 4

**1. nn.BCEWithLogitsLoss over manual sigmoid + BCE**
Same math, but `BCEWithLogitsLoss` is numerically stable via internal log-sum-exp. The manual version is fine for a textbook example; the stable version is what production code uses.

**2. Linear(2048→512)→ReLU→Dropout→Linear(512→100) over a single Linear(2048→100)**
Extra non-linearity in the head helps it learn richer threshold relationships across the 100 ordinal positions. Matches the head design in Niu et al. 2016, the paper FairAge cites for ordinal regression.

### Phase 5

**3. 30-sample minimum for the worst-group gap headline**
Groups smaller than 30 have unstable MAEs that would mislead the report — a single misprediction in a 5-sample group shifts MAE by 0.2 years. 30 is the standard small-sample floor in fairness audits.

### Phase 6

**4. Flexible folder-name resolution for the PAD dataset (`real/` or `ClientFace/`, `attack/` or `ImposterFace/`)**
The project planning file uses `real/` and `attack/`. NUAA's native folders are `ClientFace/` and `ImposterFace/`. Supporting both means the user does not have to rename folders after extracting the dataset.

**5. Small custom CNN over fine-tuned ResNet-50 for PAD**
NUAA has only ~12k images. A 23M-parameter ResNet-50 backbone would overfit. A ~500k-parameter custom CNN matches the data scale. It is also a stronger interview talking point — "designed for the dataset" beats "ResNet again".

**6. HTER (half total error rate) over plain accuracy for the PAD checkpoint metric**
PAD datasets are usually class-imbalanced. A model that always predicts "real" can score 90% accuracy on a 90/10 dataset and be useless. HTER averages FAR (false accept rate) and FRR (false reject rate), so both error types count equally. Standard PAD evaluation metric.

### Phase 7

**7. Dynamic INT8 quantization over static**
Static quantization needs a calibration dataset to compute activation ranges. Dynamic computes them at runtime instead. Same accuracy hit (~1%) as static for a CNN, simpler pipeline, no calibration set to ship. Matches what most production deployments actually do.

**8. Cross-version ONNX exporter pin via `dynamo=False` + `inspect.signature` check**
torch 2.4 (the user's version) does not have a `dynamo` kwarg in `torch.onnx.export`. torch 2.5+ does. Pinning the legacy exporter via `inspect.signature` keeps the code working on both versions. Not announced as a self-decision in chat — applied silently as part of a bug fix — but logged here for completeness.

### Phase 8

**9. Pure-numpy preprocessing inside the API**
Importing torchvision into the API container would pull in torch (~1 GB). The eval transform is six lines of numpy: PIL decode, resize, divide by 255, subtract ImageNet mean, divide by ImageNet std, transpose HWC→CHW. The math is identical to the training-time eval transform.

**10. FastAPI's `lifespan` async context manager over `@app.on_event('startup')`**
`on_event` is deprecated as of FastAPI 0.93. `lifespan` is the modern pattern. A 2026 production codebase should not use deprecated startup hooks.

### Phase 9

**11. grid=12 saliency resolution (144 forward passes per explanation)**
Smaller grids (8x8 = 64 passes) give a coarse heatmap that does not align well with face features. Bigger grids (16x16 = 256 passes) are visibly slower without proportional quality gain. 12x12 lands in the 1–2 second wall-clock range on the Hetzner box, which is acceptable for an opt-in endpoint.

## Decision Points (Rule B7 Step 5)

These are the moments Claude stopped and asked the user, because the project goal and reference files did not point to one clear answer.

### Decision Point 1 — PAD dataset (Phase 6)

**The choice:** which dataset to train the PAD model on.

| Option | How it works | Advantage | Disadvantage |
|---|---|---|---|
| A. NUAA Imposter Database | ~12k images, print attacks only, free download | Fits on laptop, fast to iterate, no institutional gate | Dated (2010), only print attacks |
| B. CelebA-Spoof | ~600 GB, multiple attack types, free | Strongest possible PAD claim | 600 GB needs HPC scratch space, blows out the timeline |
| C. Replay-Attack DB (Idiap) | Video-based, gold standard | Strongest spoofing benchmark | Requires institutional license — could take weeks, may be denied |

**My recommendation:** A. The project goal is "build a working version of ITL's product". A working PAD module on NUAA proves the architecture; the bigger datasets only matter if PAD accuracy is the headline of the interview, which it is not. NUAA finishes; B and C risk not finishing.

**User's choice:** A.

**Final pick:** NUAA Imposter Database.

### Decision Point 2 — ONNX → TFLite vs ONNX Runtime quantized (Phase 7)

**The choice:** which conversion path to use for edge deployment.

| Option | How it works | Advantage | Disadvantage |
|---|---|---|---|
| A. PyTorch → ONNX → TensorFlow → TFLite | onnx-tf converts, then `tf.lite.TFLiteConverter` | Native TFLite file, smallest possible model | onnx-tf is poorly maintained, often breaks on new ONNX opsets, needs full TF install |
| B. PyTorch → ONNX → ONNX Runtime quantized | `onnxruntime.quantization.quantize_dynamic` | Stable tooling, cross-platform, no TF needed | Not a `.tflite` file — but the goal is "fast CPU inference under 200ms", which ONNX Runtime quantized meets just as well |

**My recommendation:** B. The deeper goal is "edge-ready, fast CPU inference, small model", not literally a `.tflite` file. ONNX Runtime is the production-standard inference engine — Microsoft built it, it powers Bing, Office, and Azure ML. The README will say "ONNX Runtime int8 quantized" instead of "TFLite int8" — same meaning, more honest.

**User's choice:** B (with the recruiter framing in mind).

**Final pick:** ONNX Runtime INT8 quantized.

### Decision Point 3 — eager vs lazy model loading (Phase 8)

**The choice:** when to load the ONNX models into memory.

| Option | How it works | Advantage | Disadvantage |
|---|---|---|---|
| A. Eager loading on startup | Both ONNX models load in a FastAPI lifespan handler | Zero latency on the first request, predictable memory use | Container takes ~3 seconds to become ready |
| B. Lazy loading on first request | Model loads when the endpoint is first hit | Faster container startup | First request gets a 1–2 second cold-start penalty |

**My recommendation:** A. The project goal is "production-ready API a recruiter can hit and see fast results". Eager load means every user, including the recruiter on first click, sees a fast response. 30 MB of int8 weights in memory is nothing on a Hetzner box. The 3-second container startup happens once at deploy time, not at user time. This is what every well-engineered ML serving setup does.

**User's choice:** A (eager loading).

**Final pick:** Eager loading on FastAPI lifespan startup.

### Decision Point 4 — saliency approach (Phase 9)

**The choice:** how to implement the explainability map.

| Option | How it works | Advantage | Disadvantage |
|---|---|---|---|
| A. Captum + PyTorch in Streamlit container | Streamlit imports torch and the AgeEstimator, runs IntegratedGradients | Highest-quality explanations | Blows out the Streamlit Docker image to ~2 GB, requires shipping the .pt checkpoint |
| B. Skip saliency entirely | Drop the feature | Simplest possible demo | Loses the explainability claim that supports the GDPR Article 22 alignment |
| C. Occlusion saliency via the API | New `/explain` endpoint, slides a grey patch over a grid, measures age delta | Backend-agnostic, works with INT8 ONNX directly, no torch in the demo container | Slower than gradient-based (1–2 seconds per explanation) |

**My recommendation:** C. The served model is INT8 ONNX — ONNX Runtime does not give gradients, so Captum cannot attach to it without re-loading the float PyTorch model. Occlusion is forward-pass only, works with any backend, matches the production interpretability pattern Microsoft uses for non-PyTorch deployments.

**User's choice:** C.

**Final pick:** Occlusion saliency via a new `/explain` endpoint.

### Decision Point 5 — TLS / domain strategy (Phase 10)

**The choice:** how to expose the live demo.

| Option | How it works | Advantage | Disadvantage |
|---|---|---|---|
| A. Path-based on existing Hetzner host, HTTP only | `http://<server>/fairage-api/`, `http://<server>/fairage-demo/` via existing Nginx | Ships fastest, zero new infrastructure | Browser shows "Not secure" |
| B. Sub-path on existing Hetzner host, HTTPS via Caddy or Let's Encrypt | Same path structure as A, plus auto-TLS | Clean HTTPS, polished | Needs a domain pointed at the Hetzner IP — free options look informal |
| C. Buy a domain + HTTPS | Custom domain like `fairage.dev` + Let's Encrypt | Looks fully professional | £10–15/yr cost, 1–2 hour setup |

**Initial recommendation:** C. Recruiters notice the "Not secure" badge.

**Re-decision after planning doc consultation:** A. The planning file (`FAIRAGE_PROJECT_SETUP_REFERENCE.md` Section 12) explicitly locks path-based routing through the existing Nginx as the deployment plan. HTTPS is listed as "free upgrade later" in that file. Reverting to A respects the locked plan; HTTPS can be added without code changes whenever a domain is acquired.

**User's choice:** A (after the re-decision was explained).

**Final pick:** Path-based HTTP on existing Hetzner Nginx, HTTPS as a documented upgrade path.

## Decision pattern observed across the build

The user defers technical-detail choices to Claude but applies one consistent meta-rule: **best for CV / recruiter / standard practice**. Every Decision Point above was effectively re-evaluated through that lens once the rule became clear. Subsequent self-decisions used the same lens internally — for example, Decision 9 (pure-numpy preprocessing) was framed as "smaller production image looks more professional", and Decision 10 (FastAPI lifespan) was framed as "the deprecated decorator looks dated".

This is not a hidden bias. It is the explicit project goal: build something that signals production-engineer credibility to a hiring manager who has 30 seconds to scan the repo.
