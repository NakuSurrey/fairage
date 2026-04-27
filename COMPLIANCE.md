# Compliance

How FairAge aligns with the standards that govern age estimation systems in the UK and EU.

> **Honesty caveat:** ACCS clause numbers below are placeholders. The Age Check Certification Scheme keeps its registry behind login and the exact clause numbering changes between revisions. Before claiming alignment in a customer-facing setting, the latest published ACCS test specifications must be pulled from <https://accscheme.com> and the references in this document updated to match. The structural alignment described below is real; the citation accuracy is the part that needs verification.

This file is a working draft, not a certification. A real ACCS audit involves an accredited test lab, not a self-assessment.

## Scope of this document

Three standards are relevant to a deployed age estimation system:

| Standard | What it covers | FairAge section |
|---|---|---|
| ACCS 1:2020 — Technical Requirements for Age Estimation Technologies | Accuracy, age band performance, vulnerability to spoofing, transparency | Section 1 |
| ACCS 2:2021 — Technical Requirements for Data Protection and Privacy | Data minimisation, retention, on-device processing, third-party transfers | Section 2 |
| GDPR Article 22 — Automated individual decision-making | Right to human review, right to an explanation | Section 3 |

## 1. ACCS 1:2020 — Age Estimation Technical Requirements

### 1.1 Accuracy reporting

ACCS 1:2020 requires accuracy reporting at the population level and at age-band level.

FairAge produces a [bias audit](artifacts/bias_report.md) on every model release. The audit reports MAE for:

- Overall (single number)
- Per gender (male, female)
- Per ethnicity (white, black, asian, indian, others — the five groups labelled in UTKFace)
- Per age band — 0–12, 13–19, 20–35, 36–55, 56–100 (the bands in `src/config.py:AGE_BUCKETS`)
- Per intersectional group (gender × ethnicity)

The audit is regenerated automatically by `notebooks/04_bias_audit.ipynb` and served by the API at `/bias-report`. Accuracy claims in marketing copy must reference the version of the audit that backs them, not a separate hand-typed number.

### 1.2 Age verification at decision boundaries

For an age-gating product (the typical ACCS deployment), the regulator cares about decisions at thresholds — usually 18 or 25. A model that is accurate on average but unreliable at the threshold is not fit for purpose.

FairAge logs per-sample predictions during evaluation, so threshold accuracy can be derived offline by re-running the audit script on filtered slices. This is documented in `src/audit/bias_audit.py`. The current production model does not yet have a decision-boundary report; this is a known gap to close before shipping to a regulated deployment.

### 1.3 Presentation Attack Detection (Level 2 requirement)

ACCS 1:2020 Level 2 deployment requires the system to reject presentation attacks — printed photos, screen replays, masks. FairAge runs PAD before age estimation:

```
upload  →  PAD model  →  if P(attack) > 0.5  →  refuse, return is_spoof=true
                       └  otherwise           →  age model runs
```

The PAD model is trained on NUAA Imposter Database (print attacks). Stronger production deployments should retrain on CelebA-Spoof or Replay-Attack DB (replay + mask attacks); NUAA covers the print case end to end.

### 1.4 Transparency about limitations

Per ACCS 1:2020 transparency clauses, the system must publish its known limitations.

Known limitations of the current FairAge model:

- Trained only on UTKFace (~24,000 images). Performance on faces outside the UTKFace distribution (e.g. heavy makeup, partial occlusion, non-frontal angles) is not characterised.
- PAD covers print attacks only. Replay (screen-recorded video) and mask attacks are not in the training set.
- The model is calibrated for ages 0 to 100. Out-of-range inputs are clipped.
- Bias audit groups follow UTKFace's labelling — five ethnicity buckets, binary gender. This does not represent the full demographic landscape of any deployment population.

These limitations are documented in [`docs/DECISIONS.md`](docs/DECISIONS.md) alongside the rationale for each one.

## 2. ACCS 2:2021 — Data Protection

### 2.1 Data minimisation

The API takes one image, runs inference, returns a number. No image is written to disk on the server. No image is forwarded to a third-party API. No logs contain the image bytes.

The container runs with `tmpfs /tmp:noexec,nosuid,size=64M` — even if a process tried to write the upload to a temporary file, the tmpfs mount blocks execution and is wiped on container restart.

### 2.2 No image retention

The FastAPI handler reads the upload bytes into memory, hands them to `engine.predict()`, returns the JSON response, and the bytes go out of scope. Nothing persists between requests except the model sessions, which contain no user data.

The Streamlit demo container is similarly stateless — it forwards the upload to the API and renders the response.

### 2.3 No third-party data transfers

All inference runs locally on the Hetzner server. No outbound HTTP calls during a prediction request. The only outbound traffic from the server is for OS package updates (manual, monthly).

This makes FairAge compatible with a "data does not leave the controller" architecture, which simplifies the ACCS 2:2021 alignment substantially.

### 2.4 Logging policy

The API logs:

- Request method, path, status code, latency
- Engine startup and shutdown events
- Errors with stack traces (no upload bytes)

The API does **not** log:

- Upload bytes
- Predicted ages tied to any session identifier
- Any user-identifying metadata

Log retention is governed by Docker's default behaviour and the container restart policy. Logs go to stdout, are captured by Docker, and are reviewed monthly per the security hardening checklist in [`deploy/README.md`](deploy/README.md).

## 3. GDPR Article 22 — Automated Decision-Making

Article 22 gives the data subject the right to:

### 3.1 Not be subject to a fully automated decision with legal or similarly significant effect

FairAge does not, on its own, take a decision with legal effect. The system returns an estimated age and a confidence score. It does not gate access, refuse service, or decide eligibility for anything.

The decision (e.g. "is this person old enough to buy alcohol") is taken by the downstream consumer of the API — typically a point-of-sale terminal or an online age gate. That downstream system is the controller for Article 22 purposes; FairAge is a processor that supplies one input (the age estimate) into that decision.

The README and API docs are explicit about this split. Marketing copy must not present FairAge as making a decision — only as supplying a numeric estimate.

### 3.2 Right to human review

Two endpoints support human review of any prediction:

- **`/bias-report`** — exposes the per-group MAE table so a reviewer can see how the model performs on the demographic group of the person being evaluated. If the worst-group gap is large for the relevant group, a human review is warranted before any decision is taken.
- **`/explain`** — returns an occlusion saliency heatmap showing which pixel regions drove the age prediction. A human reviewer can see whether the model focused on face features (legitimate) or on background, lighting, or occlusion artifacts (suspicious).

Both endpoints return data, not opinions. The reviewer is the human, not the system.

### 3.3 Right to an explanation

The occlusion saliency map is the explanation surface. For any prediction the user disputes, the same image can be sent to `/explain` to produce a heatmap of what influenced the result.

The choice of occlusion (rather than gradient-based methods like Integrated Gradients) is documented in [`docs/DECISIONS.md`](docs/DECISIONS.md). The trade-off: occlusion is slower (1–2 seconds per explanation on a 4-vCPU CPU) but works with the deployed INT8 ONNX model that does not expose gradients. Speed is acceptable for an opt-in explainability endpoint that is not on the hot path.

The explanation does not reveal training-data identities, model weights, or any internal state — only the spatial influence of the input image on the output age.

## Compliance review checklist (before any regulated deployment)

Run through this list before claiming ACCS or GDPR alignment in a customer-facing setting.

- [ ] Pull the latest ACCS 1:2020 and ACCS 2:2021 specifications from <https://accscheme.com> and replace the placeholder clause references in this file with real numbers.
- [ ] Produce a decision-boundary accuracy report for the actual deployment threshold (18, 21, 25 — whichever applies).
- [ ] Run the bias audit on a held-out test set that matches the deployment population, not the UTKFace test split. UTKFace skews towards certain demographics; an external dataset is needed for an honest fairness claim.
- [ ] Upgrade the PAD model to a dataset that includes replay and mask attacks (CelebA-Spoof or Replay-Attack DB) before any Level 2 deployment.
- [ ] Have an accredited test lab perform the actual ACCS audit. Self-assessment is not certification.
- [ ] Add a Data Protection Impact Assessment (DPIA) covering the deployment context, retention policy, and downstream decision flow. ACCS 2:2021 expects this for any production rollout.
- [ ] Add a fallback path: if PAD or age model is unavailable, the downstream decision system must have a defined human-only fallback. Do not ship "fail-closed" in a way that creates a denial-of-service for legitimate users.
- [ ] Confirm the deployment region. GDPR applies to UK and EU users. ACCS is a UK scheme. Other jurisdictions (e.g. California's CCPA, Texas's HB 4) have separate rules that this document does not cover.

This checklist is the gap between "structurally aligned" (where FairAge is today) and "ready to claim certification" (a separate engagement with a test lab).
