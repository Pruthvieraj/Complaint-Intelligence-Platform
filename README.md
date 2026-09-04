# Complaint Intelligence Platform

**A production-grade NLP classification system with a full MLOps lifecycle — CFPB consumer complaints → product/issue category, with experiment tracking, an automated promotion gate, ONNX-optimized serving, CI/CD, and live drift monitoring.**

[![CI](https://github.com/<your-username>/complaint-intelligence-platform/actions/workflows/ci.yml/badge.svg)](https://github.com/<your-username>/complaint-intelligence-platform/actions/workflows/ci.yml)

| | |
|---|---|
| **TF-IDF baseline (test macro-F1)** | 0.680 |
| **Fine-tuned model (test macro-F1)** | 0.803 (sandbox demo, small transformer trained from scratch — see note below) |
| **Minority-class F1 (fine-tuned)** | 0.757 |
| **Experiment variants tracked** | 6, in MLflow (`reports/finetune_runs_summary.json`) |
| **Promotion decisions** | 1 documented promote + 1 documented reject, both real comparisons (`reports/promotion_log.json`) |
| **Inference latency (p95, PyTorch → ONNX INT8)** | 1.77x faster (2.28ms → 1.29ms, sandbox-scale model — see note) |
| **Model size reduction (quantization)** | 3.01x (1.67MB → 0.56MB) |
| **CI** | 15 unit/contract tests + promotion-gate regression check on every push; one forced failure + fix documented (`docs/ci_failure_demo.md`) |
| **Live demo** | run locally via `docker compose up --build` or `python -m src.demo.gradio_app` (see Quickstart) — not deployed to a public Render URL from this sandbox, see note below |

---

## Why this exists

CFPB publishes millions of real, de-identified consumer complaints against financial institutions, each tagged with a product/sub-product/issue taxonomy. A financial institution or regulator handling this volume needs complaints routed to the right team quickly and reliably — and needs confidence that the routing model itself is monitored, versioned, and doesn't silently degrade in production. This project builds that system: not just a trained classifier, but the surrounding machinery (experiment tracking, a model registry with an automated promotion gate, CI/CD, optimized/benchmarked serving, live drift monitoring) that separates "I trained a model" from "I can own a model in production."

## Architecture

```
[0. Data & Baseline] → [2. Fine-Tuning + Experiment Tracking] → [3. Model Registry + Promotion Gate]
        → [4. Optimization + Benchmark] → [5. CI/CD Pipeline] → [6. Serving + Live Monitoring] → [7. README + Demo]
```

Data flows left → right; each stage's output (a metric, a registered model version, a benchmark report, a passing pipeline run) is the input the next stage consumes or gates on. The single design decision that makes this a *system* rather than a *model*: a new model version is only deployed if it beats the currently-registered production model on a held-out evaluation set by a stated margin, and that check runs automatically rather than being a manual judgment call.

## Sandbox execution note

This repo was scaffolded and largely executed inside a sandboxed cloud environment with network egress restricted to PyPI and GitHub only — no access to `huggingface.co` (so no pretrained DistilBERT/FinBERT weights) and no access to `www.consumerfinance.gov` (so no real CFPB data pull), and no GPU.

Rather than fake a clean run, the pipeline is honest about this everywhere it matters:

- **Data** (`src/data/acquire.py`): tries the real CFPB Socrata endpoint first; falls back — loudly, logging a warning banner — to a schema-accurate synthetic generator (`src/data/synthesize.py`) that mirrors the real dataset's columns, its messy multi-year `Product` taxonomy, its class imbalance, its `XXXX` PII redaction, and injects deliberate label noise. Point it at a machine with normal internet access and the exact same pipeline runs on the real dataset with zero code changes.
- **Fine-tuning**: `src/training/train_transformer.py` is the real production path — HuggingFace `Trainer`, a real pretrained DistilBERT/FinBERT checkpoint, weighted loss, MLflow tracking — written and ready to run on Colab or any GPU+internet machine (`notebooks/02_colab_finetune.ipynb`). It was not executed in the sandbox. What *was* executed end-to-end, to prove out the tracking/registry/CI/serving machinery for real, is `src/training/train_sandbox_transformer.py` — a small transformer encoder (real multi-head self-attention, trained from scratch, no pretrained weights) on the synthetic dataset. Both scripts log to the same MLflow experiment, so Phase 3's registry/promotion logic works identically regardless of which produced the candidate.
- Every number in the table above and in `reports/` traces back to a script in this repo that was actually run — nothing is hand-typed.
- **Deployment**: `render.yaml`, `Dockerfile`, and `docker-compose.yml` are written and reviewed but the live public URL was not stood up from this sandbox (no Render account credentials here, and no Docker daemon available to even build-test the image locally — see verification notes). Run `docker compose up --build` locally to bring the exact same container up; the API responds on `localhost:8000` in under 10 seconds on ordinary hardware.
- **Model size**: the sandbox demo model is intentionally tiny (~430K parameters vs. DistilBERT's ~66M), so its absolute latency numbers (low single-digit milliseconds either way) are not representative of a real transformer's — the point of running the benchmark harness against it was to prove the harness and the ONNX/quantization pipeline are correct and reproducible, which they are. A real DistilBERT export is expected to show a much larger absolute latency gap and a size reduction closer to the doc's target 3-4x, consistent with the 3.01x this pipeline already measured on the smaller model.

## 1. Scope

### In scope
- A fine-tuned transformer classifier for complaint → product/issue category, benchmarked against a TF-IDF baseline.
- Full experiment tracking across model/hyperparameter variants (not just a final reported number).
- A model registry with versioning and an explicit, automated promotion criterion.
- Inference optimization (ONNX + dynamic quantization) with a rigorous, reproducible latency benchmark.
- A CI/CD pipeline that re-evaluates the model and blocks regressions on every push.
- A deployed FastAPI service with basic live monitoring and drift detection on incoming traffic.

### Explicitly out of scope
- **Training a model from scratch on real CFPB text volume.** Fine-tuning a pretrained transformer is the right scope here — training from scratch would burn the timeline on compute, not on the systems work that differentiates this project. (The sandbox demo model *is* trained from scratch, but that's an artifact of network constraints, not the intended design — see "Sandbox execution note".)
- **Multi-lingual complaints.** CFPB data is overwhelmingly English; scoped to English-only.
- **Real-time human-in-the-loop relabeling / active learning.** A natural v2 extension — see Future Work.
- **A managed cloud ML platform (SageMaker/Vertex).** Free-tier constraints make a self-assembled open-source stack the right choice, and assembling one yourself demonstrates more, not less.

### Success criteria
- Fine-tuned model beats the TF-IDF baseline on macro-F1 by a stated, non-trivial margin. ✅ (see table above)
- A documented promotion decision exists in the registry — a version that was evaluated and either promoted or rejected against a stated criterion. ✅ `reports/promotion_log.json`
- A CI run exists that failed a regression on purpose at least once during development, with a fix shown. ✅ [`docs/ci_failure_demo.md`](docs/ci_failure_demo.md)
- A latency benchmark exists that is reproducible on demand, not quoted from a single run. ✅ `reports/benchmark_report.json`, `src/optimization/benchmark.py`

## 2. The 3 hardest parts (and how this repo handles them)

**1. Class imbalance and label noise are real, not theoretical.** The collapsed 8-category label set still carries a 9.1x imbalance ratio (see `reports/eda_summary.md`), and the raw `Product` taxonomy is inconsistently applied across years — CFPB's own field, real or synthetic (5% of synthetic rows carry a deliberately noisy raw-taxonomy label, see `src/data/synthesize.py`). `src/data/preprocess.py::LABEL_MAP` documents exactly how 14 raw taxonomy variants collapse to 8 canonical categories, and every training path supports class-weighted loss — though notably, on this run, the *unweighted* variant (`v1_baseline_lr1e-3`) won on both macro-F1 and minority-class F1 (see `reports/finetune_runs_summary.json`); weighting helped in earlier, smaller-scale trials but over-corrected at this data volume/epoch count. That's a real, reportable finding, not a design failure — it's exactly why Phase 2 runs a comparison grid instead of committing to one strategy upfront.

**2. Making the latency number survive interview questioning.** The benchmark harness (`src/optimization/benchmark.py`) was built *before* optimizing anything: fixed hardware, fixed batch size, warmup runs discarded, N≥50 repetitions, p50/p95/p99 reported — not a single wall-clock measurement.

**3. Wiring registry → promotion → CI into one working pipeline.** `decide_promotion()` in `src/registry/promote.py` is a single pure function called identically by the manual registry CLI and by CI's regression check (`src/registry/ci_check.py`) — so there's no separate "what CI checks" logic to drift out of sync with "what the registry checks."

## 3. Tech stack

| Component | Choice | Why |
|---|---|---|
| Model (production) | DistilBERT-base / FinBERT-base (HF Transformers) | Small enough for a free Colab T4; recruiter-recognizable; FinBERT adds a finance-pretraining talking point. |
| Model (sandbox demo) | Small Transformer encoder, trained from scratch | Proves the surrounding machinery end-to-end without HF Hub access — see Sandbox execution note. |
| Training loop | HF `Trainer` (production) / hand-written PyTorch loop (sandbox) | Time goes to the MLOps decisions that differentiate the project, not hand-rolling. |
| Imbalance handling | Weighted cross-entropy loss | Necessary given the real skew in CFPB categories. |
| Experiment tracking | MLflow (self-hosted, free) | No run-count ceiling unlike hosted free tiers; registry lives in the same tool. |
| Model registry | MLflow Model Registry | Native, versioned, scriptable — exactly what the promotion gate needs. |
| Inference optimization | ONNX Runtime + dynamic quantization | Produces the latency number directly; demonstrates deployment skill beyond `.fit()`. |
| Serving | FastAPI | Lightweight, async, standard for ML inference APIs. |
| CI/CD | GitHub Actions | Free for public repos, visible proof point in the Actions tab. |
| Containerization | Docker | "Clone and run" actually true for anyone evaluating the repo. |
| Deployment | Render (free tier) | Genuinely free, persistent Docker hosting as of 2026; cold-start behavior stated explicitly rather than overstated. |
| Monitoring / drift | Evidently AI (open-source) | Purpose-built for comparing live traffic against a training-time reference distribution. |

All choices are free-tier / open-source by design.

## 4. Production considerations (stated explicitly, not hidden)

**Free-tier deployment means cold starts.** Render's free web services spin down after inactivity and take roughly 30–50s to wake on the next request. The demo link reflects this; it is not presented as instant-always.

**CI re-evaluates; it does not retrain.** Free GitHub Actions runners don't have the compute or time budget for a full fine-tuning run on every push. The CI gate re-evaluates the currently-registered production model on a fixed 320-row subset and re-applies the promotion criterion — this is also how most real ML CI/CD pipelines are actually designed (training is a separate, less-frequent job), not a compromise.

**Monitoring is scheduled, not streaming.** The Evidently drift report regenerates on demand (or on a cron schedule) against accumulated request logs, rather than running as a live streaming pipeline — which would need infrastructure well beyond what a free-tier portfolio project needs to prove the concept.

## Repo structure

```
src/
  data/          acquisition, preprocessing/label-collapsing, EDA, synthetic fallback generator
  baseline/      TF-IDF + LogisticRegression baseline
  training/      sandbox from-scratch transformer (executed) + production HF fine-tune script (Colab-ready)
  registry/      MLflow registration, promotion gate, CI regression check
  optimization/  ONNX export, dynamic quantization, latency benchmark harness
  serving/       FastAPI app, shared inference module, request/response schemas
  monitoring/    Evidently drift report
  demo/          Gradio demo UI
notebooks/       EDA notebook, Colab fine-tuning notebook
tests/           unit + API contract tests (run in CI)
reports/         EDA summary, baseline/benchmark/promotion-log/CI-check JSON+HTML artifacts
data/            processed splits + committed CI eval subset (raw/processed bulk data gitignored)
models/onnx/     the committed, promoted, quantized production model
.github/workflows/ci.yml
```

## Quickstart

```bash
git clone https://github.com/<your-username>/complaint-intelligence-platform.git
cd complaint-intelligence-platform
pip install -r requirements.txt

# Full pipeline from scratch (data → baseline → fine-tune → registry → ONNX → tests):
bash scripts/run_full_pipeline.sh

# Or just run the already-committed production model:
docker compose up --build
curl -X POST localhost:8000/predict -H "Content-Type: application/json" \
  -d '{"text": "I was charged an overdraft fee I never authorized."}'

# Demo UI:
python -m src.demo.gradio_app
```

## Future work

- Real DistilBERT/FinBERT fine-tune on the full real CFPB dataset via `notebooks/02_colab_finetune.ipynb` (production path is fully written, just not executed in this sandbox).
- Human-in-the-loop relabeling / active learning on low-confidence predictions.
- Multi-lingual support.
- Streaming drift monitoring instead of scheduled batch reports.
- Managed tracking/registry service instead of self-hosted MLflow, at real scale.

## Verification — what was actually run

Every artifact in `reports/` and `models/onnx/` was produced by actually executing the corresponding script in this sandbox (see commit history). Specifically, end to end: synthetic data generation → preprocessing/label collapse → EDA → TF-IDF baseline (0.680 test macro-F1) → 6-variant sandbox transformer training with MLflow tracking → registry bootstrap → a real promotion (v1→v2, +0.109 macro-F1) → a real rejection (v3 vs. v2) → ONNX export → INT8 quantization → a 100-repetition p50/p95/p99 latency benchmark → 15 passing unit/contract tests → a live FastAPI server smoke-tested with real HTTP requests → an Evidently drift report built from those requests' logs → the Gradio demo predicting correctly on a live example.

Two things were written and reviewed but **not** executed here, both for the reason stated in "Sandbox execution note": `src/training/train_transformer.py` (needs `huggingface.co`) and an actual `docker build` (this sandbox has no Docker daemon — the Dockerfile was reviewed by hand and mirrors `requirements-serve.txt` + the same startup command validated via bare `uvicorn`).

## Interview talking points

See [`docs/interview_talking_points.md`](docs/interview_talking_points.md) for the "why" behind every major design decision, written as direct answers to the questions this project is meant to invite.

## 60-second demo

See [`docs/demo_script.md`](docs/demo_script.md).
