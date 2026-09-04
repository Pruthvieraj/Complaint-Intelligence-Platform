# Interview Talking Points

**Q: Why did you build a promotion gate instead of just deploying your best model?**

"Best on my last training run" isn't the same claim as "better than what's currently in production." The gate (`src/registry/promote.py::decide_promotion`) compares a candidate against the registered production version's *test-set* metrics with a stated margin (macro-F1 must improve by ≥0.01, minority-class F1 must not regress at all) — so promotion is a decision with evidence attached, not a judgment call. I have one documented reject and one documented promote in `reports/promotion_log.json` to show the gate actually enforces this, not just that the code compiles.

**Q: How do you know your latency improvement number is real?**

The benchmark harness (`src/optimization/benchmark.py`) was built *before* I optimized anything, fixes hardware and batch size, discards warmup runs, and reports p50/p95/p99 across 100+ repetitions per model variant — not a single wall-clock measurement. It ran identically against the unoptimized PyTorch model, the FP32 ONNX export, and the INT8-quantized ONNX export, so the before/after comparison in `reports/benchmark_report.json` is apples-to-apples.

**Q: What happens if your model's input distribution shifts after deployment?**

`src/monitoring/drift_report.py` runs Evidently AI against a rolling window of live request logs versus the training-time reference distribution and produces a drift report on a schedule — it's not a real-time streaming system, and I'd say so directly in an interview. But it proves I thought about the model's behavior *after* deployment, not just at training time.

**Q: Why didn't you retrain the model in CI?**

Free CI runners don't have the compute or time budget for it, and it's also not how most real ML CI/CD pipelines are architected — training is typically a separate, less-frequent job. `src/registry/ci_check.py` re-evaluates the currently-registered production model (the committed ONNX file) against a fixed 320-row eval subset and re-applies the promotion criterion on every push — the check that actually needs to run on every change is "did this code change break the deployed model," not "let's retrain from scratch."

**Q: Your dataset is synthetic — doesn't that undercut the whole project?**

Only the *data source* is synthetic, and only because the sandbox this repo was scaffolded in has no network path to `www.consumerfinance.gov` (see the README's "Sandbox execution note" and `src/data/acquire.py`, which tries the real CFPB Socrata endpoint first and falls back loudly). Every other claim — the promotion gate rejecting a real candidate, the ONNX benchmark numbers, the CI regression check, the live API — is genuinely exercised end-to-end. Point `src/data/acquire.py` at a machine with normal internet access and the exact same pipeline runs on the real CFPB Consumer Complaint Database with zero code changes.

**Q: What would you do differently at real scale?**

Move from a self-hosted MLflow instance to a managed tracking/registry service, replace the scheduled drift job with streaming monitoring, fine-tune the real DistilBERT/FinBERT checkpoint (the sandbox demo model is a small transformer trained from scratch — architecturally real, but not pretrained) via the Colab notebook this repo already ships, and revisit the CFPB label taxonomy collapse with a domain expert rather than my own judgment call — all things I scoped out explicitly rather than half-building.
