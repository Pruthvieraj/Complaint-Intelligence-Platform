# 60-Second Demo Script

| Time | Shows |
|---|---|
| 0:00–0:15 | Paste a real complaint into the live demo UI (`python -m src.demo.gradio_app` or the Render link); show the prediction, confidence, and per-request latency appear. |
| 0:15–0:30 | Cut to the before/after latency benchmark chart (unoptimized PyTorch vs. ONNX-quantized) — `reports/benchmark_report.json`. |
| 0:30–0:42 | Cut to the GitHub Actions tab: a passing CI run, and (if kept) the one that failed on purpose. |
| 0:42–0:52 | Cut to the MLflow registry view: model versions, and the promotion decision (promoted vs. rejected) — `reports/promotion_log.json`. |
| 0:52–1:00 | Quick flash of the Evidently drift report and the architecture diagram. |

## Recording checklist

- [ ] `mlflow ui --backend-store-uri sqlite:///mlflow.db` running locally, registry tab open to `complaint-classifier`
- [ ] `docker compose up` running so the API is live for the demo UI
- [ ] `reports/benchmark_report.json` numbers visible (or plotted — see `reports/figures/`)
- [ ] GitHub Actions tab open on a green run
- [ ] `reports/drift_report.html` open in a browser tab
