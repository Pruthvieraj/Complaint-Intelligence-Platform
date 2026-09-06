"""
Phase 6 — FastAPI serving layer around the ONNX-quantized model.

Every prediction is appended as one line of structured JSON to
logs/requests.jsonl (predicted class, confidence, latency, a timestamp,
and the input length) — that log is exactly what
src/monitoring/drift_report.py reads to build the Evidently drift
report on a schedule (Section 6.3: monitoring is scheduled, not
streaming).
"""
from __future__ import annotations

import json
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

from src.serving.inference import ComplaintClassifier
from src.serving.schemas import (
    BatchPredictItem,
    BatchPredictRequest,
    BatchPredictResponse,
    HealthResponse,
    PredictRequest,
    PredictResponse,
)

LOG_PATH = Path("logs/requests.jsonl")
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
_log_lock = Lock()

# Phase 7 — a real front-end for the demo, not just Swagger UI. Read once at
# import time and served directly out of memory; it's a single self-contained
# file (inline CSS/JS, no build step, no external CDN) so there's nothing
# else to ship or configure.
_STATIC_DIR = Path(__file__).parent / "static"
_INDEX_HTML = (_STATIC_DIR / "index.html").read_text()

# Static, previously-computed results from the actual training/optimization/
# promotion runs (Phases 2-5) — see docs/ci_failure_demo.md and the
# reports/*.json this was copied from. Served as-is via /model-info so the
# dashboard's "Model & Pipeline" tab shows real numbers, not live-recomputed
# ones (recomputing them on every page load would mean re-running training).
_MODEL_INFO = json.loads((Path(__file__).parent / "model_info.json").read_text())

_classifier: ComplaintClassifier | None = None
_recent_latencies: deque[float] = deque(maxlen=500)
_request_count = 0
_category_counts: dict[str, int] = {}
_start_time = time.time()


def _record_prediction(text: str, result: dict) -> None:
    """Shared bookkeeping for both /predict and /predict/batch: update the
    in-memory counters the /metrics endpoint reports, and append one line to
    the structured request log that src/monitoring/drift_report.py reads."""
    global _request_count
    _request_count += 1
    _recent_latencies.append(result["latency_ms"])
    category = result["predicted_category"]
    _category_counts[category] = _category_counts.get(category, 0) + 1

    with _log_lock:
        with open(LOG_PATH, "a") as f:
            f.write(
                json.dumps(
                    {
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "text_length_chars": len(text),
                        "text_length_words": len(text.split()),
                        "predicted_category": category,
                        "confidence": result["confidence"],
                        "latency_ms": result["latency_ms"],
                    }
                )
                + "\n"
            )


def get_classifier() -> ComplaintClassifier:
    global _classifier
    if _classifier is None:
        _classifier = ComplaintClassifier()
    return _classifier


@asynccontextmanager
async def lifespan(_app: FastAPI):
    get_classifier()  # load model once at startup, not on first request
    yield


app = FastAPI(
    title="Complaint Intelligence Platform API",
    description="CFPB consumer-complaint -> product/issue category classifier.",
    version="1.0.0",
    lifespan=lifespan,
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.get("/health", response_model=HealthResponse)
def health():
    try:
        clf = get_classifier()
        return HealthResponse(status="ok", model_version=str(clf.model_path))
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    clf = get_classifier()
    result = clf.predict(req.text)
    _record_prediction(req.text, result)
    return PredictResponse(**result)


@app.post("/predict/batch", response_model=BatchPredictResponse)
def predict_batch(req: BatchPredictRequest):
    """Phase 7 dashboard's batch-classify mode — classify up to 25
    complaints in one round trip instead of one HTTP call per row."""
    clf = get_classifier()
    t0 = time.perf_counter()
    items = []
    for text in req.texts:
        result = clf.predict(text)
        _record_prediction(text, result)
        items.append(BatchPredictItem(text=text, **result))
    total_latency_ms = (time.perf_counter() - t0) * 1000.0
    return BatchPredictResponse(results=items, total_latency_ms=round(total_latency_ms, 3))


@app.get("/metrics")
def metrics():
    """Lightweight structured metrics endpoint (Phase 6 deliverable) —
    not Prometheus-format by design, this is a portfolio-scale service;
    the request log is the durable source of truth Evidently reads."""
    latencies = list(_recent_latencies)
    uptime_s = time.time() - _start_time
    return {
        "uptime_seconds": round(uptime_s, 1),
        "total_requests": _request_count,
        "recent_window_size": len(latencies),
        "recent_latency_ms": {
            "p50": round(sorted(latencies)[len(latencies) // 2], 2) if latencies else None,
            "mean": round(sum(latencies) / len(latencies), 2) if latencies else None,
        },
        "category_counts": dict(sorted(_category_counts.items(), key=lambda kv: -kv[1])),
        "request_log_path": str(LOG_PATH),
        "request_log_line_count": sum(1 for _ in open(LOG_PATH)) if LOG_PATH.exists() else 0,
    }


@app.get("/model-info")
def model_info():
    """Static results from the actual training/promotion/optimization runs
    (Phases 2-4), for the dashboard's 'Model & Pipeline' tab. See
    src/serving/model_info.json and docs/ci_failure_demo.md for provenance."""
    return _MODEL_INFO


@app.get("/", response_class=HTMLResponse)
def root():
    """Serves the demo UI (src/serving/static/index.html) — this is what a
    human hitting the live URL should see, not a bare JSON blob. Machine
    clients get the same info at /api."""
    return _INDEX_HTML


@app.get("/api")
def api_info():
    return {
        "service": "Complaint Intelligence Platform",
        "docs": "/docs",
        "ui": "/",
        "endpoints": ["/predict", "/predict/batch", "/health", "/metrics", "/model-info"],
    }
