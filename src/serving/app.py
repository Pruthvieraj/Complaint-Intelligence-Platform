"""
Phase 6 — FastAPI serving layer around the ONNX-quantized model.

Every prediction is appended as one line of structured JSON to
logs/requests.jsonl (predicted class, confidence, latency, a timestamp,
and the input length) — that log is exactly what
src/monitoring/drift_report.py reads to build the Evidently drift
report on a schedule (Section 6.3: monitoring is scheduled, not
streaming).

Phase 8 additions layered on top of the original Phase 6 service:
explainability (/explain), Prometheus metrics (/metrics/prometheus,
alongside the original hand-rolled JSON /metrics), request rate
limiting, and optional API-key auth. See src/serving/explain.py and
src/serving/ratelimit.py.
"""
from __future__ import annotations

import asyncio
import json
import os
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock

from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

from src.serving.explain import explain as explain_occlusion
from src.serving.inference import ComplaintClassifier
from src.serving.ratelimit import build_rate_limiter
from src.serving.schemas import (
    BatchPredictItem,
    BatchPredictRequest,
    BatchPredictResponse,
    ExplainRequest,
    ExplainResponse,
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

# Phase 8 — Prometheus metrics, scraped at /metrics/prometheus (see
# monitoring/ for a docker-compose Prometheus+Grafana stack that reads
# it). Deliberately separate from the hand-rolled JSON /metrics below,
# which the live dashboard's own UI reads directly — swapping that for
# Prometheus's text exposition format would just make the UI uglier for
# no benefit.
PREDICTIONS_TOTAL = Counter(
    "complaint_predictions_total", "Total predictions served, by predicted category", ["category"]
)
REQUEST_LATENCY_SECONDS = Histogram(
    "complaint_request_latency_seconds",
    "Request latency in seconds, by endpoint",
    ["endpoint"],
    buckets=(0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5),
)
HTTP_REQUESTS_TOTAL = Counter(
    "complaint_http_requests_total", "Total HTTP requests, by path and status code", ["endpoint", "status"]
)

# Phase 8 — request rate limiting (Redis-backed if REDIS_URL is set, a
# per-process in-memory counter otherwise). Default is generous enough
# for the live dashboard's own 5s /metrics polling plus normal demo
# traffic; override with RATE_LIMIT_PER_MINUTE.
_rate_limiter = build_rate_limiter(
    max_requests=int(os.environ.get("RATE_LIMIT_PER_MINUTE", "120")),
    window_seconds=60.0,
)


# Phase 8 — optional API-key auth, OFF by default so the public demo
# keeps working with zero configuration. Set REQUIRE_API_KEY=true and
# API_KEY=<secret> to lock /predict, /predict/batch, and /explain
# behind an `X-API-Key` header — the toggle a real paid API would need,
# demonstrated rather than actually switched on for a public portfolio
# demo.
def verify_api_key(x_api_key: str | None = Header(default=None)) -> None:
    if os.environ.get("REQUIRE_API_KEY", "false").lower() != "true":
        return
    expected = os.environ.get("API_KEY")
    if not expected or x_api_key != expected:
        raise HTTPException(status_code=401, detail="Missing or invalid API key (X-API-Key header).")


def _record_prediction(text: str, result: dict) -> None:
    """Shared bookkeeping for /predict, /predict/batch: update the
    in-memory counters the /metrics endpoint reports, update the
    Prometheus counter /metrics/prometheus exposes, and append one line
    to the structured request log that src/monitoring/drift_report.py
    reads."""
    global _request_count
    _request_count += 1
    _recent_latencies.append(result["latency_ms"])
    category = result["predicted_category"]
    _category_counts[category] = _category_counts.get(category, 0) + 1
    PREDICTIONS_TOTAL.labels(category=category).inc()

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


@app.middleware("http")
async def rate_limit_and_metrics_middleware(request: Request, call_next):
    """Enforces the per-client rate limit and records every request's
    latency/status code to Prometheus — one middleware for both since
    they both need to wrap every request identically."""
    client_key = request.client.host if request.client else "unknown"
    allowed, remaining, retry_after = _rate_limiter.allow(client_key)
    if not allowed:
        HTTP_REQUESTS_TOTAL.labels(endpoint=request.url.path, status="429").inc()
        return JSONResponse(
            status_code=429,
            content={"detail": "Rate limit exceeded. Please slow down."},
            headers={"Retry-After": str(int(retry_after) + 1)},
        )

    t0 = time.perf_counter()
    response = await call_next(request)
    REQUEST_LATENCY_SECONDS.labels(endpoint=request.url.path).observe(time.perf_counter() - t0)
    HTTP_REQUESTS_TOTAL.labels(endpoint=request.url.path, status=str(response.status_code)).inc()
    response.headers["X-RateLimit-Remaining"] = str(remaining)
    return response


@app.get("/health", response_model=HealthResponse)
def health():
    try:
        clf = get_classifier()
        return HealthResponse(status="ok", model_version=str(clf.model_path))
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))


@app.post("/predict", response_model=PredictResponse, dependencies=[Depends(verify_api_key)])
async def predict(req: PredictRequest):
    clf = get_classifier()
    # asyncio.to_thread so a burst of concurrent requests can't pile up
    # behind the event loop on CPU-bound ONNX inference (Phase 8 —
    # advanced serving hardening).
    result = await asyncio.to_thread(clf.predict, req.text)
    _record_prediction(req.text, result)
    return PredictResponse(**result)


@app.post("/predict/batch", response_model=BatchPredictResponse, dependencies=[Depends(verify_api_key)])
async def predict_batch(req: BatchPredictRequest):
    """Phase 7 dashboard's batch-classify mode — classify up to 25
    complaints in one round trip instead of one HTTP call per row."""
    clf = get_classifier()
    t0 = time.perf_counter()

    def _run_batch():
        items = []
        for text in req.texts:
            result = clf.predict(text)
            _record_prediction(text, result)
            items.append(BatchPredictItem(text=text, **result))
        return items

    items = await asyncio.to_thread(_run_batch)
    total_latency_ms = (time.perf_counter() - t0) * 1000.0
    return BatchPredictResponse(results=items, total_latency_ms=round(total_latency_ms, 3))


@app.post("/explain", response_model=ExplainResponse, dependencies=[Depends(verify_api_key)])
async def explain_endpoint(req: ExplainRequest):
    """Phase 8 — word-level occlusion importance. See
    src/serving/explain.py for the method and why it's the honest
    choice of explainability technique for this particular model."""
    clf = get_classifier()
    result = await asyncio.to_thread(explain_occlusion, req.text, clf)
    return ExplainResponse(**result)


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


@app.get("/metrics/prometheus")
def metrics_prometheus():
    """Phase 8 — Prometheus text-exposition format, for a real
    Prometheus+Grafana stack (see monitoring/) to scrape. Alongside the
    JSON /metrics above, not instead of it — the live dashboard's UI
    reads that one directly."""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


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
        "endpoints": [
            "/predict",
            "/predict/batch",
            "/explain",
            "/health",
            "/metrics",
            "/metrics/prometheus",
            "/model-info",
        ],
    }
