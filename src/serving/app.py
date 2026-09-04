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

from src.serving.inference import ComplaintClassifier
from src.serving.schemas import HealthResponse, PredictRequest, PredictResponse

LOG_PATH = Path("logs/requests.jsonl")
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
_log_lock = Lock()

_classifier: ComplaintClassifier | None = None
_recent_latencies: deque[float] = deque(maxlen=500)
_request_count = 0
_start_time = time.time()


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
    global _request_count
    clf = get_classifier()
    result = clf.predict(req.text)

    _request_count += 1
    _recent_latencies.append(result["latency_ms"])

    with _log_lock:
        with open(LOG_PATH, "a") as f:
            f.write(
                json.dumps(
                    {
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "text_length_chars": len(req.text),
                        "text_length_words": len(req.text.split()),
                        "predicted_category": result["predicted_category"],
                        "confidence": result["confidence"],
                        "latency_ms": result["latency_ms"],
                    }
                )
                + "\n"
            )

    return PredictResponse(**result)


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
        "request_log_path": str(LOG_PATH),
        "request_log_line_count": sum(1 for _ in open(LOG_PATH)) if LOG_PATH.exists() else 0,
    }


@app.get("/")
def root():
    return {
        "service": "Complaint Intelligence Platform",
        "docs": "/docs",
        "endpoints": ["/predict", "/health", "/metrics"],
    }
