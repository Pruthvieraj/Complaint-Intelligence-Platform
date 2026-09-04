"""API contract tests for the FastAPI serving layer (Phase 6). Requires
the ONNX model to already be exported (models/onnx/) — CI runs this
after the model is committed, same as a fresh clone would."""
import pytest
from fastapi.testclient import TestClient

from src.serving.app import app

pytestmark = pytest.mark.skipif(
    not (__import__("pathlib").Path("models/onnx/model_int8.onnx").exists()
         or __import__("pathlib").Path("models/onnx/model_fp32.onnx").exists()),
    reason="ONNX model not exported yet — run `python -m src.optimization.export_onnx` first",
)


@pytest.fixture()
def client():
    with TestClient(app) as c:
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_predict_returns_valid_category(client):
    r = client.post("/predict", json={"text": "My credit report has an account I do not recognize."})
    assert r.status_code == 200
    body = r.json()
    assert "predicted_category" in body
    assert 0.0 <= body["confidence"] <= 1.0
    assert body["latency_ms"] >= 0
    assert len(body["top_k"]) >= 1


def test_predict_rejects_empty_text(client):
    r = client.post("/predict", json={"text": ""})
    assert r.status_code == 422


def test_metrics_endpoint_after_predictions(client):
    client.post("/predict", json={"text": "I was charged an overdraft fee I did not expect."})
    r = client.get("/metrics")
    assert r.status_code == 200
    assert r.json()["total_requests"] >= 1
