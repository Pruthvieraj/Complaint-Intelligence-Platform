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
    body = r.json()
    assert body["total_requests"] >= 1
    assert isinstance(body["category_counts"], dict)
    assert sum(body["category_counts"].values()) >= 1


def test_predict_batch_returns_one_result_per_input(client):
    texts = [
        "My credit report has an account I do not recognize.",
        "A debt collector calls me every day about a paid-off loan.",
    ]
    r = client.post("/predict/batch", json={"texts": texts})
    assert r.status_code == 200
    body = r.json()
    assert len(body["results"]) == len(texts)
    for item, original_text in zip(body["results"], texts):
        assert item["text"] == original_text
        assert 0.0 <= item["confidence"] <= 1.0
    assert body["total_latency_ms"] >= 0


def test_predict_batch_rejects_empty_list(client):
    r = client.post("/predict/batch", json={"texts": []})
    assert r.status_code == 422


def test_predict_batch_rejects_too_many_items(client):
    r = client.post("/predict/batch", json={"texts": ["short complaint"] * 26})
    assert r.status_code == 422


def test_model_info_exposes_real_training_numbers(client):
    r = client.get("/model-info")
    assert r.status_code == 200
    body = r.json()
    assert body["production_model"]["test_macro_f1"] > body["baseline"]["test_macro_f1"]
    assert body["ci_regression_check"]["overall"] == "PASS"
