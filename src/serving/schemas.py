from typing import Annotated

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000, description="Consumer complaint narrative")


class ClassProbability(BaseModel):
    label: str
    probability: float


class PredictResponse(BaseModel):
    predicted_category: str
    confidence: float
    latency_ms: float
    top_k: list[ClassProbability]


class HealthResponse(BaseModel):
    status: str
    model_version: str | None = None


ComplaintText = Annotated[str, Field(min_length=1, max_length=5000)]


class BatchPredictRequest(BaseModel):
    """Phase 7 dashboard's batch-classify mode — one call, many complaints,
    instead of round-tripping /predict once per row from the browser."""

    texts: list[ComplaintText] = Field(..., min_length=1, max_length=25, description="Up to 25 complaint narratives")


class BatchPredictItem(PredictResponse):
    text: str


class BatchPredictResponse(BaseModel):
    results: list[BatchPredictItem]
    total_latency_ms: float


class ExplainRequest(BaseModel):
    text: ComplaintText


class WordImportance(BaseModel):
    word: str
    start: int
    end: int
    importance: float
    importance_normalized: float


class ExplainResponse(BaseModel):
    predicted_category: str
    confidence: float
    method: str
    words: list[WordImportance]
