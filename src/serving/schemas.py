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
