FROM python:3.11-slim

WORKDIR /app

# System deps kept minimal — this image only needs to *serve* the
# already-exported ONNX model, not train anything.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
# Serving only needs a slice of requirements.txt (fastapi/uvicorn/onnxruntime/
# pydantic) — installing the full training stack (torch/transformers/mlflow)
# would bloat a deploy image that never trains anything. Split file kept
# alongside requirements.txt for exactly this reason.
COPY requirements-serve.txt .
RUN pip install --no-cache-dir -r requirements-serve.txt

COPY src/ src/
COPY models/onnx/ models/onnx/

EXPOSE 8000

ENV PYTHONUNBUFFERED=1

CMD ["uvicorn", "src.serving.app:app", "--host", "0.0.0.0", "--port", "8000"]
