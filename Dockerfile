# Multi-stage lightweight production container
FROM python:3.11-slim as builder

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Final runtime image
FROM python:3.11-slim

WORKDIR /app

COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH

COPY src/ ./src/
COPY configs/ ./configs/
COPY app.py benchmark.py ./

EXPOSE 8000 7860

# Default to running the production FastAPI microservice
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
