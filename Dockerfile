FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY requirements-app.txt .
RUN pip install --no-cache-dir -r requirements-app.txt

ARG RELEASE_URL=https://github.com/bioEdam/ISA-project/releases/download/v1.0
RUN mkdir -p processed models && \
    curl -L -o processed/track_vocab.parquet ${RELEASE_URL}/track_vocab.parquet && \
    curl -L -o processed/track_meta.parquet  ${RELEASE_URL}/track_meta.parquet && \
    curl -L -o models/gru_best.pt            ${RELEASE_URL}/gru_best.pt

COPY src/models.py src/models.py
COPY demo/recommender.py demo/recommender.py
COPY app/ app/

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/health')"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]