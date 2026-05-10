FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends curl cron && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY requirements-app.txt .
RUN pip install --no-cache-dir -r requirements-app.txt


# Uncomment if you want to copy the files from local
#COPY processed/track_vocab.parquet processed/track_vocab.parquet
#COPY processed/track_meta.parquet processed/track_meta.parquet
#COPY models/gru_best.pt models_seed/gru_best.pt

# Comment out if you have already copied the files from local
ARG RELEASE_URL=https://github.com/bioEdam/ISA-project/releases/download/v1.0
RUN mkdir -p processed models models_seed && \
    curl -L -o processed/track_vocab.parquet ${RELEASE_URL}/track_vocab.parquet && \
    curl -L -o processed/track_meta.parquet  ${RELEASE_URL}/track_meta.parquet && \
    curl -L -o models_seed/gru_best.pt       ${RELEASE_URL}/gru_best.pt

COPY src/models.py src/models.py
COPY demo/recommender.py demo/recommender.py
COPY app/ app/
COPY scripts/ scripts/

RUN crontab scripts/crontab && chmod +x scripts/entrypoint.sh

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/health')"

ENTRYPOINT ["scripts/entrypoint.sh"]