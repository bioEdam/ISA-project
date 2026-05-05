FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/models.py src/models.py
COPY demo/recommender.py demo/recommender.py
COPY app/ app/

COPY processed/track_vocab.parquet processed/track_vocab.parquet
COPY processed/track_meta.parquet processed/track_meta.parquet
COPY models/gru_best.pt models/gru_best.pt

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/health')"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]