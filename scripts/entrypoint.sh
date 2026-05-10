#!/bin/bash
set -e

# Seed the models volume on first run (volume mount hides build-time files)
if [ ! -f /app/models/gru_best.pt ]; then
    echo "[entrypoint] Seeding models volume from build artifacts..."
    cp /app/models_seed/* /app/models/
fi

# Dump environment variables for cron (cron runs in a clean env)
env | grep -E '^(DATABASE_URL|MIN_NEW_PLAYLISTS|FINETUNE_LR|FINETUNE_EPOCHS|BATCH_SIZE|MIN_SEQ_LEN|PATH)=' \
    | sed 's/^/export /' > /etc/retrain.env

echo "[entrypoint] Starting cron daemon..."
cron

echo "[entrypoint] Starting uvicorn (restart loop)..."
while true; do
    uvicorn app.main:app --host 0.0.0.0 --port 8000 &
    UVICORN_PID=$!
    echo $UVICORN_PID > /tmp/uvicorn.pid
    echo "[entrypoint] uvicorn started (PID $UVICORN_PID)"
    wait $UVICORN_PID || true
    echo "[entrypoint] uvicorn exited, restarting in 5s..."
    sleep 5
done
