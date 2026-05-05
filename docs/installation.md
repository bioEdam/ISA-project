# Installation Manual

## Prerequisites

- [Docker](https://docs.docker.com/get-docker/) installed and running
- Internet connection (the Docker build downloads ~430 MB of model and data files)
- At least 4 GB of free disk space (for the Docker image)
- At least 2 GB of free RAM (for model inference)

## Quick Start

1. Extract the zip archive
2. Open a terminal in the extracted directory
3. Build and run:

```bash
docker build -t gru-recommender .
docker run -p 8000:8000 gru-recommender
```

4. Open **http://localhost:8000** in your browser

## Build Details

The Docker build automatically downloads the trained model and data files (~430 MB) from GitHub Releases:

| File | Size | Purpose |
|------|------|---------|
| `track_vocab.parquet` | ~64 MB | Track vocabulary (URI to index mapping) |
| `track_meta.parquet` | ~216 MB | Track metadata (names, artists) |
| `gru_best.pt` | ~150 MB | Trained GRU model checkpoint |

Build time is approximately 5-10 minutes depending on network speed.

## Run

```bash
docker run -p 8000:8000 gru-recommender
```

The application will start loading the model and vocabulary. This takes approximately 30-60 seconds. Once ready, open your browser and navigate to:

**http://localhost:8000**

## Stop

Press `Ctrl+C` in the terminal where the container is running, or:

```bash
docker stop $(docker ps -q --filter ancestor=gru-recommender)
```

## Configuration

To use a different port:

```bash
docker run -p 9000:8000 gru-recommender
```

Then access at `http://localhost:9000`.

## Troubleshooting

**Port already in use:**
```
Error: Bind for 0.0.0.0:8000 failed: port is already allocated
```
Solution: Use a different host port (e.g., `-p 9000:8000`).

**Out of memory:**
The application requires approximately 2 GB of RAM. If the container crashes on startup, ensure your Docker environment has sufficient memory allocated (Docker Desktop > Settings > Resources).

**Slow startup:**
The first request after startup may take 30-60 seconds as the model and vocabulary are loaded into memory. Subsequent requests are fast (<500ms).

**Download fails during build:**
If the build fails during the data download step, check your internet connection and retry. The files are hosted on GitHub Releases at:
https://github.com/bioEdam/ISA-project/releases/tag/v1.0-data