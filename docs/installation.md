# Installation Manual

## Prerequisites

- [Docker](https://docs.docker.com/get-docker/) installed and running (Docker Compose included with Docker Desktop)
- Internet connection (the Docker build downloads ~430 MB of model and data files)
- At least 4 GB of free disk space (for the Docker image)
- At least 2 GB of free RAM (for model inference)

## Quick Start with Docker Compose (Recommended)

Docker Compose runs the application with a PostgreSQL database, enabling the save-playlist feature.

```bash
docker compose up --build
```

This starts two services:
- **app** — the FastAPI web application on **http://localhost:8000**
- **db** — PostgreSQL 16 on port 12345 (schema auto-initialized from `db/schema.sql`)

Open **http://localhost:8000** in your browser.

### Seeding the Database (Optional)

To populate the database with the 50 most popular playlists from the training data, run the seed script locally (requires Python + pandas + psycopg2):

```bash
pip install pandas pyarrow psycopg2-binary
python db/seed.py
```

The seed data file (`db/seed_data.parquet`) is included in the zip. The script connects to the database exposed on port 12345 by Docker Compose.

## Build Details

The Docker build automatically downloads the trained model and data files (~430 MB) from GitHub Releases:

| File                  | Size    | Purpose                                 |
|-----------------------|---------|-----------------------------------------|
| `track_vocab.parquet` | ~64 MB  | Track vocabulary (URI to index mapping) |
| `track_meta.parquet`  | ~216 MB | Track metadata (names, artists)         |
| `gru_best.pt`         | ~150 MB | Trained GRU model checkpoint            |

Build time is approximately 5-10 minutes depending on network speed.

## Stop

**Docker Compose:**
```bash
docker compose down
```
Add `-v` to also remove the database volume: `docker compose down -v`

**Standalone Docker:**

Press `Ctrl+C` in the terminal where the container is running, or:

```bash
docker stop $(docker ps -q --filter ancestor=gru-recommender)
```

## Configuration

**Using a different port (standalone Docker):**

```bash
docker run -p 9000:8000 gru-recommender
```

Then access at `http://localhost:9000`.

**Database URL (Docker Compose):**

The `DATABASE_URL` is configured automatically in `docker-compose.yml`. For local development without Docker Compose, set it in `.env`:

```
DATABASE_URL=postgresql://musicrec:musicrec@localhost:5432/musicrec
```

If `DATABASE_URL` is not set, the app runs without database support (save-playlist feature is disabled).

## Troubleshooting

**Port already in use:**
```
Error: Bind for 0.0.0.0:8000 failed: port is already allocated
```
Solution: Use a different host port (e.g., `-p 9000:8000`) or change the port mapping in `docker-compose.yml`.

**Out of memory:**
The application requires approximately 2 GB of RAM. If the container crashes on startup, ensure your Docker environment has sufficient memory allocated (Docker Desktop > Settings > Resources).

**Slow startup:**
The first request after startup may take 30-60 seconds as the model and vocabulary are loaded into memory. Subsequent requests are fast (<500ms).

**Download fails during build:**
If the build fails during the data download step, check your internet connection and retry. The files are hosted on GitHub Releases at:
https://github.com/bioEdam/ISA-project/releases/tag/v2.0

**Database connection error (Docker Compose):**
If the app starts before the database is ready, Docker Compose will retry automatically (the `db` service has a health check, and `app` depends on `service_healthy`). If issues persist, try `docker compose down -v && docker compose up --build`.