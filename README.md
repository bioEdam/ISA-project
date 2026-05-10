# Sequential Music Recommendation on Spotify MPD

**Adam Candrák, Tomas Kubričan**

**Course:** Intelligent System Applications (ISA), 2025/2026 \
**Cookie:** 4 - Sequential/Session-Based Models

---

## Overview

This project implements a sequential next-track recommendation system trained on the [Spotify Million Playlist Dataset (MPD)](https://www.aicrowd.com/challenges/spotify-million-playlist-dataset-challenge). Given the first *k* tracks of a playlist as a seed, the system predicts the next *m* tracks. \
Metrics used: Accuracy (Hit@1), Precision@K, and Recall@K for K in {1, 5, 10, 20}.

1. **Mini-project 1** - EDA, data pipeline, and two neural sequence models (GRU, Transformer)
2. **Mini-project 2** - Alternative approaches: content-based filtering with audio features and item-item collaborative filtering
3. **Mini-project 3** - Deployment of the best model (GRU) as a web application packaged in Docker

---

## Dataset

The [Spotify Million Playlist Dataset](https://www.aicrowd.com/challenges/spotify-million-playlist-dataset-challenge) contains 1,000,000 user-created playlists with 66,346,428 track entries covering 2,262,292 unique tracks. The raw data is distributed as 1,000 JSON slices.

---

## Mini-project 1: Neural Sequence Models

### Data Pipeline

Two scripts process the raw MPD JSON slices into model-ready sequences:

```bash
# 1. Ingest: JSON slices -> three normalized parquet files (batched, bounded memory)
MPD_PATH=./data/mpd/data python src/ingest.py

# 2. Preprocess: build vocabulary, encode sequences, stratified train/val/test split
python src/preprocess.py
```

Output into `processed/`:

| File                            | Description                                                 |
|---------------------------------|-------------------------------------------------------------|
| `playlists.parquet`             | One row per playlist (pid, name, metadata)                  |
| `tracks.parquet`                | One row per track entry (pid, position, URIs, names)        |
| `track_meta.parquet`            | Deduplicated track catalog                                  |
| `track_vocab.parquet`           | Vocabulary: `track_uri -> corpus_idx` (sorted by frequency) |
| `{train,val,test}_seqs.parquet` | Encoded sequences with stratified 80/10/10 split            |

**Dataset scale (full 1000 slices):** 1,000,000 playlists, 66,346,428 track entries, 2,262,292 unique tracks, train/val/test split 800K / 100K / 100K.

### EDA

Exploratory data analysis is documented in `notebooks/EDA.ipynb`, covering distribution analysis, power-law patterns, playlist-level feature correlations, and playlist name clustering.

### Models

Both models share the same interface: `(batch, seq_len)` token indices -> `(batch, seq_len, vocab_size)` logits. Architectures are defined in `src/models.py`.

**GRURecommender:**
```
Embedding -> Dropout -> GRU (2 layers, hidden=256) -> Dropout -> Linear
```

**TransformerRecommender:**
```
Embedding + PosEmbedding -> Dropout -> TransformerEncoder (2 layers, 4 heads, causal mask) -> Linear
```

Training, hyperparameter ablations (sequence length 20/50/100, embedding dimension 64/128/256), and checkpoint saving are implemented in `notebooks/Modeling.ipynb`.

| Parameter           | Value                                           |
|---------------------|-------------------------------------------------|
| Vocabulary          | Top 100,000 tracks + PAD + UNK = 100,002 tokens |
| Embedding dim       | 128                                             |
| Hidden dim (GRU)    | 256                                             |
| Layers              | 2                                               |
| Heads (Transformer) | 4                                               |
| Dropout             | 0.2                                             |
| Batch size          | 64                                              |
| Learning rate       | 1e-3 (Adam + ReduceLROnPlateau)                 |
| Epochs              | 15                                              |
| Max sequence length | 50                                              |

Model sizes: ~5.2M parameters (GRU), ~5.0M parameters (Transformer). Training time on RTX 4090: ~430s/epoch (GRU), ~319s/epoch (Transformer).

### Evaluation

Evaluation protocol and metric computation are in `notebooks/Evaluation.ipynb`. For each test playlist, the first 80% of tracks form the seed; the remaining 20% are the holdout. Top-K predictions from the last seed position are compared against the holdout set.

A genre-aware popularity baseline (`notebooks/Genre_Modeling.ipynb`) classifies playlists by name using `src/genre_filter.py`, then recommends top-K popular tracks from the inferred genre. Genre inference accuracy: 61.3% on labeled playlists (~40% of playlists have genre-indicative names).

**Mini-project 1 results:**

| Model          | Accuracy   | Prec@1     | Prec@10    | Recall@10  | Recall@20  |
|----------------|------------|------------|------------|------------|------------|
| **GRU**        | **0.0612** | **0.1178** | **0.0720** | **0.0828** | **0.1273** |
| Transformer    | 0.0604     | 0.1074     | 0.0681     | 0.0786     | 0.1230     |
| Genre Baseline | -          | 0.0492     | 0.0386     | 0.0185     | 0.0309     |

GRU outperforms the Transformer across all metrics - local sequential co-occurrence patterns are more informative than long-range attention for playlist data. Both neural models significantly outperform the genre baseline.

---

## Mini-project 2: Alternative Approaches

The assignment required exploring an alternative technique to the neural models developed in mini-project 1. We explored two approaches: content-based filtering with audio features and item-item collaborative filtering.

### Content-Based Filtering with Audio Features

**Motivation:** We wanted to enrich the track representations beyond just playlist co-occurrence by incorporating explicit acoustic signal (danceability, energy, tempo, valence, etc.). This approach was inspired by creative track submissions in the ACM RecSys Challenge 2018 that incorporated Spotify audio features alongside collaborative signals.

**Challenge:** The Spotify Web API, which provides audio features for individual tracks, has been deprecated. We could not query features for our 2.26M tracks directly. Instead, we used a publicly available [Kaggle dataset of Spotify audio features](https://www.kaggle.com/datasets/tomigelo/spotify-audio-features) and mapped it onto our track catalog by matching on normalized (artist_name, track_name) pairs.

**Pipeline:**

1. `src/map_datasets.py` - Downloads the Kaggle dataset and maps audio features onto `track_meta.parquet` using a two-pass matching strategy (artist+track name, then track name only)
2. `src/filter_tracks_with_data.py` - Filters to tracks with audio features, rebuilds playlist/track parquets keeping only playlists with at least 20 matched tracks
3. `notebooks/Modeling_With_Audio_Features.ipynb` - Builds a content-based recommender using cosine similarity on standardized audio feature vectors

**Audio features used:** `acousticness`, `danceability`, `energy`, `instrumentalness`, `liveness`, `loudness`, `mode`, `speechiness`, `tempo`, `valence`, `popularity` (after dropping `key`, `time_signature`, and `duration_ms` for low signal or categorical issues).

**Coverage after mapping:** 539,913 out of 2,262,292 unique tracks matched (23.9%), retaining ~523K playlists with at least 20 matched tracks.

**Result:** The content-based audio recommender scored near zero across all metrics (Prec@1 = 0.0002, Recall@20 = 0.0003). Two factors explain this: limited coverage (only 23.9% of tracks have features, making most holdout tracks unreachable) and the wrong signal (acoustic similarity does not predict playlist membership - users curate by mood, theme, and personal association, not purely by acoustic proximity).

### Item-Item Collaborative Filtering

After confirming that audio features do not carry useful signal for this task, we implemented item-item collaborative filtering as an alternative classical approach in `notebooks/Modeling_ItemCF.ipynb`.

**Approach:** Build a co-occurrence matrix from training playlists - for every pair of tracks appearing in the same playlist, count how often they co-occur. Given a seed playlist, aggregate co-occurrence scores across all seed tracks and recommend the top-K candidates.

**Design choices:**
- `MIN_COOC = 3` - pairs co-occurring fewer than 3 times are discarded as noise
- `TOP_N_COOC = 100` - only the top 100 neighbors per track are stored
- No training required - purely count-based

**Result:** Item-item CF achieves Prec@1 = 0.0675 and Recall@20 = 0.0795 - significantly better than the genre baseline and dramatically better than the audio approach, confirming that co-occurrence is the dominant signal for playlist continuation.

### Full Model Comparison

| Model               | Accuracy   | Prec@1     | Prec@10    | Recall@10  | Recall@20  |
|---------------------|------------|------------|------------|------------|------------|
| **GRU**             | **0.0612** | **0.1178** | **0.0720** | **0.0828** | **0.1273** |
| Transformer         | 0.0604     | 0.1074     | 0.0681     | 0.0786     | 0.1230     |
| Item-Item CF        | 0.0675     | 0.0675     | 0.0490     | 0.0460     | 0.0795     |
| Genre Baseline      | -          | 0.0492     | 0.0386     | 0.0185     | 0.0309     |
| Content-Based Audio | -          | 0.0002     | 0.0001     | 0.0002     | 0.0003     |

The progression from content-based audio (near zero) to genre recommender to item-item CF to GRU/Transformer reflects a clear hierarchy of signal quality: co-occurrence from user behavior consistently outperforms content features, and exploiting sequential order on top of co-occurrence gives the neural models their additional edge.

---

## Mini-project 3: Deployment

The best-performing model (GRU) is deployed as a really simple web application with a Docker delivery package.

### Web Application

The application is built with **FastAPI** (backend) and a vanilla **HTML/CSS/JS** frontend with a Spotify-inspired dark theme.

**Features:**
- Search tracks by name or artist
- Browse the most popular tracks in the vocabulary
- Build a seed playlist interactively (with duplicate detection)
- Get top-K next-track recommendations (K = 5, 10, or 20)
- Add recommended tracks back to the seed for iterative exploration
- Full playlist management: browse, load, edit (rename + add/remove tracks), and delete saved playlists via a paginated side panel
- Save playlists (seed + recommendations) to a PostgreSQL database
- Context quality indicator (weak / good / excellent based on seed size)

**No user accounts:** The application does not implement any user management. All saved playlists are stored in a shared pool. The original Spotify MPD dataset contains no user identifiers so the model was trained without any notion of per-user preferences. We keep the same approach in the deployed application: playlists are treated as sequences, and all saved playlists feed directly into the retraining pipeline as collective training signal.

**Architecture:**
- `app/main.py` - FastAPI application, lifespan (model + DB pool init), middleware, router registration
- `app/dependencies.py` - Dependency injection helpers (`get_pool`, `get_demo`)
- `app/models.py` - Pydantic request/response models
- `app/routers/recommender.py` - Recommender API endpoints (`/api/search`, `/api/top`, `/api/recommend`, `/api/health`)
- `app/routers/playlists.py` - Playlist CRUD endpoints (`/api/playlists`)
- `app/templates/index.html` - Jinja2 HTML template
- `app/static/style.css` - Dark-themed responsive UI
- `app/static/api.js` - API client functions
- `app/static/ui.js` - UI rendering module
- `app/static/app.js` - Application state and event handlers
- `demo/recommender.py` - `GRUDemo` class: loads model checkpoint + track catalog, exposes search/recommend/top_popular methods
- `demo/cli.py` - Interactive CLI demo (alternative to the web interface)

### Docker

The application is containerized with a `Dockerfile` based on `python:3.11-slim`. The build installs CPU-only PyTorch and downloads the required model and data files (~430 MB) from the GitHub release.

**Option A: Docker Compose (recommended)** - runs the app with a PostgreSQL database for playlist persistence:

```bash
docker compose up --build
```

This starts two services:
- `app` - the FastAPI web application on port 8000, with a cron daemon for scheduled retraining
- `db` - PostgreSQL 16 on port 12345 (auto-initializes schema from `db/schema.sql`)

**Option B: Standalone Docker** - runs the app without a database (save-playlist feature disabled, no retraining):

```bash
docker build -t gru-recommender .
docker run -p 8000:8000 gru-recommender
```

Then open **http://localhost:8000** in your browser.

Startup takes approximately 30-60 seconds as the model and vocabulary are loaded into memory; subsequent requests complete in under 500ms.

### Daily Model Fine-Tuning

The application includes a scheduled retraining pipeline that fine-tunes the GRU model on created playlists stored in the database. A cron job runs daily at 3:00 AM UTC (`scripts/retrain.py`):

1. Checks if at least 10 new playlists have been saved since the last retraining run
2. Fetches all playlists from PostgreSQL and encodes them as training sequences
3. Fine-tunes the model from the **base checkpoint** using a low learning rate (1e-4) for 1 epoch - preventing forgetting of the patterns learned from the 1M MPD playlists
4. Saves the updated checkpoint and restarts the application

Retraining state is logged to the `retrain_log` database table and `models/retrain_state.json`. Fine-tuned checkpoints persist across container rebuilds via a Docker volume.

**Configuration** (environment variables in `docker-compose.yml`):

| Variable             | Default | Description                                  |
|----------------------|---------|----------------------------------------------|
| `MIN_NEW_PLAYLISTS`  | `10`    | Minimum new playlists to trigger retraining  |
| `FINETUNE_LR`        | `1e-4`  | Fine-tuning learning rate                    |
| `FINETUNE_EPOCHS`    | `1`     | Number of fine-tuning epochs                 |

**Manual trigger** (for testing):
```bash
docker exec -it <container> python scripts/retrain.py
```

### GitHub Release

The trained model checkpoint and required data files are hosted on [GitHub Releases (v1.0)](https://github.com/bioEdam/ISA-project/releases/tag/v1.0) so the Docker build can fetch them without bundling large files in the repository:

| File                        | Size    | Purpose                                 |
|-----------------------------|---------|-----------------------------------------|
| `track_vocab.parquet`       | ~64 MB  | Track vocabulary (URI to index mapping) |
| `track_meta.parquet`        | ~216 MB | Track metadata (names, artists)         |
| `gru_best.pt`               | ~150 MB | Trained GRU model checkpoint            |
| `model-deployment-code.zip` | -       | Minimal deployment package (see below)  |

The release also includes a **`model-deployment-code.zip`** archive containing only the files needed to build and run the Docker image - no notebooks, no training scripts, no data pipeline code. It includes the Dockerfile, the FastAPI app, the inference layer (`demo/recommender.py`, `src/models.py`), runtime dependencies (`requirements-app.txt`), and documentation. This is the self-contained package intended for deployment.

Alternatively, the Dockerfile can copy the model/data files from a local `processed/` and `models/` directory instead of downloading them (see comments in the Dockerfile).

### Documentation

- [Installation Manual](docs/installation.md) - how to build and run the Docker image
- [User Manual](docs/user_manual.md) - how to use the web application

---

## Project Structure

```
.
├── README.md
├── requirements.txt                  # Full development dependencies
├── requirements-app.txt              # Minimal runtime dependencies (for Docker)
├── Dockerfile                        # Docker image definition
├── .env.example                      # Environment variable template
├── .dockerignore
│
├── data/                             # Raw MPD JSON slices (download separately, gitignored)
│
├── notebooks/
│   ├── EDA.ipynb                     # [MP1] Exploratory data analysis
│   ├── Modeling.ipynb                # [MP1] GRU + Transformer training & ablations
│   ├── Evaluation.ipynb              # [MP1] Metric evaluation & model comparison
│   ├── Genre_Modeling.ipynb          # [MP1] Genre-based popularity baseline
│   ├── Modeling_With_Audio_Features.ipynb  # [MP2] Content-based audio recommender
│   ├── Modeling_audio_features.py         # [MP2] Script export of audio features notebook
│   └── Modeling_ItemCF.ipynb              # [MP2] Item-item collaborative filtering
│
├── src/
│   ├── models.py                     # GRURecommender, TransformerRecommender architectures
│   ├── ingest.py                     # JSON slices -> normalized parquet files
│   ├── preprocess.py                 # Vocabulary, sequence encoding, train/val/test split
│   ├── map_datasets.py               # [MP2] Map Kaggle audio features onto track catalog
│   ├── filter_tracks_with_data.py    # [MP2] Filter to tracks with audio features
│   ├── genre_filter.py               # Label playlists by genre from playlist names
│   ├── filter_playlists.py           # Filter to genre-labeled playlists
│   ├── check.py                      # Dataset validation CLI
│   ├── stats.py                      # Dataset statistics CLI
│   └── validate_ingest.py            # Verify parquet outputs match JSON input
│
├── app/                              # [MP3] Web application
│   ├── main.py                       # FastAPI application, lifespan, middleware
│   ├── dependencies.py               # Dependency injection (get_pool, get_demo)
│   ├── models.py                     # Pydantic request/response models
│   ├── routers/
│   │   ├── recommender.py            # Recommender API endpoints
│   │   └── playlists.py              # Playlist CRUD endpoints
│   ├── templates/index.html          # HTML template
│   └── static/
│       ├── style.css                 # Dark-themed responsive CSS
│       ├── api.js                    # API client functions
│       ├── ui.js                     # UI rendering module
│       └── app.js                    # Application state and event handlers
│
├── demo/                             # [MP3] Inference layer
│   ├── recommender.py                # GRUDemo class (model loading + inference)
│   └── cli.py                        # Interactive CLI demo
│
├── db/                               # [MP3] Database
│   ├── schema.sql                    # PostgreSQL schema (playlists, playlist_tracks, retrain_log)
│   └── seed.py                       # Seed DB with popular training playlists
│
├── docker-compose.yml                # Multi-service deployment (app + PostgreSQL)
│
├── docs/
│   ├── installation.md               # Docker build & run instructions
│   └── user_manual.md                # Application usage guide
│
├── testings/
│   ├── 50k vocab limit/              # Archived ablation run (50K vocab, 5 epochs)
│   └── 100k vocab limit/             # Final model checkpoints + training logs (100K vocab, 15 epochs)
│
├── processed/                        # Generated parquet artifacts (gitignored)
└── scripts/
    ├── build_release_zip.py          # Package files for GitHub release
    ├── retrain.py                    # [MP3] Daily fine-tuning on user playlists
    ├── entrypoint.sh                 # [MP3] Docker entrypoint (cron + uvicorn restart loop)
    └── crontab                       # [MP3] Cron schedule for daily retraining
```

---

## Setup

**Requirements:** Python 3.10+, PyTorch

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and configure the path to the raw dataset:

```
MPD_PATH=./data/mpd/data
```

The Spotify MPD dataset must be [downloaded separately](https://www.aicrowd.com/challenges/spotify-million-playlist-dataset-challenge). The number of slices to process is configurable via `MPD_SLICES` (default: all 1000).

---

## Reproducing Results

### Mini-project 1

```bash
# 1. Run the data pipeline
MPD_PATH=./data/mpd/data python src/ingest.py
python src/preprocess.py

# 2. Open notebooks in order
jupyter notebook notebooks/EDA.ipynb
jupyter notebook notebooks/Modeling.ipynb
jupyter notebook notebooks/Evaluation.ipynb
jupyter notebook notebooks/Genre_Modeling.ipynb
```

### Mini-project 2

```bash
# 1. Map audio features (requires Kaggle API key)
python src/map_datasets.py
python src/filter_tracks_with_data.py

# 2. Run the notebooks
jupyter notebook notebooks/Modeling_With_Audio_Features.ipynb
jupyter notebook notebooks/Modeling_ItemCF.ipynb
```

### Mini-project 3

```bash
# Option A: Docker Compose with database (recommended)
docker compose up --build

# Option B: Standalone Docker (no database, save-playlist disabled)
docker build -t gru-recommender .
docker run -p 8000:8000 gru-recommender

# Option C: Run locally (requires processed/ and models/ directories)
# Optionally set DATABASE_URL in .env for playlist persistence
uvicorn app.main:app --host 0.0.0.0 --port 8000
```