# Sequential Music Recommendation on Spotify MPD

**Adam Candrák, Tomáš Kubričan**

**Course:** Intelligent System Applications (ISA), 2025/2026 &nbsp;|&nbsp; **Cookie:** 4 — Sequential/Session-Based Models

---

## Overview

Sequential next-track recommendation system trained on the [Spotify Million Playlist Dataset (MPD)](https://www.aicrowd.com/challenges/spotify-million-playlist-dataset-challenge). Given the first *k* tracks of a playlist as a seed, the model predicts the next *m* tracks.

Two neural architectures are implemented and compared:
- **GRURecommender** — 2-layer GRU with learned track embeddings
- **TransformerRecommender** — 2-layer causal Transformer with positional embeddings

A genre-aware popularity baseline is also included for comparison.

---

## Project Structure

```
.
├── README.md
├── requirements.txt
├── .env.example                   # Environment variable template
├── data/
│   └── mpd/data/                  # Raw MPD JSON slices (download separately)
├── notebooks/
│   ├── EDA.ipynb                  # Exploratory data analysis
│   ├── Modeling.ipynb             # Model training & ablation experiments
│   ├── Evaluation.ipynb           # Metric evaluation & model comparison
│   └── Genre_Modeling.ipynb       # Genre-based popularity baseline
├── src/
│   ├── models.py                  # GRURecommender, TransformerRecommender
│   ├── ingest.py                  # JSON slices → normalized parquet files
│   ├── preprocess.py              # Vocabulary, sequence encoding, train/val/test split
│   ├── genre_filter.py            # Label playlists by genre from playlist names
│   ├── filter_playlists.py        # Filter to genre-labeled playlists
│   ├── check.py                   # Dataset validation CLI
│   ├── stats.py                   # Dataset statistics CLI
│   └── validate_ingest.py         # Verify parquet outputs match JSON input
├── processed/                     # Parquet artifacts (generated, gitignored)
└── testings/
    ├── 50k vocab limit/           # Archived ablation run (50K vocab)
    └── 100k vocab limit/          # Final model checkpoints (100K vocab)
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

## Data Pipeline

Run these two scripts in order before opening any notebook:

```bash
# 1. Ingest: JSON slices → three normalized parquet files
MPD_PATH=./data/mpd/data python src/ingest.py

# 2. Preprocess: build vocabulary, encode sequences, stratified train/val/test split
python src/preprocess.py
```

All output lands in `processed/`:

| File | Description |
|---|---|
| `playlists.parquet` | One row per playlist (pid, name, metadata) |
| `tracks.parquet` | One row per track entry (pid, pos, URIs, names) |
| `track_meta.parquet` | Deduplicated track catalog |
| `track_vocab.parquet` | Vocabulary: `track_uri → corpus_idx` (sorted by frequency) |
| `train_seqs.parquet` | Encoded training sequences |
| `val_seqs.parquet` | Encoded validation sequences |
| `test_seqs.parquet` | Encoded test sequences |

**Dataset scale (full 1000 slices):** 1,000,000 playlists · 66,346,428 track entries · 2,262,292 unique tracks · train/val/test split 800K / 100K / 100K

---

## Notebooks

Run in the following order:

| Notebook | Description |
|---|---|
| `EDA.ipynb` | Dataset exploration: distributions, power-law patterns, correlation analysis, playlist name clusters |
| `Modeling.ipynb` | GRU and Transformer training, hyperparameter ablations (sequence length, embedding size) |
| `Evaluation.ipynb` | Precision@K / Recall@K evaluation, model comparison, comparison with RecSys 2018 challenge winners |
| `Genre_Modeling.ipynb` | Genre-aware popularity baseline: classifies playlists by name, recommends top-K genre-popular tracks |

---

## Models

Both models share the same interface: `(batch, seq_len)` token indices → `(batch, seq_len, vocab_size)` logits.

**GRURecommender**
```
Embedding → Dropout → GRU (2 layers, hidden=256) → Dropout → Linear
```

**TransformerRecommender**
```
Embedding + PosEmbedding → Dropout → TransformerEncoder (2 layers, 4 heads, causal mask) → Linear
```

**Hyperparameters**

| Parameter | Value |
|---|---|
| Vocabulary | Top 100,000 tracks + PAD + UNK = 100,002 tokens |
| `EMBED_DIM` | 128 |
| `HIDDEN_DIM` | 256 |
| `NUM_LAYERS` | 2 |
| `NUM_HEADS` | 4 |
| `DROPOUT` | 0.2 |
| `BATCH_SIZE` | 64 |
| `LR` | 1e-3 (Adam + ReduceLROnPlateau) |
| `NUM_EPOCHS` | 15 |
| `MAX_SEQ_LEN` | 50 |

Model sizes: ~5.2M parameters (GRU), ~5.0M parameters (Transformer). Training time on RTX 4090: ~430s/epoch (GRU), ~319s/epoch (Transformer).

---

## Evaluation

**Protocol:** For each test playlist, the first 80% of tracks form the seed; the remaining 20% are the holdout. Top-K predictions from the last seed position are compared against the holdout set.

**Metrics:** Hit@1 (Accuracy), Precision@K, Recall@K for K ∈ {1, 5, 10, 20}

| Model | Accuracy | Prec@10 | Recall@10 | Prec@20 | Recall@20 |
|---|---|---|---|---|---|
| **GRU** | **0.0612** | **0.0720** | **0.0828** | **0.0575** | **0.1273** |
| Transformer | 0.0604 | 0.0681 | 0.0786 | 0.0550 | 0.1230 |
| Genre baseline | — | 0.0386 | 0.0185 | 0.0322 | 0.0309 |

GRU outperforms the Transformer across all metrics, suggesting local sequential co-occurrence patterns are more informative than long-range attention for playlist data. Both models significantly outperform the genre baseline.

Ablation experiments (sequence length 20/50/100 × embedding dimension 64/128/256) are documented in `Modeling.ipynb`.