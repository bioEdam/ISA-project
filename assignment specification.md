# ISA Mini-Project 1: Sequential/Session-Based RecSys (Cookie 4)

**Dataset:** Spotify Million Playlist Dataset
**Technique:** Sequential/session-based recommendation (next-item prediction)
**Team:** Pair project | **Deadline:** Week 6 | **Total:** 20 points

---

## Grading Breakdown

### 1.1 EDA & Data Preprocessing (12 pts)

- **A (4 pts) – EDA & Quest:** Explore the raw Spotify playlist data. Define the recommendation problem clearly (e.g., predict next track in a playlist). Visualize distributions, patterns, and dataset characteristics.
- **B (4 pts) – Data Preparation:** Clean, encode, and transform raw data into model-ready sequences. Handle missing values, create track/playlist features, build sequential input (e.g., playlist track order as sequences), train/test split.
- **C (4 pts) – Iterative Justification:** Document and justify every preprocessing decision with evidence. Show iteration — try approaches, compare, adjust. Explain *why* each choice was made.

### 1.2 Modeling & Evaluation (8 pts)

- **A (4 pts) – RecSys Model:** Build a sequence prediction model (LSTM, GRU, or Transformer-based) to predict the next track. Deliver the best model found through experimentation.
- **B (4 pts) – Quality Evaluation:** Evaluate with Accuracy, Precision@K, Recall@K. Compare configurations, interpret results meaningfully.

---

## Cookie 4 Reference

**Goal:** Forecast the next item a user will interact with based on sequential history.

**Recommended approach:**
- Organize data into sequences (playlist track order)
- Engineer features: frequency, position, category/genre preferences
- Model with LSTM / GRU / Transformer
- Experiment with: sequence lengths, batch sizes, embedding techniques (one-hot vs. learned embeddings)

**Evaluation metrics:** Accuracy, Precision@K, Recall@K

---

## Deliverables

- Executable code (Jupyter Notebook) with all operations, outputs, and visualizations
- Best trained model + data pipeline for inference
- Clear commentary on results — presentation quality is a grading component

## Key Requirements

- Two mini-projects total across the semester using two different techniques (classic vs NN); this is one of them
- MP2 must use a different technique and different dataset/topic
- One of the two models will later be deployed in MP3 (Docker, UI, documentation)
- Python + relevant libraries; code must be runnable and well-documented
