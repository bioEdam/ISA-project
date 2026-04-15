import os, sys, time
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

# ── Paths ────────────────────────────────────────────────
PROCESSED = Path('processed/')
TEST_DIR  = Path('testings/audio_features')
MODEL_DIR = Path('models/audio_features')

TEST_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# ── Logging ──────────────────────────────────────────────
class Tee:
    def __init__(self, path, stdout):
        self.file = open(path, 'w', buffering=1, encoding='utf-8')
        self.stdout = stdout

    def write(self, data):
        self.file.write(data)
        self.stdout.write(data)

    def flush(self):
        self.file.flush()
        self.stdout.flush()

log_file = TEST_DIR / f'training_{datetime.now().strftime("%Y%m%d_%H%M")}.log'
sys.stdout = Tee(log_file, sys.stdout)

print("Starting audio feature pipeline...")
t0 = time.time()

# ── Config ───────────────────────────────────────────────
SEED_RATIO  = 0.8
K_VALUES    = [1, 5, 10, 20]
MAX_EVAL    = 5000
RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)

# ── Load Data ────────────────────────────────────────────
audio     = pd.read_parquet(PROCESSED / 'track_audio_features.parquet')
tracks    = pd.read_parquet(PROCESSED / 'filtered_tracks_audio.parquet')
playlists = pd.read_parquet(PROCESSED / 'filtered_playlists_audio.parquet')

print(f'Playlists    : {len(playlists):,}')
print(f'Track entries: {len(tracks):,}')
print(f'Unique tracks: {tracks["track_uri"].nunique():,}')

# ── Feature Columns ──────────────────────────────────────
meta_cols = {
    'track_uri','track_name','artist_uri','artist_name',
    'album_uri','album_name','track_dur_ms'
}

feature_cols = [
    c for c in audio.columns
    if c not in meta_cols and audio[c].dtype in ['float64','int64','float32']
]

print(f'Audio feature columns ({len(feature_cols)}): {feature_cols}')

# ── Filter Valid Tracks ──────────────────────────────────
track_features = audio[audio['track_uri'].isin(tracks['track_uri'].unique())].copy()
track_features = track_features[['track_uri'] + feature_cols].dropna()

print(f'Tracks with full features: {len(track_features):,}')

# ── Normalize Features ───────────────────────────────────
scaler = StandardScaler()
track_features[feature_cols] = scaler.fit_transform(track_features[feature_cols])

# Save scaler
scaler_path = MODEL_DIR / "scaler.npy"
np.save(scaler_path, scaler.mean_)
print(f"Saved scaler stats → {scaler_path}")

# ── Build Lookup ─────────────────────────────────────────
track_to_vec = {
    row['track_uri']: row[feature_cols].values
    for _, row in track_features.iterrows()
}

# ── Evaluation ───────────────────────────────────────────
def evaluate_playlist(seed_tracks, ground_truth):
    if len(seed_tracks) == 0:
        return []

    seed_vecs = [track_to_vec[t] for t in seed_tracks if t in track_to_vec]
    if len(seed_vecs) == 0:
        return []

    profile = np.mean(seed_vecs, axis=0)

    candidates = list(track_to_vec.keys())
    matrix = np.array([track_to_vec[t] for t in candidates])

    sims = cosine_similarity(profile.reshape(1, -1), matrix)[0]

    ranked = [candidates[i] for i in np.argsort(-sims)]
    return ranked


def recall_at_k(preds, truth, k):
    return len(set(preds[:k]) & set(truth)) / len(truth) if truth else 0


# ── Run Evaluation ───────────────────────────────────────
results = []

sample_playlists = playlists.sample(min(MAX_EVAL, len(playlists)), random_state=RANDOM_SEED)

for i, row in enumerate(sample_playlists.itertuples()):
    tracks_list = row.track_uris

    split = int(len(tracks_list) * SEED_RATIO)
    seed = tracks_list[:split]
    gt   = tracks_list[split:]

    preds = evaluate_playlist(seed, gt)

    res = {f"recall@{k}": recall_at_k(preds, gt, k) for k in K_VALUES}
    results.append(res)

    if i % 500 == 0:
        print(f"Processed {i} playlists...")

# ── Aggregate Results ────────────────────────────────────
df = pd.DataFrame(results)
summary = df.mean().to_dict()

print("\nFinal Results:")
for k, v in summary.items():
    print(f"{k}: {v:.4f}")

# ── Save Results ─────────────────────────────────────────
results_path = TEST_DIR / "evaluation_results.csv"
df.to_csv(results_path, index=False)

summary_path = TEST_DIR / "summary.txt"
with open(summary_path, "w") as f:
    for k, v in summary.items():
        f.write(f"{k}: {v:.6f}\n")

print(f"\nSaved results → {results_path}")
print(f"Saved summary → {summary_path}")

print(f"\nDone in {time.time() - t0:.1f}s")