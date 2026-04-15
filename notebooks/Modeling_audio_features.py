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
print("\nLoading data...")
audio     = pd.read_parquet(PROCESSED / 'track_audio_features.parquet')
tracks    = pd.read_parquet(PROCESSED / 'filtered_tracks_audio.parquet')
playlists = pd.read_parquet(PROCESSED / 'filtered_playlists_audio.parquet')

print(f'  Playlists    : {len(playlists):,}')
print(f'  Track entries: {len(tracks):,}')
print(f'  Unique tracks: {tracks["track_uri"].nunique():,}')

# ── Feature Columns ──────────────────────────────────────
meta_cols = {
    'track_uri', 'track_name', 'artist_uri', 'artist_name',
    'album_uri', 'album_name', 'track_dur_ms'
}

# Drop low-signal and categorical features not suitable for cosine similarity
drop_cols = {'key', 'time_signature', 'duration_ms'}

feature_cols = [
    c for c in audio.columns
    if c not in meta_cols
    and c not in drop_cols
    and audio[c].dtype in ['float64', 'int64', 'float32']
]

print(f'\nAudio feature columns ({len(feature_cols)}): {feature_cols}')

# ── Filter Valid Tracks ──────────────────────────────────
print("\nFiltering tracks with full features...")
track_features = audio[audio['track_uri'].isin(tracks['track_uri'].unique())].copy()
track_features = track_features[['track_uri'] + feature_cols].dropna()
print(f'  Tracks with full features: {len(track_features):,}')

# ── Clip outliers before scaling ─────────────────────────
print("Clipping outliers...")
for col in ['loudness', 'tempo', 'popularity']:
    if col in feature_cols:
        lo = track_features[col].quantile(0.01)
        hi = track_features[col].quantile(0.99)
        track_features[col] = track_features[col].clip(lo, hi)

# ── Normalize Features ───────────────────────────────────
print("Scaling features...")
scaler = StandardScaler()
track_features[feature_cols] = scaler.fit_transform(track_features[feature_cols])

# Save scaler
scaler_path = MODEL_DIR / "scaler.npy"
np.save(scaler_path, scaler.mean_)
print(f"  Saved scaler stats → {scaler_path}")

# ── Build Lookup (vectorized, not iterrows) ───────────────
print("Building track feature lookup...")
feature_array = track_features[feature_cols].values                     # (N, F) numpy array
feature_index = track_features['track_uri'].tolist()                    # list of URIs
uri_to_pos    = {uri: i for i, uri in enumerate(feature_index)}         # URI -> row index
print(f"  Feature matrix shape: {feature_array.shape}")

# ── Build pid -> ordered track URI list ──────────────────
print("Building playlist track lists...")
pid_to_uris = (
    tracks
    .sort_values('pos')
    .groupby('pid')['track_uri']
    .apply(list)
    .to_dict()
)
print(f"  Playlists indexed: {len(pid_to_uris):,}")

# ── Evaluation functions ──────────────────────────────────
def evaluate_playlist(seed_uris: list[str], k_max: int) -> list[str]:
    """Return ranked candidate URIs by cosine similarity to seed mean vector."""
    positions = [uri_to_pos[u] for u in seed_uris if u in uri_to_pos]
    if not positions:
        return []
    profile = feature_array[positions].mean(axis=0, keepdims=True)   # (1, F)
    sims    = cosine_similarity(profile, feature_array)[0]            # (N,)
    seen    = set(seed_uris)
    ranked  = [
        feature_index[i]
        for i in np.argsort(-sims)
        if feature_index[i] not in seen
    ]
    return ranked[:k_max]


def recall_at_k(preds: list, truth: list, k: int) -> float:
    return len(set(preds[:k]) & set(truth)) / len(truth) if truth else 0.0


def precision_at_k(preds: list, truth: list, k: int) -> float:
    return len(set(preds[:k]) & set(truth)) / k if k > 0 else 0.0


# ── Run Evaluation ───────────────────────────────────────
print(f"\nRunning evaluation on up to {MAX_EVAL:,} playlists...")
t_eval = time.time()

rng        = np.random.default_rng(RANDOM_SEED)
all_pids   = playlists['pid'].tolist()
sample_pids = list(rng.choice(all_pids, min(MAX_EVAL, len(all_pids)), replace=False))

results    = []
skipped    = 0
k_max      = max(K_VALUES)

for i, pid in enumerate(sample_pids):
    tracks_list = pid_to_uris.get(pid, [])
    if len(tracks_list) < 5:
        skipped += 1
        continue

    split = max(1, int(len(tracks_list) * SEED_RATIO))
    seed  = tracks_list[:split]
    gt    = tracks_list[split:]
    if not gt:
        skipped += 1
        continue

    preds = evaluate_playlist(seed, k_max)

    res = {}
    for k in K_VALUES:
        res[f'precision@{k}'] = precision_at_k(preds, gt, k)
        res[f'recall@{k}']    = recall_at_k(preds, gt, k)
    results.append(res)

    if (i + 1) % 500 == 0:
        elapsed = time.time() - t_eval
        print(f"  [{i+1:>5}/{len(sample_pids)}]  "
              f"evaluated {len(results):,}  |  "
              f"skipped {skipped}  |  "
              f"{elapsed:.1f}s elapsed")

print(f"\nEvaluation complete: {len(results):,} playlists evaluated, {skipped} skipped")

# ── Aggregate Results ────────────────────────────────────
df      = pd.DataFrame(results)
summary = df.mean().to_dict()

print("\nFinal Results:")
print(f"{'Metric':<15} {'Score':>8}")
print("-" * 25)
for k, v in summary.items():
    print(f"{k:<15} {v:>8.4f}")

# ── Save Results ─────────────────────────────────────────
results_path = TEST_DIR / "evaluation_results.csv"
df.to_csv(results_path, index=False)

summary_path = TEST_DIR / "summary.txt"
with open(summary_path, "w") as f:
    for k, v in summary.items():
        f.write(f"{k}: {v:.6f}\n")

print(f"\nSaved results → {results_path}")
print(f"Saved summary → {summary_path}")
print(f"\nTotal time: {time.time() - t0:.1f}s")