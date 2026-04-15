"""
map_audio_features.py
---------------------
Downloads the Spotify Audio Features dataset from Kaggle and maps it onto
the existing track_meta.parquet by matching on track name + artist name.

The Kaggle dataset (130k tracks) does not use Spotify URIs, so we match
on normalized (artist_name, track_name) pairs. This is fuzzy by nature —
some tracks will not match due to naming differences.

Environment variables:
    TRACK_META_PATH   Path to track_meta.parquet  (default: processed/track_meta.parquet)
    AUDIO_OUT         Output path                 (default: processed/track_audio_features.parquet)
    DATA_DIR          Kaggle download directory   (default: data/)

Usage:
    pip install kaggle
    # Place ~/.kaggle/kaggle.json or set KAGGLE_USERNAME + KAGGLE_KEY env vars
    python map_audio_features.py
"""

from dotenv import load_dotenv
from pathlib import Path

load_dotenv(Path(__file__).parent.parent / '.env')

import os
import re
import subprocess
import pandas as pd


TRACK_META_PATH = Path(os.environ.get('TRACK_META_PATH', 'processed/track_meta.parquet'))
AUDIO_OUT       = Path(os.environ.get('AUDIO_OUT',       'processed/track_audio_features.parquet'))
DATA_DIR        = Path(os.environ.get('DATA_DIR',        'data/'))


def download_dataset(data_dir: Path) -> Path:
    dataset_dir = data_dir / 'spotify-audio-features'
    dataset_dir.mkdir(parents=True, exist_ok=True)

    existing = list(dataset_dir.glob('*.csv'))
    if existing:
        print(f'Dataset already exists at {existing[0]}, skipping download.')
        return existing[0]

    print('Downloading from Kaggle...')
    result = subprocess.run(
        ['kaggle', 'datasets', 'download',
         '-d', 'tomigelo/spotify-audio-features',
         '-p', str(dataset_dir),
         '--unzip'],
        capture_output=True, text=True
    )

    if result.returncode != 0:
        print(result.stderr)
        raise RuntimeError(
            'Kaggle download failed. Make sure:\n'
            '  1. kaggle is installed: pip install kaggle\n'
            '  2. API key exists at ~/.kaggle/kaggle.json\n'
            '     Get it at: https://www.kaggle.com/settings -> API -> Create New Token'
        )

    csv_files = list(dataset_dir.glob('*.csv'))
    if not csv_files:
        raise FileNotFoundError(f'No CSV found in {dataset_dir}')
    return csv_files[0]


def normalize(text: str) -> str:
    """Lowercase, remove punctuation, collapse whitespace."""
    if not isinstance(text, str):
        return ''
    text = text.lower()
    text = re.sub(r'\(.*?\)', '', text)          # remove parenthetical e.g. "(feat. X)"
    text = re.sub(r'\[.*?\]', '', text)          # remove bracketed e.g. "[Remix]"
    text = re.sub(r'[^a-z0-9\s]', '', text)     # remove punctuation
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def load_kaggle(csv_path: Path) -> pd.DataFrame:
    print(f'\nLoading Kaggle audio features from {csv_path} ...')
    df = pd.read_csv(csv_path)
    print(f'  Rows   : {len(df):,}')
    print(f'  Columns: {list(df.columns)}')
    return df


def load_track_meta(path: Path) -> pd.DataFrame:
    print(f'Loading track_meta from {path} ...')
    df = pd.read_parquet(path)
    print(f'  Unique tracks: {len(df):,}')
    return df


def map_features(track_meta: pd.DataFrame, kaggle_df: pd.DataFrame) -> pd.DataFrame:
    # Detect column names — Kaggle dataset uses 'artist_name' and 'track_name'
    # but column names can vary slightly
    artist_col = next((c for c in kaggle_df.columns if 'artist' in c.lower()), None)
    track_col  = next((c for c in kaggle_df.columns if 'track' in c.lower() and 'id' not in c.lower()), None)

    if not artist_col or not track_col:
        raise ValueError(f'Could not detect artist/track columns. Columns: {list(kaggle_df.columns)}')

    print(f'\nUsing Kaggle columns: artist="{artist_col}", track="{track_col}"')

    # Audio feature columns (everything except artist/track name/id cols)
    id_like = {'id', 'uri', 'url', 'name', 'artist', 'track', 'href', 'type', 'analysis'}
    feature_cols = [
        c for c in kaggle_df.columns
        if not any(k in c.lower() for k in id_like)
    ]
    print(f'Audio feature columns: {feature_cols}')

    # Normalize keys
    kaggle_df = kaggle_df.copy()
    kaggle_df['_artist_norm'] = kaggle_df[artist_col].apply(normalize)
    kaggle_df['_track_norm']  = kaggle_df[track_col].apply(normalize)
    kaggle_df['_key'] = kaggle_df['_artist_norm'] + '||' + kaggle_df['_track_norm']

    track_meta = track_meta.copy()
    track_meta['_artist_norm'] = track_meta['artist_name'].apply(normalize)
    track_meta['_track_norm']  = track_meta['track_name'].apply(normalize)
    track_meta['_key'] = track_meta['_artist_norm'] + '||' + track_meta['_track_norm']

    # Deduplicate kaggle on the join key (keep first)
    kaggle_dedup = kaggle_df.drop_duplicates('_key')[['_key'] + feature_cols]

    # ── Pass 1: artist + track match ────────────────────────────────────────
    merged = track_meta.merge(kaggle_dedup, on='_key', how='left')
    matched_p1 = merged[feature_cols[0]].notna().sum()
    print(f'\nPass 1 (artist + track match): {matched_p1:,} / {len(track_meta):,} tracks matched')

    # ── Pass 2: track name only (for unmatched rows) ─────────────────────────
    unmatched_mask = merged[feature_cols[0]].isna()
    if unmatched_mask.sum() > 0:
        kaggle_by_track = kaggle_df.drop_duplicates('_track_norm')[['_track_norm'] + feature_cols]
        kaggle_by_track.columns = ['_track_norm'] + [f + '_p2' for f in feature_cols]

        merged_p2 = merged[unmatched_mask][['track_uri', '_track_norm']].merge(
            kaggle_by_track, on='_track_norm', how='left'
        )
        matched_p2 = merged_p2[feature_cols[0] + '_p2'].notna().sum()
        print(f'Pass 2 (track name only):      {matched_p2:,} additional tracks matched')

        # Fill in pass-2 matches
        for col in feature_cols:
            fill_vals = merged_p2.set_index('track_uri')[col + '_p2']
            merged.loc[unmatched_mask, col] = merged.loc[unmatched_mask, 'track_uri'].map(fill_vals).values

    # Final stats
    total_matched = merged[feature_cols[0]].notna().sum()
    print(f'\nTotal matched: {total_matched:,} / {len(track_meta):,} tracks  '
          f'({total_matched / len(track_meta) * 100:.1f}%)')
    print(f'Unmatched    : {len(track_meta) - total_matched:,} tracks')

    # Drop temp columns
    result = merged.drop(columns=['_artist_norm', '_track_norm', '_key'])
    return result



def main():
    # Download
    csv_path = download_dataset(DATA_DIR)

    # Load
    kaggle_df  = load_kaggle(csv_path)
    track_meta = load_track_meta(TRACK_META_PATH)

    # Map
    enriched = map_features(track_meta, kaggle_df)

    # Save
    AUDIO_OUT.parent.mkdir(parents=True, exist_ok=True)
    enriched.to_parquet(AUDIO_OUT, index=False, engine='pyarrow')
    print(f'\nSaved enriched track metadata to {AUDIO_OUT}')
    print(f'File size: {AUDIO_OUT.stat().st_size / 1024 / 1024:.1f} MB')

    # Preview
    feature_cols = [c for c in enriched.columns
                    if c not in ['track_uri', 'track_name', 'artist_uri',
                                 'artist_name', 'album_uri', 'album_name', 'track_dur_ms']]
    print(f'\nFeature columns in output: {feature_cols}')
    print('\nSample rows with features:')
    print(enriched[enriched[feature_cols[0]].notna()].head(3).to_string())


if __name__ == '__main__':
    main()