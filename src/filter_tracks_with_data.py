"""
filter_audio_tracks.py
----------------------
Filters the dataset to only include tracks that have audio features,
then rebuilds clean playlists and tracks parquet files.

Pipeline:
1. Load track_audio_features.parquet — find which tracks have features
2. Filter tracks.parquet to only matched tracks
3. Filter playlists.parquet to only playlists that still have enough tracks
4. Report coverage stats
5. Save filtered_tracks.parquet and filtered_playlists_audio.parquet

Environment variables:
    AUDIO_FEATURES_PATH   Path to track_audio_features.parquet (default: processed/track_audio_features.parquet)
    TRACKS_PATH           Path to tracks.parquet               (default: processed/tracks.parquet)
    PLAYLISTS_PATH        Path to playlists.parquet            (default: processed/playlists.parquet)
    FILTERED_TRACKS_OUT   Output path for filtered tracks      (default: processed/filtered_tracks_audio.parquet)
    FILTERED_PLS_OUT      Output path for filtered playlists   (default: processed/filtered_playlists_audio.parquet)
    MIN_TRACKS            Min tracks per playlist after filter  (default: 5)

Usage:
    python filter_audio_tracks.py
"""

from dotenv import load_dotenv
from pathlib import Path

load_dotenv(Path(__file__).parent.parent / '.env')

import os
import pandas as pd


AUDIO_FEATURES_PATH = Path(os.environ.get('AUDIO_FEATURES_PATH', 'processed/track_audio_features.parquet'))
TRACKS_PATH         = Path(os.environ.get('TRACKS_PATH',         'processed/tracks.parquet'))
PLAYLISTS_PATH      = Path(os.environ.get('PLAYLISTS_PATH',      'processed/playlists.parquet'))
FILTERED_TRACKS_OUT = Path(os.environ.get('FILTERED_TRACKS_OUT', 'processed/filtered_tracks_audio.parquet'))
FILTERED_PLS_OUT    = Path(os.environ.get('FILTERED_PLS_OUT',    'processed/filtered_playlists_audio.parquet'))
MIN_TRACKS          = 20


def main():
    # ── Load ─────────────────────────────────────────────────────────────────
    print('Loading data...')
    audio   = pd.read_parquet(AUDIO_FEATURES_PATH)
    tracks  = pd.read_parquet(TRACKS_PATH,   columns=['pid', 'pos', 'track_uri', 'track_name', 'artist_name', 'album_name', 'track_dur_ms'])
    playlists = pd.read_parquet(PLAYLISTS_PATH)

    print(f'  track_audio_features : {len(audio):,} tracks')
    print(f'  tracks.parquet       : {len(tracks):,} entries, {tracks["track_uri"].nunique():,} unique tracks')
    print(f'  playlists.parquet    : {len(playlists):,} playlists')

    # ── Find matched tracks ───────────────────────────────────────────────────
    # Detect a feature column to check for non-null
    meta_cols = {'track_uri', 'track_name', 'artist_uri', 'artist_name',
                 'album_uri', 'album_name', 'track_dur_ms'}
    feature_cols = [c for c in audio.columns if c not in meta_cols]
    check_col = feature_cols[0] if feature_cols else None

    if check_col is None:
        raise ValueError('No feature columns found in track_audio_features.parquet')

    matched_uris = set(audio[audio[check_col].notna()]['track_uri'])
    print(f'\nTracks with audio features: {len(matched_uris):,} / {len(audio):,}')

    # ── Filter tracks ─────────────────────────────────────────────────────────
    filtered_tracks = tracks[tracks['track_uri'].isin(matched_uris)].copy()
    print(f'\nTrack entries after filter : {len(filtered_tracks):,} / {len(tracks):,} '
          f'({len(filtered_tracks)/len(tracks)*100:.1f}%)')

    # ── Filter playlists ──────────────────────────────────────────────────────
    # Count how many matched tracks each playlist has left
    pl_track_counts = filtered_tracks.groupby('pid').size()
    valid_pids = set(pl_track_counts[pl_track_counts >= MIN_TRACKS].index)

    filtered_tracks = filtered_tracks[filtered_tracks['pid'].isin(valid_pids)].copy()
    filtered_pls    = playlists[playlists['pid'].isin(valid_pids)].copy()

    print(f'Playlists after filter     : {len(filtered_pls):,} / {len(playlists):,} '
          f'({len(filtered_pls)/len(playlists)*100:.1f}%)')
    print(f'  (min {MIN_TRACKS} matched tracks per playlist)')

    # ── Reindex positions ─────────────────────────────────────────────────────
    # Re-number track positions within each playlist after removals
    filtered_tracks = (
        filtered_tracks
        .sort_values(['pid', 'pos'])
        .assign(pos=lambda df: df.groupby('pid').cumcount())
        .reset_index(drop=True)
    )

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f'\nFinal dataset:')
    print(f'  Playlists    : {len(filtered_pls):,}')
    print(f'  Track entries: {len(filtered_tracks):,}')
    print(f'  Unique tracks: {filtered_tracks["track_uri"].nunique():,}')
    avg_len = filtered_tracks.groupby('pid').size().mean()
    print(f'  Avg playlist length: {avg_len:.1f} tracks')

    # ── Save ──────────────────────────────────────────────────────────────────
    FILTERED_TRACKS_OUT.parent.mkdir(parents=True, exist_ok=True)
    filtered_tracks.to_parquet(FILTERED_TRACKS_OUT, index=False, engine='pyarrow')
    filtered_pls.to_parquet(FILTERED_PLS_OUT,    index=False, engine='pyarrow')

    print(f'\nSaved:')
    print(f'  {FILTERED_TRACKS_OUT}  ({FILTERED_TRACKS_OUT.stat().st_size/1024/1024:.1f} MB)')
    print(f'  {FILTERED_PLS_OUT}  ({FILTERED_PLS_OUT.stat().st_size/1024/1024:.1f} MB)')


if __name__ == '__main__':
    main()