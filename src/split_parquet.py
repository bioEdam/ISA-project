"""
Reads the raw combined parquet produced by json_to_parquet.py and splits it
into two lean parquet files:
  - playlists.parquet  : one row per playlist
  - tracks.parquet     : one row per track entry (pid + position + track info)

Environment variables:
    MPD_RAW     Path to the raw parquet file (default: processed/mpd_raw.parquet)
    MPD_OUT     Output folder for split parquets (default: processed/)
"""
from dotenv import load_dotenv
from pathlib import Path
import os
import pandas as pd
from pathlib import Path

load_dotenv(Path(__file__).parent.parent / '.env')

MPD_RAW  = Path(os.environ.get('MPD_RAW', 'processed/mpd_raw.parquet'))
MPD_OUT = Path(os.environ.get('SPLIT_OUT', 'processed/'))

PLAYLIST_COLS = [
    'pid', 'name', 'num_tracks', 'num_artists', 'num_albums',
    'num_followers', 'num_edits', 'duration_ms', 'modified_at',
    'collaborative', 'has_desc', 'description',
]

TRACK_COLS = [
    'pid', 'pos', 'track_uri', 'track_name',
    'artist_uri', 'artist_name',
    'album_uri',  'album_name', 'track_dur_ms',
]


def main():
    raw_path = MPD_RAW
    out_path = MPD_OUT
    out_path.mkdir(parents=True, exist_ok=True)

    print(f'Reading {raw_path} ...')
    df = pd.read_parquet(raw_path, engine='pyarrow')
    print(f'Loaded {len(df):,} rows, {df["pid"].nunique():,} playlists')

    # ── Playlists (one row per pid) ──────────────────────────────────────────
    playlists = df[PLAYLIST_COLS].drop_duplicates('pid').reset_index(drop=True)
    pl_out = out_path / 'playlists.parquet'
    playlists.to_parquet(pl_out, index=False, engine='pyarrow')
    print(f'playlists.parquet : {len(playlists):,} rows  ({pl_out.stat().st_size/1024:.0f} KB)')

    # ── Tracks (one row per track entry) ────────────────────────────────────
    tracks = df[TRACK_COLS].reset_index(drop=True)
    tr_out = out_path / 'tracks.parquet'
    tracks.to_parquet(tr_out, index=False, engine='pyarrow')
    print(f'tracks.parquet    : {len(tracks):,} rows  ({tr_out.stat().st_size/1024/1024:.1f} MB)')

    print('\nDone.')


if __name__ == '__main__':
    main()