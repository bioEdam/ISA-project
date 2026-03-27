"""
filter_playlists.py
-------------------
Filters playlists to those with a known genre and a minimum number of tracks.
Outputs a parquet file with the filtered playlist pids.

Environment variables:
    PLAYLISTS_PATH   Path to playlists.parquet       (default: processed/playlists.parquet)
    GENRE_PATH       Path to playlist_genres.parquet (default: processed/playlist_genres.parquet)
    FILTERED_OUT     Output path                     (default: processed/filtered_playlists.parquet)
    MIN_TRACKS       Minimum tracks per playlist      (default: 50)

Usage:
    python filter_playlists.py
"""

from dotenv import load_dotenv
from pathlib import Path

load_dotenv(Path(__file__).parent.parent / '.env')

import os
import pandas as pd


PLAYLISTS_PATH = Path(os.environ.get('PLAYLISTS_PATH', 'processed/playlists.parquet'))
GENRE_PATH     = Path(os.environ.get('GENRE_PATH',     'processed/playlist_genres.parquet'))
FILTERED_OUT   = Path(os.environ.get('FILTERED_OUT',   'processed/filtered_playlists.parquet'))
MIN_TRACKS     = int(os.environ.get('MIN_TRACKS',      50))


def main():
    print(f'Loading playlists from {PLAYLISTS_PATH} ...')
    pls   = pd.read_parquet(PLAYLISTS_PATH, columns=['pid', 'num_tracks'])
    genre = pd.read_parquet(GENRE_PATH,     columns=['pid', 'genre'])

    print(f'  Total playlists  : {len(pls):,}')
    print(f'  Labeled playlists: {len(genre):,}')

    # Keep only labeled playlists with enough tracks
    filtered = (
        pls
        .merge(genre, on='pid', how='inner')
        .query('num_tracks >= @MIN_TRACKS')
        .reset_index(drop=True)
    )

    print(f'\nAfter filtering (genre known + min_tracks={MIN_TRACKS}):')
    print(f'  Playlists remaining: {len(filtered):,}')
    print(f'\nGenre breakdown:')
    genre_counts = filtered['genre'].value_counts()
    for g, cnt in genre_counts.items():
        print(f'  {g:<12} {cnt:>7,}  ({cnt/len(filtered)*100:.1f}%)')

    FILTERED_OUT.parent.mkdir(parents=True, exist_ok=True)
    filtered[['pid', 'genre']].to_parquet(FILTERED_OUT, index=False, engine='pyarrow')
    print(f'\nSaved {len(filtered):,} filtered playlists to {FILTERED_OUT}')


if __name__ == '__main__':
    main()