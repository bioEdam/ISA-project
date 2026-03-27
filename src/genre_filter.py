"""
genre_filter.py
---------------
Assigns genre labels to playlists based on normalized playlist names and
saves the result to a parquet file.

Environment variables:
    PLAYLISTS_PATH   Path to playlists.parquet (default: processed/playlists.parquet)
    GENRE_OUT        Output path (default: processed/playlist_genres.parquet)

Usage:
    python genre_filter.py
"""

from dotenv import load_dotenv
from pathlib import Path

load_dotenv(Path(__file__).parent.parent / '.env')

import os
import re
import pandas as pd


PLAYLISTS_PATH = Path(os.environ.get('PLAYLISTS_PATH', 'processed/playlists.parquet'))
GENRE_OUT      = Path(os.environ.get('GENRE_OUT',      'processed/playlist_genres.parquet'))

GENRE_MAP = {
    'rap':       [
                    'rap', 'hip hop', 'hiphop', 'hip-hop', 'trap', 'drill',
                    'urban', 'bars', 'freestyle', 'flow', 'gang', 'street',
                    'lil',
                ],
    'country':   [
                    'country', 'bluegrass', 'western', 'southern', 'cowboy',
                    'folk', 'americana',
                ],
    'rock':      [
                    'rock', 'metal', 'punk', 'indie', 'alternative', 'alt',
                    'grunge', 'emo', 'hardcore', 'bands',
                ],
    'chill':     [
                    'chill', 'sleep', 'vibes', 'vibe', 'feels', 'lofi', 'lo fi',
                    'relax', 'calm', 'peaceful', 'ambient', 'mellow', 'easy',
                    'slow', 'acoustic', 'coffee', 'rainy', 'study', 'focus',
                ],
    'workout':   [
                    'workout', 'gym', 'running', 'run', 'cardio', 'fitness',
                    'lifting', 'motivation', 'grind', 'hustle', 'pump', 
                    'training', 'exercise', 'gains',
                ],
    'throwback': [
                    'throwback', 'throwbacks', 'oldies', 'classics', 'classic',
                    'retro', 'old school', 'oldschool', 'nostalgia',
                    '60s', '70s', '80s', '90s', '2000s',
                ],
    'party':     [
                    'party', 'lit', 'jams', 'turn up', 'turnup', 'pregame',
                    'hype', 'banger', 'bangers', 'club', 'friday',
                    'weekend', 'drunk', 'shots',
                ],
    'pop':       [
                    'pop', 'hits', 'summer', 'radio', 'chart',
                    'mainstream', 'sing along', 'singalong', 'feel good',
                ],
    'rnb':       [
                    'r&b', 'rnb', 'soul', 'neo soul', 'smooth', 'groove',
                    'motown', 'slow jam', 'slow jams', 'love songs', 'romance',
                ],
    'christian': [
                    'worship', 'christian', 'gospel', 'praise', 'church',
                    'jesus', 'god', 'faith', 'spiritual', 'devotion',
                ],
    'christmas': [
                    'christmas', 'xmas', 'x-mas', 'holiday', 'festive', 'santa', 'jingle',
                ],
    'edm':       [
                    'edm', 'electronic', 'techno', 'house', 'trance', 'dubstep',
                    'rave', 'dance', 'dj', 'bass', 'beats', 'remix',
                ],
    'latin':     [
                    'latin', 'reggaeton', 'salsa', 'bachata', 'cumbia',
                    'spanish', 'reggae', 'afrobeats',
                ],
    'jazz':      [
                    'jazz', 'blues', 'swing', 'bebop',
                ],
}


def normalize_name(name: str) -> str:
    name = name.lower()
    name = re.sub(r"[.,\/#!$%\^\*;:{}=\_`~()@]", " ", name)
    name = re.sub(r"\s+", " ", name).strip()
    return name


def assign_genre(name_norm: str) -> str:
    for genre, keywords in GENRE_MAP.items():
        for kw in keywords:
            if kw in name_norm:
                return genre
    return 'unknown'


def main():
    print(f'Loading playlists from {PLAYLISTS_PATH} ...')
    pls = pd.read_parquet(PLAYLISTS_PATH, columns=['pid', 'name'])
    print(f'Total playlists: {len(pls):,}')

    pls['name_norm'] = pls['name'].apply(normalize_name)
    pls['genre']     = pls['name_norm'].apply(assign_genre)

    genre_counts = pls['genre'].value_counts()
    total        = len(pls)
    labeled      = total - genre_counts.get('unknown', 0)

    print(f'\nGenre distribution:')
    print(f'{"Genre":<12} {"Count":>8} {"% of total":>12}')
    print('-' * 35)
    for genre, count in genre_counts.items():
        print(f'{genre:<12} {count:>8,} {count / total * 100:>11.1f}%')
    print('-' * 35)
    print(f'{"labeled":<12} {labeled:>8,} {labeled / total * 100:>11.1f}%')
    print(f'{"total":<12} {total:>8,} {"100.0%":>12}')

    out = pls[pls['genre'] != 'unknown'][['pid', 'genre']].reset_index(drop=True)
    GENRE_OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(GENRE_OUT, index=False, engine='pyarrow')
    print(f'\nSaved {len(out):,} labeled playlists to {GENRE_OUT}')


if __name__ == '__main__':
    main()