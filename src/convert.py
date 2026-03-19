"""
json_to_parquet.py
------------------
Converts MPD JSON slices into a single parquet file containing all raw playlist
and track data. Processes slices in batches to keep memory usage manageable.

Environment variables:
    MPD_PATH        Path to the folder containing mpd.slice.*.json files (required)
    MPD_OUT         Output parquet file path (default: processed/mpd_raw.parquet)
    MPD_SLICES      Number of slices to process (default: 500, max: 1000)
    MPD_BATCH       Number of slices per batch (default: 50)

Usage:
    MPD_PATH=data/mpd/data python json_to_parquet.py
    MPD_PATH=data/mpd/data MPD_OUT=processed/mpd_raw.parquet MPD_SLICES=500 MPD_BATCH=50 python json_to_parquet.py
"""

from dotenv import load_dotenv
from pathlib import Path

load_dotenv(Path(__file__).parent.parent / '.env')

import os
import json
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


MPD_PATH   = Path(os.environ['MPD_PATH'])
MPD_OUT    = Path(os.environ.get('MPD_OUT',    'processed/mpd_raw.parquet'))
MPD_SLICES = int(os.environ.get('MPD_SLICES', 500))
MPD_BATCH  = int(os.environ.get('MPD_BATCH',  50))


def load_slice(path: str) -> list[dict]:
    """Load one JSON slice and return a flat list of track rows."""
    rows = []
    with open(path, encoding='utf-8') as f:
        s = json.load(f)
    for pl in s['playlists']:
        pl_info = {
            'pid':           pl['pid'],
            'name':          pl['name'],
            'num_tracks':    pl['num_tracks'],
            'num_artists':   pl['num_artists'],
            'num_albums':    pl['num_albums'],
            'num_followers': pl['num_followers'],
            'num_edits':     pl['num_edits'],
            'duration_ms':   pl['duration_ms'],
            'modified_at':   pl['modified_at'],
            'collaborative': pl['collaborative'],
            'has_desc':      'description' in pl,
            'description':   pl.get('description', ''),
        }
        for t in pl['tracks']:
            rows.append({
                **pl_info,
                'pos':          t['pos'],
                'track_uri':    t['track_uri'],
                'track_name':   t['track_name'],
                'artist_uri':   t['artist_uri'],
                'artist_name':  t['artist_name'],
                'album_uri':    t['album_uri'],
                'album_name':   t['album_name'],
                'track_dur_ms': t['duration_ms'],
            })
    return rows


def process_batch(file_paths: list[str]) -> pd.DataFrame:
    rows = []
    for path in file_paths:
        rows.extend(load_slice(path))
    return pd.DataFrame(rows)


def main():
    MPD_OUT.parent.mkdir(parents=True, exist_ok=True)
    tmp_dir = MPD_OUT.parent / '_tmp_batches'
    tmp_dir.mkdir(exist_ok=True)

    all_slices = sorted(MPD_PATH.glob('mpd.slice.*.json'))
    use_slices = all_slices[:MPD_SLICES]

    print(f'Found {len(all_slices)} slices, processing {len(use_slices)}')
    print(f'Batch size: {MPD_BATCH} slices')
    print(f'Output: {MPD_OUT}')
    print()

    batches = [
        use_slices[i:i + MPD_BATCH]
        for i in range(0, len(use_slices), MPD_BATCH)
    ]

    total_rows = 0
    batch_files = []

    # Write each batch to its own temp parquet file
    for i, batch in enumerate(batches):
        print(f'Batch {i+1}/{len(batches)} ({len(batch)} slices)...', end=' ', flush=True)
        df = process_batch([str(p) for p in batch])
        total_rows += len(df)

        batch_file = tmp_dir / f'batch_{i:04d}.parquet'
        df.to_parquet(batch_file, index=False, engine='pyarrow')
        batch_files.append(batch_file)
        del df

        print(f'done ({total_rows:,} rows so far)')

    # Stream-combine all batch files into one without loading into pandas
    print('\nCombining batches...', end=' ', flush=True)
    writer = None
    for bf in batch_files:
        table = pq.read_table(bf)
        if writer is None:
            writer = pq.ParquetWriter(MPD_OUT, table.schema)
        writer.write_table(table)
    if writer:
        writer.close()
    print('done')

    # Clean up temp files
    for bf in batch_files:
        bf.unlink()
    tmp_dir.rmdir()

    print(f'\nFinished. Total rows: {total_rows:,}')
    print(f'File size: {MPD_OUT.stat().st_size / 1024 / 1024:.1f} MB')


if __name__ == '__main__':
    main()