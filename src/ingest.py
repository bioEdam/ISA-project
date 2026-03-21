"""
ingest.py
---------
Converts MPD JSON slices into three normalized parquet files.

Three writers run concurrently over the same slice data, so each batch is
processed once and flushed to disk immediately — peak memory is bounded by
batch size, not dataset size.

Output files (written to INGEST_OUT):
    playlists.parquet  — one row per playlist (full metadata, EDA-ready)
    tracks.parquet     — one row per (playlist, track) entry, ordered by pos
    track_meta.parquet — deduplicated track catalog; one row per unique track_uri

Environment variables:
    MPD_PATH    Directory containing mpd.slice.*.json files (required)
    INGEST_OUT  Output directory (default: processed/)
    MPD_SLICES  Number of slices to process (default: 500, max: 1000)
    MPD_BATCH   Slices per in-memory batch (default: 50)
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
INGEST_OUT = Path(os.environ.get('INGEST_OUT', 'processed/'))
MPD_SLICES = int(os.environ.get('MPD_SLICES', 500))
MPD_BATCH  = int(os.environ.get('MPD_BATCH',  50))


def process_batch(
    file_paths: list[Path],
    seen_tracks: set[str],
) -> tuple[pa.Table, pa.Table, pa.Table]:
    """
    Parse a batch of JSON slices into three Arrow tables.

    track_meta rows are emitted only for track URIs not yet seen in prior
    batches, ensuring the output file contains exactly one row per unique track.
    """
    pl_rows: list[dict] = []
    tr_rows: list[dict] = []
    tm_rows: list[dict] = []

    for path in file_paths:
        with open(path, encoding='utf-8') as f:
            slc = json.load(f)

        for pl in slc['playlists']:
            pl_rows.append({
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
            })

            for t in pl['tracks']:
                tr_rows.append({
                    'pid':          pl['pid'],
                    'pos':          t['pos'],
                    'track_uri':    t['track_uri'],
                    'track_name':   t['track_name'],
                    'artist_uri':   t['artist_uri'],
                    'artist_name':  t['artist_name'],
                    'album_uri':    t['album_uri'],
                    'album_name':   t['album_name'],
                    'track_dur_ms': t['duration_ms'],
                })

                if t['track_uri'] not in seen_tracks:
                    seen_tracks.add(t['track_uri'])
                    tm_rows.append({
                        'track_uri':    t['track_uri'],
                        'track_name':   t['track_name'],
                        'artist_uri':   t['artist_uri'],
                        'artist_name':  t['artist_name'],
                        'album_uri':    t['album_uri'],
                        'album_name':   t['album_name'],
                        'track_dur_ms': t['duration_ms'],
                    })

    pl_tbl = pa.Table.from_pandas(pd.DataFrame(pl_rows),  preserve_index=False)
    tr_tbl = pa.Table.from_pandas(pd.DataFrame(tr_rows),  preserve_index=False)
    tm_tbl = pa.Table.from_pandas(
        pd.DataFrame(tm_rows) if tm_rows
        else pd.DataFrame(columns=[
            'track_uri', 'track_name', 'artist_uri', 'artist_name',
            'album_uri', 'album_name', 'track_dur_ms',
        ]),
        preserve_index=False,
    )
    return pl_tbl, tr_tbl, tm_tbl


def main() -> None:
    INGEST_OUT.mkdir(parents=True, exist_ok=True)

    all_slices = sorted(MPD_PATH.glob('mpd.slice.*.json'))
    use_slices = all_slices[:MPD_SLICES]

    print(f'Found {len(all_slices)} slices, processing {len(use_slices)}')
    print(f'Batch size : {MPD_BATCH} slices')
    print(f'Output dir : {INGEST_OUT}')
    print()

    batches = [
        use_slices[i:i + MPD_BATCH]
        for i in range(0, len(use_slices), MPD_BATCH)
    ]

    pl_path = INGEST_OUT / 'playlists.parquet'
    tr_path = INGEST_OUT / 'tracks.parquet'
    tm_path = INGEST_OUT / 'track_meta.parquet'

    seen_tracks: set[str] = set()
    pl_writer = tr_writer = tm_writer = None
    total_playlists = total_tracks = 0

    for i, batch in enumerate(batches):
        print(f'Batch {i + 1}/{len(batches)} ({len(batch)} slices)...', end=' ', flush=True)

        pl_tbl, tr_tbl, tm_tbl = process_batch(batch, seen_tracks)

        if pl_writer is None:
            pl_writer = pq.ParquetWriter(pl_path, pl_tbl.schema)
            tr_writer = pq.ParquetWriter(tr_path, tr_tbl.schema)
            tm_writer = pq.ParquetWriter(tm_path, tm_tbl.schema)

        pl_writer.write_table(pl_tbl)
        tr_writer.write_table(tr_tbl)
        tm_writer.write_table(tm_tbl)

        total_playlists += len(pl_tbl)
        total_tracks    += len(tr_tbl)
        print(f'{total_playlists:,} playlists  |  {total_tracks:,} track entries  |  {len(seen_tracks):,} unique tracks')

    if pl_writer:
        pl_writer.close()
        tr_writer.close()
        tm_writer.close()

    print(f'\nDone.')
    print(f'  playlists.parquet  : {total_playlists:,} rows   ({pl_path.stat().st_size / 1024:.0f} KB)')
    print(f'  tracks.parquet     : {total_tracks:,} rows   ({tr_path.stat().st_size / 1024 / 1024:.1f} MB)')
    print(f'  track_meta.parquet : {len(seen_tracks):,} unique tracks   ({tm_path.stat().st_size / 1024:.0f} KB)')


if __name__ == '__main__':
    main()