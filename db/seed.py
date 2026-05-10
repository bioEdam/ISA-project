#!/usr/bin/env python3
"""
Seed the database with playlists from the processed Parquet files.

The script picks the top-N playlists by follower count from the training
corpus and inserts them (with all their tracks) into the playlists /
playlist_tracks tables.  It skips playlists whose name already exists so
it is safe to run more than once.

Usage
-----
    python db/seed.py [--n N] [--processed PATH]

Environment
-----------
    DATABASE_URL  PostgreSQL DSN
                  (default: postgresql://musicrec:musicrec@localhost:5432/musicrec)

Examples
--------
    # seed top 20 training playlists (default)
    python db/seed.py

    # seed top 50 using a custom processed dir
    python db/seed.py --n 50 --processed /data/processed

    # point at a remote DB
    DATABASE_URL=postgresql://user:pass@host/db python db/seed.py
"""
import argparse
import os
import sys
from pathlib import Path
import pandas as pd
import psycopg2
import psycopg2.extras

def main() -> None:
    parser = argparse.ArgumentParser(description="Seed DB with training playlists")
    parser.add_argument(
        "--n",
        type=int,
        default=20,
        help="Number of playlists to seed, ranked by follower count (default: 20)",
    )
    parser.add_argument(
        "--processed",
        default=str(Path(__file__).resolve().parent.parent / "processed"),
        help="Path to the processed/ directory (default: <repo-root>/processed)",
    )
    args = parser.parse_args()

    database_url = os.environ.get(
        "DATABASE_URL", "postgresql://musicrec:musicrec@localhost:12345/musicrec"
    )

    processed = Path(args.processed)
    playlists_path = processed / "playlists.parquet"
    tracks_path = processed / "tracks.parquet"

    for p in (playlists_path, tracks_path):
        if not p.exists():
            sys.exit(
                f"File not found: {p}\n"
                "Run  python src/ingest.py  first to generate the processed parquets."
            )

    print(f"Loading parquets from {processed} …")
    playlists_df = pd.read_parquet(playlists_path, columns=["pid", "name", "num_followers"])
    tracks_df = pd.read_parquet(
        tracks_path,
        columns=["pid", "pos", "track_uri", "track_name", "artist_name", "album_name", "track_dur_ms"],
    )

    top = playlists_df.nlargest(args.n, "num_followers")[["pid", "name"]]
    pids = set(top["pid"])
    playlist_tracks = tracks_df[tracks_df["pid"].isin(pids)]

    print(f"Connecting to {database_url} …")
    conn = psycopg2.connect(database_url)
    cur = conn.cursor()

    # Fetch names that are already in the DB so we can skip duplicates
    cur.execute("SELECT name FROM playlists")
    existing_names = {row[0] for row in cur.fetchall()}

    inserted = skipped = 0
    for _, pl in top.iterrows():
        if pl["name"] in existing_names:
            skipped += 1
            continue

        cur.execute(
            "INSERT INTO playlists (name) VALUES (%s) RETURNING id",
            (pl["name"],),
        )
        pl_id = cur.fetchone()[0]

        pl_tracks = (
            playlist_tracks[playlist_tracks["pid"] == pl["pid"]]
            .sort_values("pos")
        )
        rows = [
            (
                pl_id,
                int(row["pos"]),
                row["track_uri"],
                row["track_name"],
                row["artist_name"],
                row["album_name"],
                int(row["track_dur_ms"]) if pd.notna(row.get("track_dur_ms")) else None,
                True,
            )
            for _, row in pl_tracks.iterrows()
        ]
        psycopg2.extras.execute_values(
            cur,
            """INSERT INTO playlist_tracks
               (playlist_id, position, track_uri, track_name, artist_name, album_name, duration_ms, is_seed)
               VALUES %s""",
            rows,
        )
        inserted += 1

    conn.commit()
    cur.close()
    conn.close()

    print(f"Done — inserted {inserted} playlist(s), skipped {skipped} duplicate(s).")


if __name__ == "__main__":
    main()