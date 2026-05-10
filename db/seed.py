#!/usr/bin/env python3
"""
Seed the database with playlists from a pre-built seed data file.

The seed data file (db/seed_data.parquet) contains the top-N playlists
by follower count from the training corpus.  It is small enough to ship
in a GitHub release (~300 KB for 50 playlists).

Generate it with:  python scripts/build_seed_data.py

Usage
-----
    python db/seed.py [--data PATH]

Environment
-----------
    DATABASE_URL  PostgreSQL DSN
                  (default: postgresql://musicrec:musicrec@localhost:12345/musicrec)

Examples
--------
    # seed from the default file (db/seed_data.parquet)
    python db/seed.py

    # seed from a custom path
    python db/seed.py --data /app/db/seed_data.parquet

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
        "--data",
        default=str(Path(__file__).resolve().parent / "seed_data.parquet"),
        help="Path to the seed data parquet (default: db/seed_data.parquet)",
    )
    args = parser.parse_args()

    database_url = os.environ.get(
        "DATABASE_URL", "postgresql://musicrec:musicrec@localhost:12345/musicrec"
    )

    data_path = Path(args.data)
    if not data_path.exists():
        sys.exit(
            f"File not found: {data_path}\n"
            "Generate it with:  python scripts/build_seed_data.py\n"
            "Or download seed_data.parquet from the GitHub release."
        )

    print(f"Loading seed data from {data_path} …")
    df = pd.read_parquet(data_path)

    playlists = df.groupby("pid").first()[["playlist_name"]].reset_index()
    playlists = playlists.rename(columns={"playlist_name": "name"})

    print(f"Connecting to {database_url} …")
    conn = psycopg2.connect(database_url)
    cur = conn.cursor()

    cur.execute("SELECT name FROM playlists")
    existing_names = {row[0] for row in cur.fetchall()}

    inserted = skipped = 0
    for _, pl in playlists.iterrows():
        if pl["name"] in existing_names:
            skipped += 1
            continue

        cur.execute(
            "INSERT INTO playlists (name) VALUES (%s) RETURNING id",
            (pl["name"],),
        )
        pl_id = cur.fetchone()[0]

        pl_tracks = df[df["pid"] == pl["pid"]].sort_values("pos")
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
