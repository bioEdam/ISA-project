"""
Build a small seed-data parquet from the full processed/ directory.

Extracts the top-N playlists by follower count and their tracks into
a single file (db/seed_data.parquet) suitable for shipping in a
GitHub release (~300 KB for top 50).

Usage:
    python scripts/build_seed_data.py [--n 50] [--processed processed/]
"""

import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=50)
    parser.add_argument("--processed", default=str(ROOT / "processed"))
    args = parser.parse_args()

    processed = Path(args.processed)

    pl = pd.read_parquet(processed / "playlists.parquet", columns=["pid", "name", "num_followers"])
    top = pl.nlargest(args.n, "num_followers")
    pids = set(top["pid"])

    tr = pd.read_parquet(
        processed / "tracks.parquet",
        columns=["pid", "pos", "track_uri", "track_name", "artist_name", "album_name", "track_dur_ms"],
    )
    tracks = tr[tr["pid"].isin(pids)].copy()
    tracks["playlist_name"] = tracks["pid"].map(dict(zip(top["pid"], top["name"])))

    out = ROOT / "db" / "seed_data.parquet"
    tracks.to_parquet(out, index=False)
    print(f"Wrote {out}  ({len(top)} playlists, {len(tracks)} tracks, {out.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
