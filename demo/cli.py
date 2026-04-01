"""
cli.py
------
Interactive CLI demo for the GRU music recommender.

Usage (from project root):
    python demo/cli.py [--top K]

Commands inside the REPL:
    search <query>   search tracks by name or artist
    top [N]          show N most popular vocab tracks  (default: 10)
    add <N>          add result N from last search/top to seed playlist
    playlist / show  print the current seed playlist
    clear            clear the seed playlist
    recommend [K]    get K next-track recommendations  (default: 10)
    help             show this list
    quit / exit      quit
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent


def _seed_hint(n: int) -> str:
    if n == 0:
        return "add at least 1 track to get recommendations"
    if n <= 2:
        return "weak context — add more tracks for better results"
    if n <= 10:
        return "good context"
    if n <= 50:
        return "excellent context"
    return f"showing last 50 of {n} tracks to model"


def _fmt_track(track_name: str, artist_name: str, corpus_idx: int) -> str:
    name_col  = f"{track_name}  —  {artist_name}"
    rank_col  = f"[rank #{corpus_idx + 1} in vocab]"
    return f"{name_col:<55} {rank_col}"


def _print_list(items: list[dict]) -> None:
    for i, t in enumerate(items, 1):
        print(f"  {i:>2}. {_fmt_track(t['track_name'], t['artist_name'], t['corpus_idx'])}")


def main(default_k: int = 10) -> None:
    print("=" * 65)
    print("  GRU Music Recommender  —  demo")
    print("  Trained on 1M Spotify playlists  |  100K-track vocab")
    print("=" * 65)

    print("Loading vocab, catalog, and model...", end=" ", flush=True)
    sys.path.insert(0, str(Path(__file__).parent))
    from recommender import GRUDemo  # import here so banner prints first
    demo = GRUDemo(root=ROOT)
    print(f"done  ({len(demo.catalog):,} tracks indexed, device: {demo.device})")
    print()
    print("Tip: type  top        to browse popular tracks")
    print("     type  search <name or artist>  to find specific tracks")
    print("     type  help       for all commands")
    print()

    seed: list[dict] = []          # {corpus_idx, track_name, artist_name}
    last_results: list[dict] = []  # last search / top output

    while True:
        try:
            raw = input(">> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not raw:
            continue

        parts = raw.split(maxsplit=1)
        cmd   = parts[0].lower()
        rest  = parts[1] if len(parts) > 1 else ""

        # ---- search ----
        if cmd == "search":
            if not rest:
                print("  usage: search <track name or artist>")
                continue
            last_results = demo.search(rest)
            if not last_results:
                print(f"  No tracks found matching '{rest}'")
            else:
                print(f"  Results for '{rest}':")
                _print_list(last_results)

        # ---- top ----
        elif cmd == "top":
            n = int(rest) if rest.isdigit() else 10
            last_results = demo.top_popular(n)
            print(f"  Top {n} most popular tracks in vocabulary:")
            _print_list(last_results)

        # ---- add ----
        elif cmd == "add":
            if not rest.isdigit():
                print("  usage: add <number>  (from the last search or top result)")
                continue
            idx = int(rest) - 1
            if idx < 0 or idx >= len(last_results):
                print(f"  No result #{rest} — run  search  or  top  first")
                continue
            track = last_results[idx]
            seed.append({
                "corpus_idx":  track["corpus_idx"],
                "track_name":  track["track_name"],
                "artist_name": track["artist_name"],
            })
            print(f"  Added: {track['track_name']}  —  {track['artist_name']}")
            print(f"  Seed: {len(seed)} track(s)  ({_seed_hint(len(seed))})")

        # ---- playlist / show ----
        elif cmd in ("playlist", "show"):
            if not seed:
                print("  Seed playlist is empty.")
            else:
                print(f"  Seed playlist ({len(seed)} tracks, {_seed_hint(len(seed))}):")
                for i, t in enumerate(seed, 1):
                    print(f"  {i:>2}. {t['track_name']}  —  {t['artist_name']}")

        # ---- clear ----
        elif cmd == "clear":
            seed.clear()
            last_results.clear()
            print("  Seed playlist cleared.")

        # ---- recommend ----
        elif cmd == "recommend":
            k = int(rest) if rest.isdigit() else default_k
            if not seed:
                print("  Seed playlist is empty — add tracks first.")
                continue
            idxs = [t["corpus_idx"] for t in seed]
            recs = demo.recommend(idxs, k=k)
            shown = min(len(idxs), 50)
            print(f"  Recommendations (seed: {len(seed)} track(s), {_seed_hint(len(seed))}):")
            if len(seed) > 50:
                print(f"  (using last {shown} tracks as context)")
            for r in recs:
                print(f"  {r['rank']:>2}. {_fmt_track(r['track_name'], r['artist_name'], r['corpus_idx'])}")

        # ---- help ----
        elif cmd == "help":
            print("""
  Commands:
    search <query>     search tracks by name or artist
    top [N]            show N most popular vocab tracks (default: 10)
    add <N>            add result N from last search / top to seed
    playlist / show    print current seed playlist
    clear              clear seed playlist
    recommend [K]      get K recommendations (default: {k})
    help               show this message
    quit / exit        quit
""".format(k=default_k))

        # ---- quit ----
        elif cmd in ("quit", "exit", "q"):
            break

        else:
            print(f"  Unknown command '{cmd}' — type  help  for the list")

    print("Bye.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GRU music recommender demo")
    parser.add_argument("--top", type=int, default=10,
                        help="default number of recommendations (default: 10)")
    args = parser.parse_args()
    main(default_k=args.top)
