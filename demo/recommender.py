"""
recommender.py
--------------
GRUDemo: loads the trained GRU model + track catalog and exposes
search / top_popular / recommend for the CLI demo.

Usage (from project root):
    from demo.recommender import GRUDemo
    demo = GRUDemo(root=Path("."))
    results = demo.search("ed sheeran")
    recs    = demo.recommend([r["corpus_idx"] for r in results[:3]])
"""

import sys
from pathlib import Path

import pandas as pd
import torch


def _rows_to_dicts(df) -> list[dict]:
    return [
        {
            "corpus_idx":  int(idx),
            "corpus_freq": int(freq),
            "track_name":  tname,
            "artist_name": aname,
        }
        for idx, freq, tname, aname in zip(
            df["corpus_idx"], df["corpus_freq"],
            df["track_name"], df["artist_name"],
        )
    ]


class GRUDemo:
    VOCAB_LIMIT = 100_000
    PAD_IDX     = 100_000
    UNK_IDX     = 100_001
    NUM_TOKENS  = 100_002
    MAX_SEQ_LEN = 50
    EMBED_DIM   = 128
    HIDDEN_DIM  = 256
    NUM_LAYERS  = 2
    DROPOUT     = 0.2

    CHECKPOINT  = Path("models") / "gru_best.pt"

    def __init__(self, root: Path):
        self.root = Path(root)
        sys.path.insert(0, str(self.root / "src"))
        from models import GRURecommender  # noqa: PLC0415

        # -- vocabulary --
        vocab = pd.read_parquet(self.root / "processed" / "track_vocab.parquet",
                                columns=["track_uri", "corpus_idx", "corpus_freq"])

        # -- track metadata (names) --
        meta = pd.read_parquet(self.root / "processed" / "track_meta.parquet",
                               columns=["track_uri", "track_name", "artist_name"])

        # join: keep only vocab tracks that have metadata (should be ~100%)
        catalog = vocab.merge(meta, on="track_uri", how="inner")
        catalog.sort_values("corpus_freq", ascending=False, inplace=True)
        catalog.reset_index(drop=True, inplace=True)
        self.catalog = catalog

        # O(1) lookups
        self.uri2idx: dict[str, int] = dict(zip(catalog["track_uri"], catalog["corpus_idx"]))
        self.idx2row: dict[int, dict] = {
            int(idx): {
                "corpus_idx":  int(idx),
                "corpus_freq": int(freq),
                "track_name":  tname,
                "artist_name": aname,
            }
            for idx, freq, tname, aname in zip(
                catalog["corpus_idx"], catalog["corpus_freq"],
                catalog["track_name"], catalog["artist_name"],
            )
        }

        # -- model --
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = GRURecommender(
            self.NUM_TOKENS, self.EMBED_DIM, self.HIDDEN_DIM,
            self.NUM_LAYERS, self.DROPOUT, self.PAD_IDX,
        ).to(self.device)
        ckpt = self.root / self.CHECKPOINT
        model.load_state_dict(torch.load(ckpt, map_location=self.device, weights_only=True))
        model.eval()
        self.model = model

    # ------------------------------------------------------------------

    def search(self, query: str, max_results: int = 10) -> list[dict]:
        """Return up to max_results vocab tracks matching query (name or artist)."""
        q = query.lower().strip()
        mask = (
            self.catalog["track_name"].str.lower().str.contains(q, regex=False)
            | self.catalog["artist_name"].str.lower().str.contains(q, regex=False)
        )
        hits = self.catalog[mask].head(max_results)
        return _rows_to_dicts(hits)

    def top_popular(self, n: int = 10) -> list[dict]:
        """Return the n most frequent tracks in the vocabulary."""
        return _rows_to_dicts(self.catalog.head(n))

    @torch.no_grad()
    def recommend(self, seed_corpus_idxs: list[int], k: int = 10) -> list[dict]:
        """
        Given a list of corpus indices (seed tracks), return top-k next-track predictions.
        Only the last MAX_SEQ_LEN tokens are fed to the model.
        """
        if not seed_corpus_idxs:
            return []

        # OOV → UNK; truncate to last MAX_SEQ_LEN
        mapped = [
            i if i < self.VOCAB_LIMIT else self.UNK_IDX
            for i in seed_corpus_idxs[-self.MAX_SEQ_LEN:]
        ]

        inp = torch.tensor([mapped], dtype=torch.long, device=self.device)
        logits = self.model(inp)[0, -1, : self.VOCAB_LIMIT]   # (VOCAB_LIMIT,)
        topk = logits.topk(k).indices.tolist()

        results = []
        for rank, idx in enumerate(topk, 1):
            row = self.idx2row.get(idx, {})
            results.append({
                "rank":        rank,
                "corpus_idx":  idx,
                "corpus_freq": row.get("corpus_freq", 0),
                "track_name":  row.get("track_name", "<unknown>"),
                "artist_name": row.get("artist_name", "<unknown>"),
            })
        return results