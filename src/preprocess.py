"""
preprocess.py
-------------
Transforms ingested track sequences into model-ready artifacts for the
next-track prediction task.

Pipeline
--------
1. Corpus construction  — group track entries into ordered per-playlist
                          sequences (sorted by position index)
2. Corpus filtering     — discard sequences below the minimum support
                          threshold (too-short playlists add noise and make
                          seed/holdout splits degenerate)
3. Vocabulary building  — assign integer indices to track URIs ranked by
                          descending corpus frequency (most frequent = 0).
                          High-frequency items get low indices, which is
                          convenient for frequency-cutoff slicing and
                          popularity-baseline construction.
4. Sequence encoding    — integer-encode each sequence using the vocabulary;
                          all downstream models operate on index tensors, not
                          raw URI strings
5. Stratified partition — split the corpus into train / val / test by
                          playlist length decile so the length distribution
                          is preserved across all three partitions

Output files (written to PREPROCESS_OUT)
-----------------------------------------
    track_vocab.parquet  — vocabulary table: track_uri, corpus_idx, corpus_freq
    train_seqs.parquet   — training partition:   pid, track_idxs (list[int]), seq_len
    val_seqs.parquet     — validation partition
    test_seqs.parquet    — test partition

Notes
-----
- No OOV token is reserved here because vocabulary is built from the full
  corpus before splitting, so every token in val/test was seen at training
  time. Models that need a PAD or OOV index should offset corpus_idx by the
  required number of special tokens.
- Reproducibility is guaranteed by RANDOM_SEED; change it to get a different
  but equally valid split.

Environment variables
---------------------
    TRACKS_PATH      Path to tracks.parquet (default: processed/tracks.parquet)
    PREPROCESS_OUT   Output directory (default: processed/)
    MIN_SEQ_LEN      Minimum sequence length / support threshold (default: 5)
    VAL_RATIO        Validation partition fraction (default: 0.1)
    TEST_RATIO       Test partition fraction (default: 0.1)
    RANDOM_SEED      RNG seed for reproducible partitioning (default: 42)
"""

from dotenv import load_dotenv
from pathlib import Path

load_dotenv(Path(__file__).parent.parent / '.env')

import os
import collections
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


TRACKS_PATH    = Path(os.environ.get('TRACKS_PATH',    'processed/tracks.parquet'))
PREPROCESS_OUT = Path(os.environ.get('PREPROCESS_OUT', 'processed/'))
MIN_SEQ_LEN    = int(os.environ.get('MIN_SEQ_LEN',    5))
VAL_RATIO      = float(os.environ.get('VAL_RATIO',    0.1))
TEST_RATIO     = float(os.environ.get('TEST_RATIO',   0.1))
RANDOM_SEED    = int(os.environ.get('RANDOM_SEED',    42))


# ── 1. Corpus construction + filtering ──────────────────────────────────────

def build_corpus(tracks_path: Path, min_seq_len: int) -> pd.DataFrame:
    """
    Load track entries and assemble the sequence corpus.

    Reads only the columns required for sequence modeling (pid, pos,
    track_uri); string display columns are left in track_meta.parquet.

    Returns a DataFrame with columns:
        pid        : int   — playlist identifier
        track_uris : list  — ordered track URIs (ascending position)
        seq_len    : int   — sequence length
    """
    print(f'Loading track entries from {tracks_path} ...')
    trks = pd.read_parquet(tracks_path, columns=['pid', 'pos', 'track_uri'])
    print(f'  {len(trks):,} track entries across {trks["pid"].nunique():,} playlists')

    corpus = (
        trks
        .sort_values(['pid', 'pos'])
        .groupby('pid')['track_uri']
        .apply(list)
        .reset_index(name='track_uris')
    )
    corpus['seq_len'] = corpus['track_uris'].str.len()

    n_before = len(corpus)
    corpus = corpus[corpus['seq_len'] >= min_seq_len].reset_index(drop=True)
    n_dropped = n_before - len(corpus)
    print(f'  Corpus filtering (min_seq_len={min_seq_len}): '
          f'{n_before:,} → {len(corpus):,} sequences  ({n_dropped:,} dropped)')

    return corpus


# ── 2. Vocabulary construction ───────────────────────────────────────────────

def build_vocabulary(corpus: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """
    Build the track vocabulary from corpus-wide token frequencies.

    Indices are assigned in descending frequency order (rank 0 = most frequent),
    matching the convention used by most embedding layers and enabling
    efficient popularity-baseline construction via simple index slicing.

    Returns
    -------
    vocab_df : DataFrame
        track_uri, corpus_idx (0-based rank), corpus_freq
    uri2idx : dict
        Direct lookup map track_uri → corpus_idx for O(1) encoding.
    """
    freq: collections.Counter = collections.Counter(
        uri for seq in corpus['track_uris'] for uri in seq
    )
    vocab_df = pd.DataFrame([
        {'track_uri': uri, 'corpus_idx': idx, 'corpus_freq': cnt}
        for idx, (uri, cnt) in enumerate(freq.most_common())
    ])
    uri2idx: dict[str, int] = dict(zip(vocab_df['track_uri'], vocab_df['corpus_idx']))

    top1_pct  = max(1, len(vocab_df) // 100)
    top1_cov  = vocab_df.iloc[:top1_pct]['corpus_freq'].sum() / vocab_df['corpus_freq'].sum() * 100
    top10_pct = max(1, len(vocab_df) // 10)
    top10_cov = vocab_df.iloc[:top10_pct]['corpus_freq'].sum() / vocab_df['corpus_freq'].sum() * 100

    print(f'  Vocabulary size     : {len(vocab_df):,} unique tracks')
    print(f'  Total corpus tokens : {vocab_df["corpus_freq"].sum():,}')
    print(f'  Top-1%  coverage    : {top1_cov:.1f}% of all playlist entries')
    print(f'  Top-10% coverage    : {top10_cov:.1f}% of all playlist entries')

    return vocab_df, uri2idx


# ── 3. Sequence encoding ─────────────────────────────────────────────────────

def encode_sequences(corpus: pd.DataFrame, uri2idx: dict[str, int]) -> pd.DataFrame:
    """
    Integer-encode each URI sequence using the vocabulary index map.

    Adds a `track_idxs` column (list of ints) alongside the existing
    `track_uris` column.  Both are retained until partitioning; only
    `track_idxs` is written to the output parquet files.
    """
    corpus = corpus.copy()
    corpus['track_idxs'] = corpus['track_uris'].apply(
        lambda uris: [uri2idx[u] for u in uris]
    )
    return corpus


# ── 4. Stratified train / val / test partition ───────────────────────────────

def stratified_partition(
    corpus: pd.DataFrame,
    val_ratio: float,
    test_ratio: float,
    random_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Partition the encoded corpus into train / val / test splits using
    stratified sampling over playlist length deciles.

    Stratification ensures the empirical length distribution (right-skewed,
    median ~30 tracks) is preserved in all three partitions, preventing the
    val/test sets from being dominated by short or long playlists.

    Returns (train_df, val_df, test_df), each with columns:
        pid, track_idxs (list[int]), seq_len
    """
    rng    = np.random.default_rng(random_seed)
    corpus = corpus.copy()
    corpus['_length_bin'] = pd.qcut(
        corpus['seq_len'], q=10, labels=False, duplicates='drop'
    )

    train_idx: list[int] = []
    val_idx:   list[int] = []
    test_idx:  list[int] = []

    for _, grp in corpus.groupby('_length_bin'):
        idx = grp.index.values.copy()
        rng.shuffle(idx)
        n      = len(idx)
        n_test = max(1, round(n * test_ratio))
        n_val  = max(1, round(n * val_ratio))
        test_idx .extend(idx[:n_test])
        val_idx  .extend(idx[n_test:n_test + n_val])
        train_idx.extend(idx[n_test + n_val:])

    cols = ['pid', 'track_idxs', 'seq_len']
    return (
        corpus.loc[train_idx, cols].reset_index(drop=True),
        corpus.loc[val_idx,   cols].reset_index(drop=True),
        corpus.loc[test_idx,  cols].reset_index(drop=True),
    )


# ── 5. Persistence ───────────────────────────────────────────────────────────

def write_sequences(df: pd.DataFrame, path: Path) -> None:
    """
    Persist a sequence partition to parquet using Arrow's list<int32> type.

    Explicit typing avoids Arrow's default object-array serialization for
    Python lists and ensures compatibility with PyTorch DataLoader batch
    collation.
    """
    schema = pa.schema([
        pa.field('pid',        pa.int32()),
        pa.field('track_idxs', pa.list_(pa.int32())),
        pa.field('seq_len',    pa.int32()),
    ])
    tbl = pa.Table.from_pandas(df, schema=schema, preserve_index=False)
    pq.write_table(tbl, path)


# ── Entry point ──────────────────────────────────────────────────────────────

def main() -> None:
    PREPROCESS_OUT.mkdir(parents=True, exist_ok=True)

    # 1. Corpus construction + filtering
    corpus = build_corpus(TRACKS_PATH, MIN_SEQ_LEN)

    # 2. Vocabulary construction
    print('\nBuilding vocabulary ...')
    vocab_df, uri2idx = build_vocabulary(corpus)

    # 3. Sequence encoding
    print('\nEncoding sequences ...')
    corpus = encode_sequences(corpus, uri2idx)

    # 4. Stratified partition
    print('\nPartitioning corpus ...')
    train, val, test = stratified_partition(corpus, VAL_RATIO, TEST_RATIO, RANDOM_SEED)
    total = len(train) + len(val) + len(test)
    print(f'  Train : {len(train):,}  ({len(train) / total * 100:.1f}%)')
    print(f'  Val   : {len(val):,}   ({len(val)   / total * 100:.1f}%)')
    print(f'  Test  : {len(test):,}  ({len(test)  / total * 100:.1f}%)')

    # 5. Persist artifacts
    print('\nPersisting artifacts ...')

    vocab_path = PREPROCESS_OUT / 'track_vocab.parquet'
    vocab_df.to_parquet(vocab_path, index=False, engine='pyarrow')
    print(f'  track_vocab.parquet  : {len(vocab_df):,} entries  ({vocab_path.stat().st_size / 1024:.0f} KB)')

    for split_name, split_df in [('train', train), ('val', val), ('test', test)]:
        path = PREPROCESS_OUT / f'{split_name}_seqs.parquet'
        write_sequences(split_df, path)
        print(f'  {split_name}_seqs.parquet    : {len(split_df):,} sequences  ({path.stat().st_size / 1024:.0f} KB)')

    print('\nDone.')
    print(f'Vocab size : {len(vocab_df):,}  (use as embedding table size; offset by special tokens if needed)')


if __name__ == '__main__':
    main()
