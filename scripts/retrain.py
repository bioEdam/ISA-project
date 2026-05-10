"""
retrain.py
----------
Daily fine-tuning of the GRU recommender on user-created playlists
stored in PostgreSQL.

Designed to run as a cron job inside the app container.
Always fine-tunes from the original base checkpoint (not the last
fine-tuned version) to prevent catastrophic drift.

Environment variables:
    DATABASE_URL        PostgreSQL connection string (required)
    MIN_NEW_PLAYLISTS   Minimum new playlists to trigger retraining (default: 10)
    FINETUNE_LR         Learning rate (default: 1e-4)
    FINETUNE_EPOCHS     Number of epochs (default: 1)
    BATCH_SIZE          Training batch size (default: 64)
    MIN_SEQ_LEN         Minimum playlist length after vocab mapping (default: 5)
"""

import asyncio
import json
import logging
import os
import shutil
import signal
import sys
from datetime import datetime, timezone
from pathlib import Path

import asyncpg
import pandas as pd
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from models import GRURecommender

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("retrain")

VOCAB_LIMIT = 100_000
PAD_IDX = 100_000
UNK_IDX = 100_001
NUM_TOKENS = 100_002
MAX_SEQ_LEN = 50
EMBED_DIM = 128
HIDDEN_DIM = 256
NUM_LAYERS = 2
DROPOUT = 0.2

DATABASE_URL = os.environ.get("DATABASE_URL", "")
MIN_NEW_PLAYLISTS = int(os.environ.get("MIN_NEW_PLAYLISTS", "10"))
FINETUNE_LR = float(os.environ.get("FINETUNE_LR", "1e-4"))
FINETUNE_EPOCHS = int(os.environ.get("FINETUNE_EPOCHS", "1"))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "64"))
MIN_SEQ_LEN = int(os.environ.get("MIN_SEQ_LEN", "5"))

BASE_CKPT = ROOT / "models" / "gru_base.pt"
BEST_CKPT = ROOT / "models" / "gru_best.pt"
STATE_FILE = ROOT / "models" / "retrain_state.json"


class PlaylistDataset(Dataset):
    def __init__(self, sequences: list[list[int]]):
        self.sequences = sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        s = torch.tensor(self.sequences[idx], dtype=torch.long)
        return s[:-1], s[1:]


def collate_fn(batch):
    inputs, targets = zip(*batch)
    inp = pad_sequence(inputs, batch_first=True, padding_value=PAD_IDX)
    tgt = pad_sequence(targets, batch_first=True, padding_value=PAD_IDX)
    mask = inp == PAD_IDX
    return inp, tgt, mask


def load_vocab() -> dict[str, int]:
    vocab_path = ROOT / "processed" / "track_vocab.parquet"
    vocab = pd.read_parquet(vocab_path, columns=["track_uri", "corpus_idx"])
    return dict(zip(vocab["track_uri"], vocab["corpus_idx"]))


def load_retrain_state() -> dict:
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())
    return {}


def save_retrain_state(state: dict):
    STATE_FILE.write_text(json.dumps(state, indent=2))


async def fetch_playlists(db_url: str) -> list[list[str]]:
    """Fetch all user playlists from DB, return list of track_uri sequences."""
    conn = await asyncpg.connect(db_url)
    try:
        rows = await conn.fetch(
            """
            SELECT pt.playlist_id, pt.track_uri
            FROM playlist_tracks pt
            JOIN playlists p ON p.id = pt.playlist_id
            ORDER BY pt.playlist_id, pt.position
            """
        )
    finally:
        await conn.close()

    playlists: dict[int, list[str]] = {}
    for row in rows:
        playlists.setdefault(row["playlist_id"], []).append(row["track_uri"])
    return list(playlists.values())


async def get_playlist_count(db_url: str) -> int:
    conn = await asyncpg.connect(db_url)
    try:
        return await conn.fetchval("SELECT COUNT(*) FROM playlists")
    finally:
        await conn.close()


async def log_retrain_start(db_url: str, num_playlists: int) -> int:
    conn = await asyncpg.connect(db_url)
    try:
        return await conn.fetchval(
            """
            INSERT INTO retrain_log (started_at, num_playlists, status)
            VALUES ($1, $2, 'running') RETURNING id
            """,
            datetime.now(timezone.utc),
            num_playlists,
        )
    finally:
        await conn.close()


async def log_retrain_finish(
    db_url: str, log_id: int, num_sequences: int, avg_loss: float, status: str,
    error_message: str | None = None,
):
    conn = await asyncpg.connect(db_url)
    try:
        await conn.execute(
            """
            UPDATE retrain_log
            SET finished_at = $1, num_sequences = $2, avg_loss = $3,
                status = $4, error_message = $5
            WHERE id = $6
            """,
            datetime.now(timezone.utc),
            num_sequences,
            avg_loss,
            status,
            error_message,
            log_id,
        )
    finally:
        await conn.close()


def encode_playlists(
    playlists: list[list[str]], uri2idx: dict[str, int],
) -> list[list[int]]:
    sequences = []
    for uris in playlists:
        seq = []
        for uri in uris:
            idx = uri2idx.get(uri)
            if idx is not None and idx < VOCAB_LIMIT:
                seq.append(idx)
            else:
                seq.append(UNK_IDX)
        if len(seq) > MAX_SEQ_LEN + 1:
            seq = seq[: MAX_SEQ_LEN + 1]
        if len(seq) >= max(MIN_SEQ_LEN, 2):
            sequences.append(seq)
    return sequences


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    total_tokens = 0
    for inp, tgt, mask in loader:
        inp, tgt, mask = inp.to(device), tgt.to(device), mask.to(device)
        logits = model(inp, pad_mask=mask)
        loss = criterion(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1))
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        valid = tgt != PAD_IDX
        n = valid.sum().item()
        total_loss += loss.item() * n
        total_tokens += n
    return total_loss / total_tokens if total_tokens > 0 else 0.0


def ensure_base_checkpoint():
    if not BASE_CKPT.exists():
        if not BEST_CKPT.exists():
            raise FileNotFoundError(f"No checkpoint found at {BEST_CKPT}")
        log.info("Creating base checkpoint: %s -> %s", BEST_CKPT, BASE_CKPT)
        shutil.copy2(BEST_CKPT, BASE_CKPT)


def restart_uvicorn():
    pid_file = Path("/tmp/uvicorn.pid")
    if not pid_file.exists():
        log.warning("No uvicorn PID file found, skipping restart")
        return
    pid = int(pid_file.read_text().strip())
    log.info("Sending SIGTERM to uvicorn (PID %d)", pid)
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        log.warning("Uvicorn process %d not found", pid)


async def main():
    if not DATABASE_URL:
        log.error("DATABASE_URL not set, aborting")
        sys.exit(1)

    log.info("=== Retrain job started ===")

    state = load_retrain_state()
    current_count = await get_playlist_count(DATABASE_URL)
    last_count = state.get("num_playlists", 0)
    new_playlists = current_count - last_count

    log.info(
        "Playlists: %d total, %d at last retrain, %d new (threshold: %d)",
        current_count, last_count, new_playlists, MIN_NEW_PLAYLISTS,
    )

    if new_playlists < MIN_NEW_PLAYLISTS:
        log.info("Below threshold, skipping retraining")
        return

    log_id = await log_retrain_start(DATABASE_URL, current_count)

    try:
        log.info("Loading vocabulary...")
        uri2idx = load_vocab()

        log.info("Fetching playlists from database...")
        raw_playlists = await fetch_playlists(DATABASE_URL)
        log.info("Fetched %d playlists", len(raw_playlists))

        log.info("Encoding sequences...")
        sequences = encode_playlists(raw_playlists, uri2idx)
        log.info(
            "Encoded %d sequences (dropped %d below min length %d)",
            len(sequences), len(raw_playlists) - len(sequences), MIN_SEQ_LEN,
        )

        if not sequences:
            log.warning("No valid sequences after filtering, skipping")
            await log_retrain_finish(
                DATABASE_URL, log_id, 0, 0.0, "skipped",
                "No valid sequences after filtering",
            )
            return

        ensure_base_checkpoint()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        log.info("Device: %s", device)

        model = GRURecommender(
            NUM_TOKENS, EMBED_DIM, HIDDEN_DIM, NUM_LAYERS, DROPOUT, PAD_IDX,
        ).to(device)
        model.load_state_dict(
            torch.load(BASE_CKPT, map_location=device, weights_only=True)
        )
        log.info("Loaded base checkpoint from %s", BASE_CKPT)

        dataset = PlaylistDataset(sequences)
        loader = DataLoader(
            dataset, batch_size=BATCH_SIZE, shuffle=True,
            collate_fn=collate_fn, num_workers=0,
        )

        optimizer = torch.optim.Adam(model.parameters(), lr=FINETUNE_LR)
        criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

        log.info(
            "Fine-tuning: %d sequences, %d batches, lr=%.1e, epochs=%d",
            len(dataset), len(loader), FINETUNE_LR, FINETUNE_EPOCHS,
        )

        avg_loss = 0.0
        for epoch in range(1, FINETUNE_EPOCHS + 1):
            avg_loss = train_epoch(model, loader, optimizer, criterion, device)
            log.info("Epoch %d/%d — loss: %.4f", epoch, FINETUNE_EPOCHS, avg_loss)

        torch.save(model.state_dict(), BEST_CKPT)
        log.info("Saved fine-tuned checkpoint to %s", BEST_CKPT)

        save_retrain_state({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "num_playlists": current_count,
            "num_sequences": len(sequences),
            "avg_loss": avg_loss,
        })

        await log_retrain_finish(
            DATABASE_URL, log_id, len(sequences), avg_loss, "completed",
        )

        log.info("=== Retrain job completed successfully ===")
        restart_uvicorn()

    except Exception as e:
        log.exception("Retrain job failed")
        await log_retrain_finish(
            DATABASE_URL, log_id, 0, 0.0, "failed", str(e),
        )
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
