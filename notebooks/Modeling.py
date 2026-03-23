#!/usr/bin/env python
# coding: utf-8

# # Modeling — Next-Track Prediction
# 
# This notebook builds and trains sequence prediction models for the next-track recommendation task on the Spotify Million Playlist Dataset.
# 
# **Models:**
# - **GRU** (Gated Recurrent Unit) — recurrent sequential model
# - **Transformer** — attention-based sequential model with causal masking
# 
# **Pipeline:** Load preprocessed sequences &rarr; Truncate vocabulary &rarr; Train both models &rarr; Run ablation experiments
# 
# > **Prerequisites:** Run `src/ingest.py` and `src/preprocess.py` first. See EDA.ipynb for data exploration.

# In[1]:


import sys, os, time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

sys.path.insert(0, '..')
from src.models import GRURecommender, TransformerRecommender

sns.set_theme(style='whitegrid', palette='muted')
plt.rcParams['figure.dpi'] = 120

torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

PROCESSED = Path('../processed')
OUT       = Path('outputs'); OUT.mkdir(exist_ok=True)
os.makedirs('../models', exist_ok=True)

# ── Timestamped log file ──────────────────────────────────────────────────────
class _Tee:
    """Duplicate stdout to a timestamped log file."""
    def __init__(self, path, stdout):
        self._f   = open(path, 'w', buffering=1, encoding='utf-8')
        self._out = stdout
    def write(self, s):
        if s:
            self._f.write(s)
        return self._out.write(s)
    def flush(self):
        self._f.flush()
        self._out.flush()

_ts      = datetime.now().strftime('%Y%m%d_%H%M')
LOG_FILE = f'../training_{_ts}.log'
sys.stdout = _Tee(LOG_FILE, sys.stdout)

print(f'Device: {device}')
if device.type == 'cuda':
    print(f'  GPU: {torch.cuda.get_device_name()}')
    print(f'  VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB')
print(f'Log:    {LOG_FILE}')

# ── Load preprocessed data ───────────────────────────────────────────────────
vocab      = pd.read_parquet(PROCESSED / 'track_vocab.parquet')
train_seqs = pd.read_parquet(PROCESSED / 'train_seqs.parquet')
val_seqs   = pd.read_parquet(PROCESSED / 'val_seqs.parquet')
test_seqs  = pd.read_parquet(PROCESSED / 'test_seqs.parquet')

total = len(train_seqs) + len(val_seqs) + len(test_seqs)
print(f'\nVocab:  {len(vocab):,} tracks')
print(f'Train:  {len(train_seqs):,} seqs ({len(train_seqs)/total*100:.0f}%)')
print(f'Val:    {len(val_seqs):,} seqs ({len(val_seqs)/total*100:.0f}%)')
print(f'Test:   {len(test_seqs):,} seqs ({len(test_seqs)/total*100:.0f}%)')

# ── Hyperparameters ──────────────────────────────────────────────────────────
VOCAB_LIMIT  = 50_000
MAX_SEQ_LEN  = 50
EMBED_DIM    = 128
HIDDEN_DIM   = 256
NUM_LAYERS   = 2
NUM_HEADS    = 4
DROPOUT      = 0.2
BATCH_SIZE   = 64
LR           = 1e-3
NUM_EPOCHS   = 15
TRAIN_SUBSET = None    # full training

PAD_IDX    = VOCAB_LIMIT
UNK_IDX    = VOCAB_LIMIT + 1
NUM_TOKENS = VOCAB_LIMIT + 2

# Coverage check
total_freq = vocab['corpus_freq'].sum()
kept_freq  = vocab[vocab['corpus_idx'] < VOCAB_LIMIT]['corpus_freq'].sum()
print(f'\nVocab truncated to {VOCAB_LIMIT:,} tracks  |  '
      f'Coverage: {kept_freq/total_freq*100:.1f}%  |  '
      f'UNK rate: {(1-kept_freq/total_freq)*100:.1f}%')

print(f'\nHyperparameters')
print(f'  VOCAB_LIMIT  = {VOCAB_LIMIT:,}')
print(f'  MAX_SEQ_LEN  = {MAX_SEQ_LEN}')
print(f'  EMBED_DIM    = {EMBED_DIM}')
print(f'  HIDDEN_DIM   = {HIDDEN_DIM}')
print(f'  NUM_LAYERS   = {NUM_LAYERS}')
print(f'  NUM_HEADS    = {NUM_HEADS}')
print(f'  DROPOUT      = {DROPOUT}')
print(f'  BATCH_SIZE   = {BATCH_SIZE}')
print(f'  LR           = {LR}')
print(f'  NUM_EPOCHS   = {NUM_EPOCHS}')
print(f'  TRAIN_SUBSET = {TRAIN_SUBSET}')


# ---
# ## 1. Data Preparation
# 
# For each playlist sequence `[t1, t2, ..., tn]`:
# - **Input:** `[t1, ..., tn-1]` — all tracks except the last
# - **Target:** `[t2, ..., tn]` — the next track at each position
# 
# Sequences longer than `MAX_SEQ_LEN` are truncated; shorter ones are padded within each batch. Tracks with `corpus_idx >= VOCAB_LIMIT` are mapped to a special `UNK` token.

# In[2]:


class PlaylistDataset(Dataset):
    """Next-track prediction dataset from preprocessed sequences."""

    def __init__(self, seqs_df, vocab_limit, max_len, subset=None):
        self.sequences = []
        track_lists = seqs_df['track_idxs'].tolist()
        if subset and subset < len(track_lists):
            idxs = np.random.default_rng(42).choice(len(track_lists), subset, replace=False)
            track_lists = [track_lists[i] for i in idxs]
        unk = vocab_limit + 1
        for track_idxs in track_lists:
            seq = [idx if idx < vocab_limit else unk for idx in track_idxs]
            if len(seq) > max_len + 1:
                seq = seq[:max_len + 1]
            if len(seq) >= 2:
                self.sequences.append(seq)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        s = torch.tensor(self.sequences[idx], dtype=torch.long)
        return s[:-1], s[1:]


def collate_fn(batch):
    """Pad variable-length sequences to form a batch."""
    inputs, targets = zip(*batch)
    inp = pad_sequence(inputs,  batch_first=True, padding_value=PAD_IDX)
    tgt = pad_sequence(targets, batch_first=True, padding_value=PAD_IDX)
    mask = (inp == PAD_IDX)
    return inp, tgt, mask


train_ds = PlaylistDataset(train_seqs, VOCAB_LIMIT, MAX_SEQ_LEN, subset=TRAIN_SUBSET)
val_ds   = PlaylistDataset(val_seqs,   VOCAB_LIMIT, MAX_SEQ_LEN)
test_ds  = PlaylistDataset(test_seqs,  VOCAB_LIMIT, MAX_SEQ_LEN)

kw = dict(collate_fn=collate_fn, num_workers=0, pin_memory=(device.type == 'cuda'))
train_dl = DataLoader(train_ds, BATCH_SIZE, shuffle=True,  **kw)
val_dl   = DataLoader(val_ds,   BATCH_SIZE, shuffle=False, **kw)
test_dl  = DataLoader(test_ds,  BATCH_SIZE, shuffle=False, **kw)

print(f'Train: {len(train_ds):,} seqs  ({len(train_dl):,} batches)')
print(f'Val:   {len(val_ds):,} seqs  ({len(val_dl):,} batches)')
print(f'Test:  {len(test_ds):,} seqs  ({len(test_dl):,} batches)')


# ---
# ## 2. Model Architectures
# 
# Both architectures are defined in `src/models.py` and imported above.
# 
# ### 2.1 GRU Recommender
# 
# The GRU processes tracks sequentially, building a hidden state that captures listening context. It is inherently causal — no future leakage, no masking needed.
# 
# `Embedding -> Dropout -> GRU (2 layers, 256 hidden) -> Dropout -> Linear`
# 
# ### 2.2 Transformer Recommender
# 
# The Transformer processes all positions in parallel using multi-head self-attention with a causal mask. Learnable positional embeddings encode track order.
# 
# `Embedding + PosEmbedding -> Dropout -> TransformerEncoder (2 layers, 4 heads, causal) -> Linear`

# In[3]:


gru_model = GRURecommender(
    NUM_TOKENS, EMBED_DIM, HIDDEN_DIM, NUM_LAYERS, DROPOUT, PAD_IDX
).to(device)
print(f'GRU:         {sum(p.numel() for p in gru_model.parameters()):>12,} parameters')

transformer_model = TransformerRecommender(
    NUM_TOKENS, EMBED_DIM, NUM_HEADS, NUM_LAYERS, DROPOUT, MAX_SEQ_LEN, PAD_IDX
).to(device)
print(f'Transformer: {sum(p.numel() for p in transformer_model.parameters()):>12,} parameters')


# ---
# ## 3. Training
# 
# Both models are trained identically:
# - **Loss:** Cross-entropy with padding ignored
# - **Optimizer:** Adam (lr = 1e-3)
# - **Scheduler:** ReduceLROnPlateau — halves LR after 2 stagnant epochs
# - **Gradient clipping:** Max norm 1.0
# - **Checkpointing:** Best model (by validation loss) saved to `models/`

# In[4]:


def train_epoch(model, loader, optimizer, criterion, dev):
    model.train()
    total_loss = total_hit = total_tok = 0
    for inp, tgt, mask in loader:
        inp, tgt, mask = inp.to(dev), tgt.to(dev), mask.to(dev)
        logits = model(inp, pad_mask=mask)
        loss = criterion(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1))
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        valid = (tgt != PAD_IDX)
        n = valid.sum().item()
        total_loss += loss.item() * n
        total_hit  += (logits.argmax(-1)[valid] == tgt[valid]).sum().item()
        total_tok  += n
    return total_loss / total_tok, total_hit / total_tok


@torch.no_grad()
def eval_epoch(model, loader, criterion, dev):
    model.eval()
    total_loss = total_hit = total_tok = 0
    for inp, tgt, mask in loader:
        inp, tgt, mask = inp.to(dev), tgt.to(dev), mask.to(dev)
        logits = model(inp, pad_mask=mask)
        loss = criterion(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1))
        valid = (tgt != PAD_IDX)
        n = valid.sum().item()
        total_loss += loss.item() * n
        total_hit  += (logits.argmax(-1)[valid] == tgt[valid]).sum().item()
        total_tok  += n
    return total_loss / total_tok, total_hit / total_tok


def train_model(model, tr_dl, va_dl, epochs, lr, dev, name='model'):
    """Full training loop with validation and checkpointing."""
    opt   = torch.optim.Adam(model.parameters(), lr=lr)
    crit  = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=2, factor=0.5)
    hist  = {'tl': [], 'vl': [], 'ta': [], 'va': []}
    best  = float('inf')
    for ep in range(1, epochs + 1):
        t0 = time.time()
        tl, ta = train_epoch(model, tr_dl, opt, crit, dev)
        vl, va = eval_epoch(model, va_dl, crit, dev)
        sched.step(vl)
        hist['tl'].append(tl); hist['vl'].append(vl)
        hist['ta'].append(ta); hist['va'].append(va)
        tag = ''
        if vl < best:
            best = vl
            torch.save(model.state_dict(), f'../models/{name}_best.pt')
            tag = ' *'
        print(f'  Ep {ep:2d}/{epochs} | '
              f'Train {tl:.4f} / {ta:.4f} | '
              f'Val {vl:.4f} / {va:.4f} | '
              f'{time.time()-t0:.0f}s{tag}')
    return hist


def plot_curves(hist, title):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4))
    ep = range(1, len(hist['tl']) + 1)
    a1.plot(ep, hist['tl'], 'o-', ms=3, label='Train')
    a1.plot(ep, hist['vl'], 's-', ms=3, label='Val')
    a1.set(title=f'{title} — Loss', xlabel='Epoch', ylabel='CE Loss'); a1.legend()
    a2.plot(ep, hist['ta'], 'o-', ms=3, label='Train')
    a2.plot(ep, hist['va'], 's-', ms=3, label='Val')
    a2.set(title=f'{title} — Accuracy', xlabel='Epoch', ylabel='Accuracy'); a2.legend()
    plt.tight_layout()
    plt.savefig(OUT / f'train_{title.lower().replace(" ", "_")}.png', bbox_inches='tight')
    plt.close()


# ### 3.1 Train GRU

# In[5]:


print('Training GRU Recommender')
print('=' * 70)
gru_hist = train_model(gru_model, train_dl, val_dl, NUM_EPOCHS, LR, device, name='gru')
plot_curves(gru_hist, 'GRU')


# ### 3.2 Train Transformer

# In[ ]:


print('Training Transformer Recommender')
print('=' * 70)
tf_hist = train_model(transformer_model, train_dl, val_dl, NUM_EPOCHS, LR, device, name='transformer')
plot_curves(tf_hist, 'Transformer')


# ---
# ## 4. Experiments
# 
# We run ablation experiments to understand how key design choices affect performance. Each experiment trains a fresh GRU for 5 epochs on a 100K-sequence training subset.
# 
# ### 4.1 Sequence length
# How much playlist context does the model need? We compare max sequence lengths of 20, 50, and 100.
# 
# ### 4.2 Embedding dimension
# Does a richer track representation help? We compare embedding dimensions of 64, 128, and 256.

# In[ ]:


ABLATION_EPOCHS = 5
ABLATION_SUBSET = 100_000


def quick_train(model, max_len, subset=ABLATION_SUBSET):
    """Quick training run for ablation experiments."""
    ds_tr = PlaylistDataset(train_seqs, VOCAB_LIMIT, max_len, subset=subset)
    ds_va = PlaylistDataset(val_seqs, VOCAB_LIMIT, max_len)
    dl_tr = DataLoader(ds_tr, BATCH_SIZE, shuffle=True, collate_fn=collate_fn, num_workers=0)
    dl_va = DataLoader(ds_va, BATCH_SIZE, shuffle=False, collate_fn=collate_fn, num_workers=0)
    crit = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    opt  = torch.optim.Adam(model.parameters(), lr=LR)
    for _ in range(ABLATION_EPOCHS):
        train_epoch(model, dl_tr, opt, crit, device)
    vl, va = eval_epoch(model, dl_va, crit, device)
    return vl, va


print('Experiment 1: Sequence Length')
print('=' * 50)
seq_results = []
for ml in [20, 50, 100]:
    m = GRURecommender(NUM_TOKENS, EMBED_DIM, HIDDEN_DIM, NUM_LAYERS, DROPOUT, PAD_IDX).to(device)
    vl, va = quick_train(m, max_len=ml)
    print(f'  max_len={ml:3d}  val_loss={vl:.4f}  val_acc={va:.4f}')
    seq_results.append({'max_len': ml, 'val_loss': round(vl, 4), 'val_acc': round(va, 4)})

print('\nExperiment 2: Embedding Dimension')
print('=' * 50)
emb_results = []
for ed in [64, 128, 256]:
    m = GRURecommender(NUM_TOKENS, ed, ed * 2, NUM_LAYERS, DROPOUT, PAD_IDX).to(device)
    vl, va = quick_train(m, max_len=MAX_SEQ_LEN)
    params = sum(p.numel() for p in m.parameters())
    print(f'  embed_dim={ed:3d}  val_loss={vl:.4f}  val_acc={va:.4f}  params={params:,}')
    emb_results.append({'embed_dim': ed, 'val_loss': round(vl, 4), 'val_acc': round(va, 4), 'params': params})


# In[ ]:


fig, axes = plt.subplots(1, 2, figsize=(12, 4))

ax = axes[0]
sl = pd.DataFrame(seq_results)
bars = ax.bar(sl['max_len'].astype(str), sl['val_acc'], color='steelblue')
ax.set(title='Val Accuracy vs. Sequence Length', xlabel='Max Sequence Length', ylabel='Val Accuracy')
for bar, val in zip(bars, sl['val_acc']):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
            f'{val:.4f}', ha='center', va='bottom', fontsize=9)

ax = axes[1]
ed_df = pd.DataFrame(emb_results)
bars = ax.bar(ed_df['embed_dim'].astype(str), ed_df['val_acc'], color='darkorange')
ax.set(title='Val Accuracy vs. Embedding Dim', xlabel='Embedding Dimension', ylabel='Val Accuracy')
for bar, val in zip(bars, ed_df['val_acc']):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
            f'{val:.4f}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig(OUT / 'experiments_ablation.png', bbox_inches='tight')
plt.show()

print('Sequence length results:')
print(pd.DataFrame(seq_results).to_string(index=False))
print('\nEmbedding dimension results:')
print(pd.DataFrame(emb_results).to_string(index=False))


# ### 4.3 Experiment Analysis
# 
# **Sequence length:** Longer context windows give the model more information about the playlist's thematic trajectory. However, very long sequences may introduce noise from early, less relevant tracks. The results show the trade-off between richer context and diminishing returns.
# 
# **Embedding dimension:** Larger embeddings capture finer distinctions between tracks but increase model size and may overfit. The parameter counts show the computational cost, and the accuracy results indicate the optimal balance for this dataset.
# 
# These experiments guided the final hyperparameter selection used in the main training runs above.
