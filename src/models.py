"""
models.py
---------
Model architectures for sequential next-track prediction.

Both models share the same interface:
    Input:  (batch, seq_len) tensor of track indices
    Output: (batch, seq_len, num_tokens) logits over vocabulary

Used by notebooks/Modeling.ipynb and notebooks/Evaluation.ipynb.
"""

import torch
import torch.nn as nn


class GRURecommender(nn.Module):
    """GRU-based sequential recommender.

    Architecture: Embedding -> Dropout -> GRU -> Dropout -> Linear
    """

    def __init__(self, num_tokens, embed_dim, hidden_dim, num_layers, dropout, pad_idx):
        super().__init__()
        self.embedding = nn.Embedding(num_tokens, embed_dim, padding_idx=pad_idx)
        self.gru = nn.GRU(
            embed_dim, hidden_dim, num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, num_tokens)

    def forward(self, x, pad_mask=None):
        emb = self.dropout(self.embedding(x))
        out, _ = self.gru(emb)
        return self.fc(self.dropout(out))


class TransformerRecommender(nn.Module):
    """Transformer-based sequential recommender with causal masking.

    Architecture: Embedding + PosEmbedding -> Dropout -> TransformerEncoder (causal) -> Linear
    """

    def __init__(self, num_tokens, embed_dim, num_heads, num_layers, dropout, max_len, pad_idx):
        super().__init__()
        self.embedding = nn.Embedding(num_tokens, embed_dim, padding_idx=pad_idx)
        self.pos_embedding = nn.Embedding(max_len, embed_dim)
        self.dropout = nn.Dropout(dropout)
        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads,
            dim_feedforward=embed_dim * 4, dropout=dropout, batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers)
        self.fc = nn.Linear(embed_dim, num_tokens)

    def forward(self, x, pad_mask=None):
        B, L = x.shape
        pos = torch.arange(L, device=x.device).unsqueeze(0)
        emb = self.dropout(self.embedding(x) + self.pos_embedding(pos))
        causal = torch.triu(torch.full((L, L), float('-inf'), device=x.device), diagonal=1)
        out = self.encoder(emb, mask=causal, src_key_padding_mask=pad_mask)
        return self.fc(out)
