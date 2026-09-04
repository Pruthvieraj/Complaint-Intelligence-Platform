"""A small Transformer encoder classifier, trained from scratch.

This is deliberately NOT DistilBERT. It's the sandbox stand-in used to
actually execute Phase 2 end-to-end in an environment with no network
access to the HuggingFace Hub (see train_sandbox_transformer.py's
module docstring and the README's "Sandbox execution note" for why).
It's architecturally a real transformer encoder (multi-head
self-attention + positional embeddings + feed-forward blocks) so it
genuinely exercises the same training/tracking/registry/ONNX/serving
machinery a real DistilBERT fine-tune would — it just starts from
random weights instead of pretrained ones, which is why its absolute
accuracy is a weaker signal than a real fine-tune's would be. The
production script (train_transformer.py) is the one that produces a
number worth putting on a resume.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 256):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class SmallTransformerClassifier(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_classes: int,
        d_model: int = 96,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        max_len: int = 128,
        dropout: float = 0.1,
        pad_idx: int = 0,
    ):
        super().__init__()
        self.pad_idx = pad_idx
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.pos_encoding = PositionalEncoding(d_model, max_len=max_len)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        pad_mask = input_ids == self.pad_idx  # (B, T), True where padded
        x = self.embedding(input_ids) * math.sqrt(self.embedding.embedding_dim)
        x = self.pos_encoding(x)
        x = self.encoder(x, src_key_padding_mask=pad_mask)

        # mean-pool over non-pad tokens
        mask = (~pad_mask).unsqueeze(-1).float()
        summed = (x * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1.0)
        pooled = summed / counts

        pooled = self.dropout(pooled)
        return self.classifier(pooled)
