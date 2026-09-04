from __future__ import annotations

import torch
from torch.utils.data import Dataset

from src.data.preprocess import LABEL_COL, NARRATIVE_COL
from src.training.vocab import encode


class ComplaintDataset(Dataset):
    def __init__(self, df, vocab: dict[str, int], label2id: dict[str, int], max_len: int):
        self.texts = df[NARRATIVE_COL].tolist()
        self.labels = [label2id[l] for l in df[LABEL_COL].tolist()]
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        ids = encode(self.texts[idx], self.vocab, self.max_len)
        return torch.tensor(ids, dtype=torch.long), torch.tensor(self.labels[idx], dtype=torch.long)
