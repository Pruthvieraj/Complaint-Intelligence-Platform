"""Minimal word-level vocabulary + encoder for the sandbox from-scratch
transformer (see train_sandbox_transformer.py). The production path
(train_transformer.py) uses a real HuggingFace subword tokenizer
instead — this exists only because the sandbox this repo was built in
has no network access to download one."""
from __future__ import annotations

import re
from collections import Counter

PAD, UNK = "<pad>", "<unk>"
_token_re = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _token_re.findall(text.lower())


def build_vocab(texts, max_vocab_size: int = 12_000) -> dict[str, int]:
    counter = Counter()
    for t in texts:
        counter.update(tokenize(t))
    most_common = counter.most_common(max_vocab_size - 2)
    vocab = {PAD: 0, UNK: 1}
    for word, _ in most_common:
        vocab[word] = len(vocab)
    return vocab


def encode(text: str, vocab: dict[str, int], max_len: int) -> list[int]:
    ids = [vocab.get(tok, vocab[UNK]) for tok in tokenize(text)][:max_len]
    ids = ids + [vocab[PAD]] * (max_len - len(ids))
    return ids
