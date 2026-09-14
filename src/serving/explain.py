"""
Phase 8 — model explainability.

A lightweight, model-agnostic proxy for SHAP/LIME: leave-one-word-out
occlusion. For each word in the input, re-run the classifier with just
that word masked out and measure how much the predicted class's
probability drops. A big drop means that word was doing a lot of work
for the prediction; a small (or negative) drop means it barely
mattered, or the model got *more* confident without it — itself a
useful signal about spurious cues the model latched onto.

This is deliberately not a "real" SHAP/attention-based explanation:
the production model is a plain PyTorch nn.Module exported to ONNX
(see docs/... and src/optimization/export_onnx.py), so there's no
HuggingFace attention API to read weights off of, and Shapley-value
attribution would need a library dependency and combinatorial sampling
this project's serving image deliberately avoids shipping (see
requirements-serve.txt's comment on keeping the deployed image
minimal). Occlusion needs nothing but the classifier that's already
loaded, is trivially fast (one batched ONNX call), and is honest about
what it is: a single-feature ablation, exactly the method LIME's
"perturb and see what changes" idea is built on, minus LIME's local
surrogate-model-fitting step.
"""
from __future__ import annotations

import re

import numpy as np

from src.training.vocab import encode, tokenize

_WORD_RE = re.compile(r"[A-Za-z0-9]+")


def explain(text: str, classifier, max_words: int = 60) -> dict:
    baseline = classifier.predict(text, top_k=1)
    predicted_label = baseline["predicted_category"]
    label2id = {v: k for k, v in classifier.id2label.items()}
    target_idx = label2id[predicted_label]

    vocab, max_len = classifier.vocab, classifier.max_len
    unk_id = vocab.get("<unk>", 1)

    # tokenize() applies the same "runs of [a-z0-9]" regex (case-folded)
    # that _WORD_RE applies to the original text, so spans and tokens
    # line up 1:1 in order and count — this is what lets us highlight
    # importance back onto the *original*, un-lowercased text.
    spans = list(_WORD_RE.finditer(text))
    tokens = tokenize(text)
    n = min(len(spans), len(tokens), max_len, max_words)

    if n == 0:
        return {
            "predicted_category": predicted_label,
            "confidence": baseline["confidence"],
            "method": "leave_one_word_out_occlusion",
            "words": [],
        }

    base_ids = encode(text, vocab, max_len)
    batch = np.tile(np.array(base_ids, dtype=np.int64), (n + 1, 1))
    for i in range(n):
        batch[i + 1, i] = unk_id  # row 0 is left untouched as the baseline

    logits = classifier.session.run(None, {"input_ids": batch})[0]
    exps = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs = exps / exps.sum(axis=1, keepdims=True)
    target_probs = probs[:, target_idx]
    baseline_prob = float(target_probs[0])

    words = []
    for i in range(n):
        drop = baseline_prob - float(target_probs[i + 1])
        words.append(
            {
                "word": spans[i].group(0),
                "start": spans[i].start(),
                "end": spans[i].end(),
                "importance": round(drop, 4),
            }
        )

    max_abs = max((abs(w["importance"]) for w in words), default=0.0) or 1.0
    for w in words:
        w["importance_normalized"] = round(w["importance"] / max_abs, 4)

    return {
        "predicted_category": predicted_label,
        "confidence": baseline["confidence"],
        "method": "leave_one_word_out_occlusion",
        "words": words,
    }
