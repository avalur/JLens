"""Corpora. DEFAULT_TEXTS is enough for validation; wiki_texts()/stream_batches() build a
paper-scale (e.g. 128 x 64-token) averaging corpus from the locally-cached English Wikipedia
(Arrow, read via pyarrow -- no `datasets` install needed)."""

from __future__ import annotations

import glob
import os

import torch

DEFAULT_TEXTS = [
    "The capital of France is Paris, a city on the river Seine.",
    "Water is made of hydrogen and oxygen atoms bonded together tightly.",
    "The Eiffel Tower is a famous landmark located in the French capital.",
    "Mount Everest is the tallest mountain above sea level on planet Earth.",
    "William Shakespeare wrote many famous plays including Hamlet and Macbeth.",
    "The sun is a star at the center of our solar system today.",
    "Photosynthesis allows green plants to convert sunlight into chemical energy.",
    "The Great Wall of China stretches for thousands of kilometers across the land.",
]


def fixed_batches(tokenizer, texts=None, seq_len: int = 16, device: str = "cpu"):
    """Tokenize to fixed-length [1, seq_len] batches (no padding); skip too-short texts."""
    texts = texts or DEFAULT_TEXTS
    out = []
    for t in texts:
        ids = tokenizer(t, return_tensors="pt").input_ids
        if ids.shape[1] < seq_len:
            continue
        out.append(ids[:, :seq_len].to(device))
    return out


def wiki_texts(n_docs: int = 200, lang: str = "en"):
    """Read article text from the locally-cached HuggingFace Wikipedia dump (Arrow shard 0)."""
    import pyarrow as pa

    pattern = os.path.expanduser(
        f"~/.cache/huggingface/datasets/wikipedia/20220301.{lang}/*/*/"
        "wikipedia-train-00000-of-*.arrow"
    )
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"no cached wikipedia shard at {pattern}")
    texts = []
    with pa.memory_map(files[0], "r") as src:
        try:
            reader = pa.ipc.open_stream(src)
        except Exception:
            src.seek(0)
            reader = pa.ipc.open_file(src)
        for batch in reader:
            texts += batch.column("text").to_pylist()
            if len(texts) >= n_docs:
                break
    return texts[:n_docs]


def stream_batches(tokenizer, texts, n_frag=128, seq_len=64, per_doc_chars=1000,
                   micro_bs=16, device="cpu"):
    """Build n_frag fixed-length fragments of seq_len tokens by concatenating token streams
    from many docs (capped per doc for diversity). Returns a list of [micro_bs, seq_len] batches."""
    need = n_frag * seq_len
    ids = []
    for t in texts:
        ids += tokenizer(t[:per_doc_chars], add_special_tokens=False).input_ids
        if len(ids) >= need:
            break
    frags = [ids[i:i + seq_len] for i in range(0, need, seq_len)]
    frags = [f for f in frags if len(f) == seq_len][:n_frag]
    batches = [torch.tensor(frags[i:i + micro_bs]).to(device)
               for i in range(0, len(frags), micro_bs)]
    return batches
