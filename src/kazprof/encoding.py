"""Turn a comment into model input, once, for both training and inference.

Keeping this in one place is the practical way to avoid train/serve skew: the
training script and the predictor must cut long comments identically.
"""
from __future__ import annotations

from .normalize import normalize


def encode_head_tail(tokenizer, text: str, max_len: int = 128, head: int = 94, tail: int = 32) -> dict:
    cls_id = tokenizer.cls_token_id if tokenizer.cls_token_id is not None else tokenizer.bos_token_id
    sep_id = tokenizer.sep_token_id if tokenizer.sep_token_id is not None else tokenizer.eos_token_id
    ids = tokenizer.encode(normalize(text).norm, add_special_tokens=False)
    truncated = len(ids) > head + tail
    if truncated:
        ids = ids[:head] + ids[-tail:]
    ids = [cls_id] + ids + [sep_id]
    pad = max(0, max_len - len(ids))
    return {
        "input_ids": ids[:max_len] + [tokenizer.pad_token_id] * pad,
        "attention_mask": [1] * min(len(ids), max_len) + [0] * pad,
        "truncated": truncated,
        "n_subwords": len(ids),
    }


def describe_truncation(text: str, max_len: int = 128, head: int = 94, tail: int = 32) -> str:
    """Human-readable explanation used by the CLI --explain flag."""
    n = normalize(text)
    return (
        f"normalised={n.norm!r} lang={n.lang} "
        f"tokens={len(n.tokens)} (head {head} + tail {tail} subword budget)"
    )
