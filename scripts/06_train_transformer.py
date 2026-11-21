"""Fine-tune kz-transformers/kaz-roberta-conversational for Kazakh profanity.

Design notes
------------
* Input is the normalised text from ``kazprof.normalize`` (lowercased, homoglyphs
  folded, PII masked, repeated letters collapsed). Keeping the same normaliser at
  inference time is what prevents train/serve skew.
* Long comments: the tokenizer runs without truncation and the sequence is cut
  **head + tail** (94 + 32 subword tokens), because in reviews the profanity is
  often at the end ("...бірақ қызметі қотақ") while the context is at the start.
* Class imbalance (about 35/65) is handled with class weights in the loss.
* Chosen by validation macro-F1 for the profanity class; the test split is not
  touched here.

Run:  python scripts/06_train_transformer.py
"""
from __future__ import annotations

import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kazprof.normalize import normalize  # noqa: E402

MODEL_NAME = "kz-transformers/kaz-roberta-conversational"
SEED = 42
MAX_LEN = 128
HEAD_TOKENS, TAIL_TOKENS = 94, 32
EPOCHS = 4
BATCH = 16
LR = 2e-5
WEIGHT_DECAY = 0.01
WARMUP = 0.1

MODELS = ROOT / "models"
REPORTS = ROOT / "reports"
OUT_DIR = MODELS / "kazroberta_kk_profanity"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(False)


def load_rows(split: str) -> list[dict]:
    return [json.loads(l) for l in (ROOT / f"data/processed/{split}.jsonl").read_text(encoding="utf-8").splitlines()]


class ProfanityDataset(Dataset):
    def __init__(self, rows: list[dict], tok) -> None:
        self.rows = rows
        self.tok = tok
        self.cls_id = tok.cls_token_id if tok.cls_token_id is not None else tok.bos_token_id
        self.sep_id = tok.sep_token_id if tok.sep_token_id is not None else tok.eos_token_id

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int) -> dict:
        row = self.rows[i]
        ids = self.tok.encode(normalize(row["text"]).norm, add_special_tokens=False)
        if len(ids) > HEAD_TOKENS + TAIL_TOKENS:
            ids = ids[:HEAD_TOKENS] + ids[-TAIL_TOKENS:]
        ids = [self.cls_id] + ids + [self.sep_id]
        return {"ids": ids[:MAX_LEN], "label": row["label"]}


def collate(batch: list[dict], pad_id: int) -> dict:
    """Dynamic padding: pad to the longest sequence in the batch, not to 128.

    A fixed 128-token pad makes most batches 5x more expensive than necessary,
    because the median comment is ~25 subwords.
    """
    width = max(len(b["ids"]) for b in batch)
    input_ids, attn, labels = [], [], []
    for b in batch:
        pad = width - len(b["ids"])
        input_ids.append(b["ids"] + [pad_id] * pad)
        attn.append([1] * len(b["ids"]) + [0] * pad)
        labels.append(b["label"])
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attn, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }


@torch.no_grad()
def predict(model, loader) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    probs, labels = [], []
    for batch in loader:
        logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits
        probs.append(torch.softmax(logits, dim=-1)[:, 1].numpy())
        labels.append(batch["labels"].numpy())
    return np.concatenate(probs), np.concatenate(labels)


def f1_at(y_true, scores, t: float) -> float:
    pred = (scores >= t).astype(int)
    tp = int(((pred == 1) & (y_true == 1)).sum())
    fp = int(((pred == 1) & (y_true == 0)).sum())
    fn = int(((pred == 0) & (y_true == 1)).sum())
    return 2 * tp / max(1, 2 * tp + fp + fn)


def main() -> int:
    torch.set_num_threads(os.cpu_count() or 4)
    set_seed(SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(exist_ok=True)

    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

    train_rows, val_rows = load_rows("train"), load_rows("val")
    ds_tr = ProfanityDataset(train_rows, tok)
    ds_va = ProfanityDataset(val_rows, tok)
    pad_id = tok.pad_token_id
    dl_tr = DataLoader(ds_tr, batch_size=BATCH, shuffle=True, num_workers=0,
                       collate_fn=lambda b: collate(b, pad_id))
    dl_va = DataLoader(ds_va, batch_size=64, shuffle=False, num_workers=0,
                       collate_fn=lambda b: collate(b, pad_id))

    n_pos = sum(r["label"] for r in train_rows)
    n_neg = len(train_rows) - n_pos
    weights = torch.tensor([len(train_rows) / (2 * n_neg), len(train_rows) / (2 * n_pos)], dtype=torch.float)
    loss_fn = torch.nn.CrossEntropyLoss(weight=weights)

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    total_steps = len(dl_tr) * EPOCHS
    sched = get_linear_schedule_with_warmup(opt, int(WARMUP * total_steps), total_steps)

    log = {"model": MODEL_NAME, "seed": SEED, "max_len": MAX_LEN, "head_tail": [HEAD_TOKENS, TAIL_TOKENS],
           "epochs": EPOCHS, "batch": BATCH, "lr": LR, "weight_decay": WEIGHT_DECAY,
           "class_weights": weights.tolist(), "train_rows": len(train_rows), "epochs_log": []}
    best = (-1.0, None)
    t_start = time.time()
    for epoch in range(1, EPOCHS + 1):
        model.train()
        running = 0.0
        for step, batch in enumerate(dl_tr, 1):
            out = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
            loss = loss_fn(out.logits, batch["labels"])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad()
            running += float(loss)
            if step % 25 == 0:
                print(f"epoch {epoch} step {step}/{len(dl_tr)} loss {running / step:.4f}", flush=True)

        probs, labels = predict(model, dl_va)
        best_t, best_f1 = max(((t, f1_at(labels, probs, t)) for t in np.arange(0.1, 0.91, 0.05)), key=lambda x: x[1])
        entry = {"epoch": epoch, "train_loss": round(running / len(dl_tr), 4),
                 "val_f1_at_0.5": round(f1_at(labels, probs, 0.5), 4),
                 "val_best_f1": round(best_f1, 4), "val_best_threshold": round(float(best_t), 2),
                 "elapsed_s": round(time.time() - t_start, 1)}
        log["epochs_log"].append(entry)
        print(json.dumps(entry), flush=True)
        if best_f1 > best[0]:
            best = (best_f1, epoch)
            model.save_pretrained(OUT_DIR)
            tok.save_pretrained(OUT_DIR)

    log["best_val_f1"] = round(best[0], 4)
    log["best_epoch"] = best[1]
    log["train_seconds"] = round(time.time() - t_start, 1)
    (REPORTS / "transformer_train.json").write_text(json.dumps(log, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(log, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
