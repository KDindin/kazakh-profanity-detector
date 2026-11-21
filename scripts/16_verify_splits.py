"""One-off check: no near-duplicate pair crosses a split boundary."""
import json
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

ROOT = Path(__file__).resolve().parents[1]
texts, splits = [], []
for split in ("train", "val", "test"):
    for line in (ROOT / f"data/processed/{split}.jsonl").read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        texts.append(r["text"])
        splits.append(split)

vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1, max_features=60000)
X = vec.fit_transform(texts)
X = X.multiply(1.0 / np.sqrt(X.multiply(X).sum(axis=1))).tocsr()
sim = (X @ X.T).toarray()
np.fill_diagonal(sim, 0.0)
splits = np.array(splits)
cross = np.zeros_like(sim, dtype=bool)
for s in ("train", "val", "test"):
    for t in ("train", "val", "test"):
        if s >= t:
            continue
        m = np.outer(splits == s, splits == t)
        cross |= m
vals = sim[cross]
print(f"texts {len(texts)} | cross-split pairs {vals.size} | max similarity {vals.max():.3f} | pairs >= 0.85: {(vals >= 0.85).sum()}")
