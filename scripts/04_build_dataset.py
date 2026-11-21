"""Build the final dataset: balance, deduplicate, cluster near-duplicates, split.

Run:  python scripts/04_build_dataset.py
Outputs:
  data/processed/{train,val,test}.jsonl
  data/processed/dataset_stats.json
"""
from __future__ import annotations

import csv
import hashlib
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kazprof.normalize import normalize  # noqa: E402
from kazprof.synthetic import generate  # noqa: E402

SEED = 42
TARGET_POS, TARGET_NEG = 700, 1300
NEAR_DUP_SIM = 0.85
SPLIT = (0.6, 0.2, 0.2)
PROCESSED = ROOT / "data" / "processed"


class UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def cluster_near_duplicates(texts: list[str], threshold: float = NEAR_DUP_SIM) -> list[int]:
    """Union-find over char n-gram cosine similarity."""
    if len(texts) > 6000:  # safety: dense similarity is O(n^2)
        texts = texts[:6000]
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1, max_features=60000)
    X = vec.fit_transform(texts)
    X = X.multiply(1.0 / np.sqrt(X.multiply(X).sum(axis=1))).tocsr()
    sim = (X @ X.T).toarray()
    uf = UnionFind(len(texts))
    rows, cols = np.where(sim >= threshold)
    for i, j in zip(rows, cols):
        if i < j:
            uf.union(int(i), int(j))
    return [uf.find(i) for i in range(len(texts))]


def main() -> int:
    PROCESSED.mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED)

    pool = list(csv.DictReader((ROOT / "data/interim/pool.csv").open(encoding="utf-8")))
    rows: list[dict] = []

    def make(text: str, label: int, source: str, origin: str, synthetic: bool,
             template_id: str = "", scope: str | None = None) -> dict:
        n = normalize(text)
        return {
            "id": hashlib.sha1(f"{source}|{n.squash[:400]}".encode()).hexdigest()[:12],
            "text": text,
            "label": label,
            "source": source,
            "origin": origin,
            "synthetic": synthetic,
            "template_id": template_id,
            # 'scope' records the language the row was admitted under. Synthetic
            # mixed frames and the seed sentences are mixed by construction and
            # are declared as such; detected lang is kept for analysis.
            "lang": scope or n.lang,
            "lang_detected": n.lang,
        }

    # ---- real rows ------------------------------------------------------
    for p in pool:
        if p["lang"] not in {"kk", "mixed"}:
            continue
        scope = None
        if p["source"] == "swear_sentences_seed":
            scope = "mixed"  # Kazakh profanity written in Russian orthography
        rows.append(
            make(p["text"], int(p["rule_label"]), p["source"], p["origin"],
                 synthetic=False, scope=scope)
        )

    # ---- synthetic rows -------------------------------------------------
    real_pos = [r for r in rows if r["label"] == 1]
    n_syn_pos = max(0, TARGET_POS - len(real_pos))
    for g in generate(n_syn_pos, 40, seed=SEED):
        rows.append(
            make(g.text, g.label, "synthetic_template", "synthetic", True,
                 g.template_id, scope="mixed" if g.template_id.startswith("mix") else "kk")
        )

    # ---- exact dedupe ---------------------------------------------------
    seen: set[str] = set()
    unique: list[dict] = []
    for r in rows:
        key = normalize(r["text"]).squash[:300]
        if key in seen:
            continue
        seen.add(key)
        unique.append(r)
    rows = unique

    # ---- balance negatives ---------------------------------------------
    pos = [r for r in rows if r["label"] == 1]
    neg = [r for r in rows if r["label"] == 0]
    by_src: dict[str, list[dict]] = defaultdict(list)
    for r in neg:
        by_src[r["source"]].append(r)
    for k in by_src:
        rng.shuffle(by_src[k])

    chosen_neg: list[dict] = []
    for src, cap in [
        ("kaspi_reviews", 999),
        ("2gis_reviews", 999),
        ("synthetic_template", 999),
    ]:
        chosen_neg.extend(by_src.get(src, [])[:cap])
    remaining = TARGET_NEG - len(chosen_neg)
    chosen_neg.extend(by_src.get("telegram_kk", [])[:max(0, remaining)])
    rng.shuffle(chosen_neg)
    rows = pos + chosen_neg

    # ---- cluster near duplicates ---------------------------------------
    cluster = cluster_near_duplicates([r["text"] for r in rows])
    for r, c in zip(rows, cluster):
        r["cluster"] = c

    # ---- cluster-aware stratified split --------------------------------
    groups: dict[int, list[dict]] = defaultdict(list)
    for r, c in zip(rows, cluster):
        groups[c].append(r)
    items = list(groups.values())
    rng.shuffle(items)

    strata = [(lab, syn) for lab in (0, 1) for syn in (False, True)]
    total = Counter((r["label"], r["synthetic"]) for r in rows)
    target = {(lab, syn): SPLIT for lab, syn in strata}
    counts = {name: Counter() for name in ("train", "val", "test")}
    assigned: dict[str, list[dict]] = {"train": [], "val": [], "test": []}

    for grp in items:
        lab, syn = grp[0]["label"], grp[0]["synthetic"]
        # every row of a cluster shares the label by construction of the task
        best, best_score = "train", -1e9
        for name, share in zip(("train", "val", "test"), SPLIT):
            want = total[(lab, syn)] * share
            score = want - counts[name][(lab, syn)]
            if name == "train":
                score *= 1.0
            if score > best_score:
                best, best_score = name, score
        for r in grp:
            counts[best][(lab, syn)] += 1
        assigned[best].extend(grp)

    stats = {
        "seed": SEED,
        "near_dup_threshold": NEAR_DUP_SIM,
        "total": len(rows),
        "labels": dict(Counter(r["label"] for r in rows)),
        "synthetic_rows": sum(1 for r in rows if r["synthetic"]),
        "synthetic_positives": sum(1 for r in rows if r["synthetic"] and r["label"] == 1),
        "by_source": dict(Counter(r["source"] for r in rows)),
        "by_lang": dict(Counter(r["lang"] for r in rows)),
        "clusters": len({r["cluster"] for r in rows}),
        "splits": {},
    }

    for name, data in assigned.items():
        rng.shuffle(data)
        with (PROCESSED / f"{name}.jsonl").open("w", encoding="utf-8") as f:
            for r in data:
                f.write(json.dumps({k: v for k, v in r.items() if k != "cluster"}, ensure_ascii=False) + "\n")
        stats["splits"][name] = {
            "rows": len(data),
            "positives": sum(1 for r in data if r["label"] == 1),
            "synthetic": sum(1 for r in data if r["synthetic"]),
            "real_positives": sum(1 for r in data if r["label"] == 1 and not r["synthetic"]),
            "by_source": dict(Counter(r["source"] for r in data)),
        }

    (PROCESSED / "dataset_stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
