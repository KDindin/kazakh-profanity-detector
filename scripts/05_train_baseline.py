"""Baseline: TF-IDF (word + char n-grams) + Logistic Regression.

The baseline exists to show what a classical model achieves on this data and to
give the transformer something to beat. Everything learned (vocabulary, IDF,
coefficients, threshold) is fitted on the training split only.

Run:  python scripts/05_train_baseline.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_recall_fscore_support
from sklearn.pipeline import FeatureUnion, Pipeline

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kazprof.normalize import normalize  # noqa: E402

SEED = 42
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"


def load(split: str) -> tuple[list[str], np.ndarray]:
    rows = [json.loads(l) for l in (ROOT / f"data/processed/{split}.jsonl").read_text(encoding="utf-8").splitlines()]
    return [normalize(r["text"]).norm for r in rows], np.array([r["label"] for r in rows])


def best_threshold(y_true, scores) -> tuple[float, float]:
    best_t, best_f1 = 0.5, -1.0
    for t in np.arange(0.05, 0.96, 0.01):
        f1 = f1_score(y_true, (scores >= t).astype(int), pos_label=1, zero_division=0)
        if f1 > best_f1:
            best_t, best_f1 = float(t), float(f1)
    return best_t, best_f1


def main() -> int:
    MODELS.mkdir(exist_ok=True)
    REPORTS.mkdir(exist_ok=True)
    Xtr, ytr = load("train")
    Xva, yva = load("val")

    features = FeatureUnion(
        [
            ("word", TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
            ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2, sublinear_tf=True, max_features=120000)),
        ]
    )
    grid = [0.5, 1.0, 2.0, 4.0, 8.0]
    results = []
    best = None
    for C in grid:
        pipe = Pipeline(
            [
                ("features", features),
                ("clf", LogisticRegression(C=C, class_weight="balanced", max_iter=3000, solver="liblinear", random_state=SEED)),
            ]
        )
        pipe.fit(Xtr, ytr)
        scores = pipe.predict_proba(Xva)[:, 1]
        t, f1 = best_threshold(yva, scores)
        results.append({"C": C, "threshold": t, "val_f1": round(f1, 4)})
        if best is None or f1 > best[0]:
            best = (f1, C, t, pipe, scores)

    val_f1, C, threshold, pipe, scores = best
    pred = (scores >= threshold).astype(int)
    p, r, f, _ = precision_recall_fscore_support(yva, pred, labels=[1], zero_division=0)
    metrics = {
        "model": "tfidf_word_char + logistic_regression",
        "seed": SEED,
        "grid": results,
        "chosen": {"C": C, "threshold": threshold},
        "val": {
            "n": int(len(yva)),
            "precision": round(float(p[0]), 4),
            "recall": round(float(r[0]), 4),
            "f1": round(float(f[0]), 4),
            "confusion_matrix": confusion_matrix(yva, pred, labels=[0, 1]).tolist(),
            "report": classification_report(yva, pred, labels=[0, 1], digits=4, zero_division=0),
        },
    }
    joblib.dump({"pipeline": pipe, "threshold": threshold}, MODELS / "baseline_tfidf_lr.joblib")
    (REPORTS / "baseline_val.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in metrics.items() if k != "val"} | {"val": {kk: vv for kk, vv in metrics["val"].items() if kk != "report"}}, ensure_ascii=False, indent=2))
    print(metrics["val"]["report"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
