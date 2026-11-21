"""Explain a baseline decision: which n-grams pushed the score up or down?

    python scripts/13_explain_features.py "Уй мал быля, қораға қайтшы"
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from kazprof.infer import Detector  # noqa: E402
from kazprof.normalize import normalize  # noqa: E402


def explain(text: str, top: int = 8) -> None:
    det = Detector(backend="baseline")
    pipe = det._baseline
    feats = pipe.named_steps["features"]
    clf = pipe.named_steps["clf"]
    X = feats.transform([normalize(text).norm])
    names = feats.get_feature_names_out()
    contrib = X.multiply(clf.coef_[0]).tocsr()
    row = contrib.getrow(0).toarray().ravel()
    order = np.argsort(row)
    print(f"text: {text}")
    print(f"P(profanity) = {det.score(text):.4f} (bias {clf.intercept_[0]:+.3f})")
    print("  pushes toward 1:")
    for i in order[::-1][:top]:
        if row[i] <= 0:
            break
        print(f"    {names[i]!r:28s} x={X[0, i]:.3f}  w={clf.coef_[0][i]:+.3f}  +{row[i]:.3f}")
    print("  pushes toward 0:")
    for i in order[:top]:
        if row[i] >= 0:
            break
        print(f"    {names[i]!r:28s} x={X[0, i]:.3f}  w={clf.coef_[0][i]:+.3f}  {row[i]:.3f}")
    print()


if __name__ == "__main__":
    for t in (sys.argv[1:] or ["Уй мал быля, қораға қайтшы"]):
        explain(t)
