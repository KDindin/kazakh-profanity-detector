"""Post-hoc analysis: what would a severe-term rule override change?

This is **not** part of the shipped model and was not used for model selection.
It answers the obvious follow-up question after the error analysis ("the baseline
missed 'бля', 'пизда', 'пидор' in long comments - would a rule fix it?") with
numbers instead of an opinion.

Run after 07:  python scripts/12_rule_override_analysis.py
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from kazprof.lexicon import Lexicon  # noqa: E402

rows = list(csv.DictReader((ROOT / "reports/predictions_test.csv").open(encoding="utf-8")))
lex = Lexicon()

y, model_pred, override_pred = [], [], []
changed = []
for r in rows:
    label = int(r["label"])
    pred = int(r["pred"])
    hits = lex.hits_text(r["text"])
    severe = any(h.severity == "severe" for h in hits)
    ov = 1 if (pred == 1 or severe) else 0
    y.append(label)
    model_pred.append(pred)
    override_pred.append(ov)
    if ov != pred:
        changed.append({"text": r["text"], "label": label, "model": pred, "override": ov,
                        "hits": [h.term_id for h in hits]})


def block(pred):
    p, r, f, _ = precision_recall_fscore_support(y, pred, labels=[1], zero_division=0)
    return {"precision": round(float(p[0]), 4), "recall": round(float(r[0]), 4),
            "f1": round(float(f[0]), 4),
            "confusion_matrix": confusion_matrix(y, pred, labels=[0, 1]).tolist()}


out = {
    "n": len(y),
    "shipped_model": block(model_pred),
    "with_severe_term_override": block(override_pred),
    "rows_changed": len(changed),
    "changed_examples": changed[:15],
    "note": "post-hoc analysis only; the shipped model and its reported metrics are unchanged",
}
(ROOT / "reports/rule_override_analysis.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps({k: v for k, v in out.items() if k != "changed_examples"}, ensure_ascii=False, indent=2))
