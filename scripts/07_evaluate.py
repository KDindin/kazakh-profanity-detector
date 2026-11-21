"""Final evaluation: tune on validation, evaluate once on the held-out test split.

Run after both models are trained:  python scripts/07_evaluate.py
Outputs: reports/metrics.json, reports/confusion_matrix.png,
         reports/predictions_test.csv, reports/threshold_table.csv
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kazprof.infer import Detector  # noqa: E402

REPORTS = ROOT / "reports"


def load(split: str) -> list[dict]:
    return [json.loads(l) for l in (ROOT / f"data/processed/{split}.jsonl").read_text(encoding="utf-8").splitlines()]


def scores_for(det: Detector, rows: list[dict]) -> np.ndarray:
    return np.array([det.score(r["text"]) for r in rows])


def tune_threshold(y: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    grid = np.arange(0.05, 0.96, 0.01)
    f1s = [f1_score(y, (scores >= t).astype(int), pos_label=1, zero_division=0) for t in grid]
    i = int(np.argmax(f1s))
    return round(float(grid[i]), 2), float(f1s[i])


def block(y: np.ndarray, scores: np.ndarray, t: float) -> dict:
    pred = (scores >= t).astype(int)
    p, r, f, _ = precision_recall_fscore_support(y, pred, labels=[1], zero_division=0)
    return {
        "n": int(len(y)),
        "precision": round(float(p[0]), 4),
        "recall": round(float(r[0]), 4),
        "f1": round(float(f[0]), 4),
        "accuracy": round(float(accuracy_score(y, pred)), 4),
        "confusion_matrix": confusion_matrix(y, pred, labels=[0, 1]).tolist(),
        "report": classification_report(y, pred, labels=[0, 1], digits=4, zero_division=0),
    }


def main() -> int:
    REPORTS.mkdir(exist_ok=True)
    val, test = load("val"), load("test")
    y_val = np.array([r["label"] for r in val])
    y_test = np.array([r["label"] for r in test])

    out: dict = {"threshold_source": "validation split only", "models": {}}
    trained = {}

    for name in ("baseline", "transformer"):
        try:
            det = Detector(backend="baseline" if name == "baseline" else "transformer")
        except Exception as exc:  # model not available
            out["models"][name] = {"error": str(exc)}
            continue
        s_val = scores_for(det, val)
        t, f1_val = tune_threshold(y_val, s_val)
        s_test = scores_for(det, test)
        res = {
            "backend": det.backend,
            "threshold_from_val": t,
            "val_f1_at_threshold": round(f1_val, 4),
            "val": block(y_val, s_val, t),
            "test": block(y_test, s_test, t),
            "test_real_only": block(
                np.array([r["label"] for r in test if not r["synthetic"]]),
                s_test[[i for i, r in enumerate(test) if not r["synthetic"]]],
                t,
            ),
            "test_synthetic_only": block(
                np.array([r["label"] for r in test if r["synthetic"]]),
                s_test[[i for i, r in enumerate(test) if r["synthetic"]]],
                t,
            ),
        }
        out["models"][name] = res
        trained[name] = (det, t, s_test)

    # pick the final model on validation F1, never on test.
    # Tie-break (stated before looking at test scores): if the two candidates are
    # within 0.01 validation F1, prefer the smaller, faster, fully explainable model.
    scored = {k: v["val_f1_at_threshold"] for k, v in out["models"].items() if "error" not in v}
    final = max(scored, key=scored.get) if scored else None
    if scored.get("baseline") is not None and scored.get("transformer") is not None:
        if abs(scored["baseline"] - scored["transformer"]) < 0.01:
            final = "baseline"
    out["final_model"] = final

    if final:
        det, t, s_test = trained[final]
        pred = (s_test >= t).astype(int)
        rows = []
        for r, p, s in zip(test, pred, s_test):
            rows.append({
                "id": r["id"], "label": r["label"], "pred": int(p), "score": round(float(s), 4),
                "source": r["source"], "synthetic": r["synthetic"], "lang": r["lang"],
                "text": r["text"],
            })
        with (REPORTS / "predictions_test.csv").open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

        # threshold trade-off table used in the error analysis section
        table = []
        for tt in np.arange(0.2, 0.86, 0.05):
            pr = (s_test >= tt).astype(int)
            p, r, f, _ = precision_recall_fscore_support(y_test, pr, labels=[1], zero_division=0)
            table.append({"threshold": round(float(tt), 2), "precision": round(float(p[0]), 4),
                          "recall": round(float(r[0]), 4), "f1": round(float(f[0]), 4),
                          "flagged": int(pr.sum())})
        with (REPORTS / "threshold_table.csv").open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(table[0].keys()))
            w.writeheader()
            w.writerows(table)
        out["threshold_table"] = table

        # confusion matrix picture
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            cm = confusion_matrix(y_test, (s_test >= t).astype(int), labels=[0, 1])
            fig, ax = plt.subplots(figsize=(4.2, 3.6), dpi=150)
            im = ax.imshow(cm, cmap="Blues")
            for (i, j), v in np.ndenumerate(cm):
                ax.text(j, i, str(v), ha="center", va="center", color="black", fontsize=13)
            ax.set_xticks([0, 1], ["pred 0", "pred 1"])
            ax.set_yticks([0, 1], ["true 0", "true 1"])
            ax.set_title(f"{final} @ threshold {t:.2f}")
            fig.colorbar(im, ax=ax, fraction=0.046)
            fig.tight_layout()
            fig.savefig(REPORTS / "confusion_matrix.png")
        except Exception as exc:
            out["confusion_matrix_png_error"] = str(exc)

    (REPORTS / "metrics.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    if final:
        (ROOT / "models" / "final_model.json").write_text(
            json.dumps(
                {
                    "final_model": final,
                    "threshold": out["models"][final]["threshold_from_val"],
                    "selection_rule": "highest F1 for class 1 on the validation split; if the two candidates are within 0.01, the smaller/faster model wins; test evaluated once afterwards",
                    "candidates": scored,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        (ROOT / "models" / "threshold.json").write_text(
            json.dumps(
                {
                    "threshold": out["models"][final]["threshold_from_val"],
                    "backend": final,
                    "tuned_on": "validation split",
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
    for name, res in out["models"].items():
        if "error" in res:
            print(name, "ERROR", res["error"])
            continue
        print(f"{name}: threshold={res['threshold_from_val']} valF1={res['val_f1_at_threshold']} "
              f"test P={res['test']['precision']} R={res['test']['recall']} F1={res['test']['f1']} "
              f"CM={res['test']['confusion_matrix']}")
    print("final:", final)
    return 0


if __name__ == "__main__":
    sys.exit(main())
