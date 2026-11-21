"""Error analysis draft: group mistakes and print the most informative ones.

Run after 07:  python scripts/09_error_analysis.py
Writes reports/error_candidates.md (for a human to interpret, not a final report).
"""
from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"


def main() -> None:
    rows = list(csv.DictReader((REPORTS / "predictions_test.csv").open(encoding="utf-8")))
    for r in rows:
        r["label"] = int(r["label"])
        r["pred"] = int(r["pred"])
        r["score"] = float(r["score"])
        r["synthetic"] = r["synthetic"] == "True"

    fp = [r for r in rows if r["label"] == 0 and r["pred"] == 1]
    fn = [r for r in rows if r["label"] == 1 and r["pred"] == 0]
    tp = [r for r in rows if r["label"] == 1 and r["pred"] == 1]

    lines = ["# Error candidates", "", f"TP {len(tp)}  FP {len(fp)}  FN {len(fn)}", ""]
    lines.append("## False positives (clean text flagged)")
    for r in sorted(fp, key=lambda x: -x["score"]):
        lines.append(f'- score={r["score"]:.3f} [{r["source"]}] {r["text"][:160]}')
    lines.append("")
    lines.append("## False negatives (profanity missed)")
    for r in sorted(fn, key=lambda x: x["score"]):
        lines.append(f'- score={r["score"]:.3f} [{r["source"]}] {r["text"][:160]}')
    lines.append("")
    lines.append("## Confident true positives (for contrast)")
    for r in sorted(tp, key=lambda x: -x["score"])[:10]:
        lines.append(f'- score={r["score"]:.3f} [{r["source"]}] {r["text"][:160]}')
    lines.append("")
    lines.append("## Mistake mix")
    lines.append(f'- FP by source: {dict(Counter(r["source"] for r in fp))}')
    lines.append(f'- FN by source: {dict(Counter(r["source"] for r in fn))}')
    lines.append(f'- FP synthetic: {sum(r["synthetic"] for r in fp)} / {len(fp)}')
    lines.append(f'- FN synthetic: {sum(r["synthetic"] for r in fn)} / {len(fn)}')

    (REPORTS / "error_candidates.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"tp": len(tp), "fp": len(fp), "fn": len(fn), "rows": len(rows)}))


if __name__ == "__main__":
    main()
