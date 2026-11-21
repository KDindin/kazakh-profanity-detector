"""Show random labelled examples per split (useful during the defense).

    python scripts/11_show_examples.py --split test --label 1 --n 5
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import kazprof  # noqa: E402,F401  (configures UTF-8 stdout on Windows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--label", type=int, default=None, choices=[0, 1])
    ap.add_argument("--synthetic", action="store_true", help="only synthetic rows")
    ap.add_argument("--real", action="store_true", help="only real rows")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rows = [json.loads(l) for l in (ROOT / f"data/processed/{args.split}.jsonl").read_text(encoding="utf-8").splitlines()]
    if args.label is not None:
        rows = [r for r in rows if r["label"] == args.label]
    if args.synthetic:
        rows = [r for r in rows if r["synthetic"]]
    if args.real:
        rows = [r for r in rows if not r["synthetic"]]
    random.Random(args.seed).shuffle(rows)
    for r in rows[: args.n]:
        tag = "SYN" if r["synthetic"] else "REAL"
        print(f'[{r["label"]}] {tag:4s} {r["source"]:20s} {r["text"][:160]}')
    print(f"\n{len(rows)} rows match, showing {min(args.n, len(rows))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
