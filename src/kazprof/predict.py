"""Command line: classify one comment (or read lines from a file).

    python -m kazprof.predict "тауар жақсы, бірақ қызметі қотақ"
    python -m kazprof.predict --explain "Сігейін, тауар келмеді"
    python -m kazprof.predict --file comments.txt
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kazprof.infer import Detector  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="Detect Kazakh profanity in a review or comment.")
    ap.add_argument("text", nargs="?", help="comment text")
    ap.add_argument("--file", help="file with one comment per line")
    ap.add_argument("--backend", default="auto", choices=["auto", "transformer", "baseline"])
    ap.add_argument("--threshold", type=float, default=None, help="override the saved decision threshold")
    ap.add_argument("--explain", action="store_true", help="show normalised text and lexicon hits")
    args = ap.parse_args()

    det = Detector(backend=args.backend, threshold=args.threshold)
    if args.file:
        texts = [l.strip() for l in Path(args.file).read_text(encoding="utf-8").splitlines() if l.strip()]
    elif args.text:
        texts = [args.text]
    else:
        texts = [sys.stdin.read().strip()]

    for t in texts:
        print(json.dumps(det.predict(t, explain=args.explain), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
