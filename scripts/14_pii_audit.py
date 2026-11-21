"""Privacy audit over everything that would be committed.

Checks data/processed/*.jsonl and reports/predictions_test.csv for the patterns that
must never leave the machine: e-mails, phones, URLs, profile links, @mentions,
long digit runs, and e-mail-like tokens. Exit code 1 if anything is found.

    python scripts/14_pii_audit.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

PATTERNS = {
    "email": re.compile(r"[\w.+-]+@[\w-]+\.[\w.]{2,}"),
    "url": re.compile(r"https?://|www\.|t\.me/|tg://", re.I),
    "phone": re.compile(r"(\+7|8)[\s\-()]?\d{3}[\s\-()]?\d{3}[\s\-]?\d{2}[\s\-]?\d{2}\b"),
    "mention": re.compile(r"(?<![\w<])@\w{2,}"),
    "long_digits": re.compile(r"\b\d{6,}\b"),
    "markdown_link": re.compile(r"\[[^\]]+\]\([^)]+\)"),
}


def iter_texts():
    for split in ("train", "val", "test"):
        for line in (ROOT / f"data/processed/{split}.jsonl").read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            yield f"{split}:{row['id']}", row["text"]
    csv_path = ROOT / "reports/predictions_test.csv"
    if csv_path.exists():
        import csv as _csv

        for row in _csv.DictReader(csv_path.open(encoding="utf-8")):
            yield f"pred:{row['id']}", row["text"]


def main() -> int:
    findings: list[tuple[str, str, str]] = []
    n = 0
    for where, text in iter_texts():
        n += 1
        for name, rx in PATTERNS.items():
            m = rx.search(text)
            if m:
                findings.append((where, name, m.group(0)))
    print(f"scanned {n} texts")
    if not findings:
        print("no PII patterns found")
        return 0
    for where, name, hit in findings[:40]:
        print(f"{where:24s} {name:14s} {hit!r}")
    print(f"TOTAL {len(findings)}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
