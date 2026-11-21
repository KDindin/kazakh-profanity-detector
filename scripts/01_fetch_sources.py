"""Download every public source used by the project into data/raw/.

Nothing here is committed to the repository: data/raw/ is git-ignored.
Run:  python scripts/01_fetch_sources.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"

SOURCES = [
    # (repo_id, filename, local name)
    ("kkenbbi/kazakh_reviews_2gis", "Reviews - Reviews.csv", "reviews_2gis_kk.csv"),
    ("R3iwan/entertainment-reviews-kazakh", "kazakh_entertainment_reviews_large.csv", "reviews_entertainment_kk.csv"),
    ("Rustem-Kaimolla/kazakh-swear-words", "data.jsonl", "swear_terms.jsonl"),
    ("Rustem-Kaimolla/kazakh-swear-words", "sentences.jsonl", "swear_sentences.jsonl"),
    ("kurumikz/telegram-corpus-russian-kazakh", "telegram_data.txt", "telegram_kk_ru.txt"),
]


def main() -> int:
    RAW.mkdir(parents=True, exist_ok=True)
    report = []
    for repo_id, filename, local_name in SOURCES:
        target = RAW / local_name
        if target.exists() and target.stat().st_size > 0:
            report.append((local_name, target.stat().st_size, "cached"))
            continue
        path = hf_hub_download(
            repo_id=repo_id, filename=filename, repo_type="dataset", local_dir=str(RAW / "_hf")
        )
        target.write_bytes(Path(path).read_bytes())
        report.append((local_name, target.stat().st_size, "downloaded"))

    for name, size, state in report:
        print(f"{name:32s} {size/1e6:8.2f} MB  {state}")
    (RAW / "sources.json").write_text(
        json.dumps(
            [
                {"repo_id": r, "filename": f, "local_name": n}
                for r, f, n in SOURCES
            ],
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
