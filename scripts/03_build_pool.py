"""Assemble every candidate row into one pool with stable ids.

Sources
-------
real reviews  : Kaspi.kz marketplace reviews (kk subset), 2GIS restaurant reviews
real comments : Telegram corpus (Kazakh / Kazakh-Russian mixed lines)
real seeds    : 63 hand-written profane sentences from the public swear-words set
synthetic     : generated later, in 04_build_dataset.py

Run:  python scripts/03_build_pool.py
"""
from __future__ import annotations

import csv
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kazprof.lexicon import Lexicon  # noqa: E402
from kazprof.normalize import normalize  # noqa: E402

RAW = ROOT / "data" / "raw"
INTERIM = ROOT / "data" / "interim"
MIN_CHARS, MAX_CHARS = 12, 400
SEED_MIN_CHARS = 8

BOILERPLATE = re.compile(r"\s*\d+\s*человек\(а\)\s*посчитал\(и\)\s*отзыв полезным\s*", re.I)
PREFIX = re.compile(r"^\s*(Достоинства|Недостатки|Комментарий)\s*:\s*", re.I)


def row_id(source: str, text: str) -> str:
    return hashlib.sha1(f"{source}|{normalize(text).squash[:400]}".encode()).hexdigest()[:12]


def clean_review(text: str) -> str:
    text = BOILERPLATE.sub(" ", text or "")
    text = PREFIX.sub("", text.strip())
    return re.sub(r"\s+", " ", text).strip()


def main() -> int:
    INTERIM.mkdir(parents=True, exist_ok=True)
    lex = Lexicon()
    pool: list[dict] = []
    seen: set[str] = set()
    stats = Counter()

    def add(text: str, source: str, origin: str, meta: dict | None = None, min_chars: int = MIN_CHARS) -> None:
        text = re.sub(r"\s+", " ", (text or "")).strip()
        if not (min_chars <= len(text) <= MAX_CHARS):
            stats[f"skip_len_{source}"] += 1
            return
        n = normalize(text)
        if source == "telegram_kk":
            # drop chat rows that are really quiz items, news, or profile links:
            # they are not comments about goods and only add domain noise
            if re.match(r"^\d+[\.\)]\s", text) or (text.endswith("?") and len(text) > 120):
                stats["skip_quiz_telegram"] += 1
                return
            if "<user>" in n.norm or "<url>" in n.norm:
                stats["skip_link_telegram"] += 1
                return
        key = n.squash[:300]
        if key in seen:
            stats[f"dup_{source}"] += 1
            return
        seen.add(key)
        hits = lex.hits(n)
        clean_text = n.masked or text
        pool.append(
            {
                "id": row_id(source, clean_text),
                "text": clean_text,
                "source": source,
                "origin": origin,
                "lang": n.lang,
                "hits": "|".join(sorted({h.term_id for h in hits})),
                "severity": "|".join(sorted({h.severity for h in hits})),
                "rule_label": int(bool(hits)),
                "sentiment": (meta or {}).get("sentiment", ""),
            }
        )
        stats[f"kept_{source}"] += 1

    # --- real product/venue reviews -------------------------------------
    kaspi = list(csv.DictReader((RAW / "kaspi_reviews.csv").open(encoding="utf-8-sig")))
    for r in kaspi:
        text = clean_review(r.get("Пікір", ""))
        if normalize(text).lang not in {"kk", "mixed"}:
            stats["kaspi_russian"] += 1
            continue
        add(text, "kaspi_reviews", "real_review")

    gis = list(csv.DictReader((RAW / "reviews_2gis_kk.csv").open(encoding="utf-8")))
    for r in gis:
        add(clean_review(r.get("text", "")), "2gis_reviews", "real_review",
            {"sentiment": r.get("label", "")})

    # --- real comments ---------------------------------------------------
    with (INTERIM / "telegram_positive_candidates.csv").open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["lang"] not in {"kk", "mixed"}:
                stats["telegram_pos_not_kk"] += 1
                continue
            add(r["text"], "telegram_kk", "real_comment")
    with (INTERIM / "telegram_negative_pool.csv").open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["lang"] not in {"kk", "mixed"}:
                continue
            add(r["text"], "telegram_kk", "real_comment")

    # --- real profane seed sentences ------------------------------------
    seeds = [json.loads(l) for l in (RAW / "swear_sentences.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    for s in seeds:
        text = s["text"]
        # two rows of this public set are not profane at all ("Щщс", "667")
        if not lex.has_profanity(text):
            stats["seed_dropped_unlabelled_noise"] += 1
            continue
        add(text, "swear_sentences_seed", "real_seed", min_chars=SEED_MIN_CHARS)

    with (INTERIM / "pool.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(pool[0].keys()))
        w.writeheader()
        w.writerows(pool)

    stats["pool_total"] = len(pool)
    stats["pool_positive_rule"] = sum(p["rule_label"] for p in pool)
    stats["by_source"] = dict(Counter(p["source"] for p in pool))
    stats["by_lang"] = dict(Counter(p["lang"] for p in pool))
    (INTERIM / "pool_stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
