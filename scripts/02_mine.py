"""Mine profanity candidates from the Kazakh/Russian Telegram corpus.

The corpus is unmoderated comment text, so it is the only public source with a
realistic supply of genuine profanity. Everything mined here is *candidate*
material: each row is manually reviewed before it enters the dataset.

Run:  python scripts/02_mine.py
"""
from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kazprof.lexicon import Lexicon  # noqa: E402
from kazprof.normalize import is_in_scope, normalize  # noqa: E402

RAW = ROOT / "data" / "raw"
INTERIM = ROOT / "data" / "interim"
MIN_CHARS, MAX_CHARS = 12, 400


def main() -> int:
    INTERIM.mkdir(parents=True, exist_ok=True)
    lex = Lexicon()
    seen: set[str] = set()
    keep_neg: list[dict] = []
    keep_pos: list[dict] = []
    stats = Counter()

    path = RAW / "telegram_kk_ru.txt"
    with path.open(encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            stats["lines"] += 1
            if len(line) < MIN_CHARS:
                stats["too_short"] += 1
                continue
            n = normalize(line)
            if not is_in_scope(n):
                stats[f"out_of_scope_{n.lang}"] += 1
                continue
            if len(n.norm) < MIN_CHARS:
                stats["empty_after_norm"] += 1
                continue
            key = n.squash[:300]
            if key in seen:
                stats["duplicate"] += 1
                continue
            seen.add(key)

            hits = lex.hits(n)
            row = {
                "text": line[:MAX_CHARS],
                "norm": n.norm[:MAX_CHARS],
                "lang": n.lang,
                "hits": "|".join(sorted({h.term_id for h in hits})),
                "severities": "|".join(sorted({h.severity for h in hits})),
            }
            if hits:
                keep_pos.append(row)
            else:
                keep_neg.append(row)

    # Cap the negative pool; positives are kept in full for manual review.
    keep_neg = keep_neg[:6000]

    with (INTERIM / "telegram_positive_candidates.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(keep_pos[0].keys()))
        w.writeheader()
        w.writerows(keep_pos)
    with (INTERIM / "telegram_negative_pool.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(keep_neg[0].keys()))
        w.writeheader()
        w.writerows(keep_neg)

    stats["pos_candidates"] = len(keep_pos)
    stats["neg_pool"] = len(keep_neg)
    stats["pos_by_lang"] = dict(Counter(r["lang"] for r in keep_pos))
    stats["pos_by_term"] = dict(
        Counter(t for r in keep_pos for t in r["hits"].split("|") if t).most_common(20)
    )
    (INTERIM / "mine_stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
