import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kazprof.normalize import normalize  # noqa: E402

REQUIRED = {"id", "text", "label", "source", "origin", "synthetic", "template_id", "lang"}


def load(split):
    return [json.loads(l) for l in (ROOT / f"data/processed/{split}.jsonl").read_text(encoding="utf-8").splitlines()]


def test_schema_and_labels():
    for split in ("train", "val", "test"):
        rows = load(split)
        assert rows, f"{split} is empty"
        for r in rows:
            assert REQUIRED <= set(r), r
            assert r["label"] in (0, 1)
            assert isinstance(r["synthetic"], bool)
            assert r["text"].strip()


def test_split_balance():
    stats = json.loads((ROOT / "data/processed/dataset_stats.json").read_text(encoding="utf-8"))
    assert stats["total"] == 2000
    assert stats["labels"] == {"0": 1300, "1": 700}
    for split in ("train", "val", "test"):
        assert stats["splits"][split]["positives"] > 0


def test_no_duplicate_text_across_splits():
    seen = {}
    for split in ("train", "val", "test"):
        for r in load(split):
            key = normalize(r["text"]).squash[:300]
            assert key not in seen, f"duplicate across {seen.get(key)} and {split}"
            seen[key] = split


def test_synthetic_rows_are_flagged_and_labelled():
    rows = [r for s in ("train", "val", "test") for r in load(s)]
    syn = [r for r in rows if r["synthetic"]]
    assert syn, "expected synthetic rows"
    assert all(r["template_id"] for r in syn)
    assert all(r["origin"] == "synthetic" for r in syn)


def test_no_pii_markers_left_unmasked():
    for r in [r for s in ("train", "val", "test") for r in load(s)]:
        assert "tg://" not in r["text"]
        assert "@" not in r["text"] or "<" in r["text"]


def test_scope_is_kazakh_or_mixed():
    rows = [r for s in ("train", "val", "test") for r in load(s)]
    langs = Counter(r["lang"] for r in rows)
    assert set(langs) <= {"kk", "mixed"}, langs
