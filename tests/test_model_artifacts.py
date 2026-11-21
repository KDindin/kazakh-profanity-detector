"""Artifact-level guarantees: the saved model is self-contained and was fitted on train only."""
import json
import re
import sys
from collections import Counter
from pathlib import Path

import joblib
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kazprof.normalize import normalize  # noqa: E402

WORD = re.compile(r"\b\w\w+\b", re.UNICODE)


def load_words(split):
    rows = [json.loads(l) for l in (ROOT / f"data/processed/{split}.jsonl").read_text(encoding="utf-8").splitlines()]
    return Counter(w for r in rows for w in WORD.findall(normalize(r["text"]).norm))


@pytest.fixture(scope="module")
def pipe():
    return joblib.load(ROOT / "models/baseline_tfidf_lr.joblib")["pipeline"]


def test_vocabulary_was_fitted_on_train_only(pipe):
    word_vec = pipe.named_steps["features"].transformer_list[0][1]
    vocab = word_vec.vocabulary_
    train, test = load_words("train"), load_words("test")

    seen_in_train = [w for w, c in train.items() if c >= 3][:2000]
    assert any(w in vocab for w in seen_in_train), "expected training words in the vocabulary"

    test_only = [w for w in test if w not in train][:50]
    assert test_only, "expected at least one test-only token to check leakage"
    leaked = [w for w in test_only if w in vocab]
    assert not leaked, f"test-only tokens leaked into the fitted vocabulary: {leaked[:5]}"


def test_saved_threshold_records_where_it_came_from():
    th = json.loads((ROOT / "models/threshold.json").read_text(encoding="utf-8"))
    assert th["tuned_on"] == "validation split"
    assert 0.0 < th["threshold"] < 1.0
    final = json.loads((ROOT / "models/final_model.json").read_text(encoding="utf-8"))
    assert final["final_model"] in {"baseline", "transformer"}
    assert final["threshold"] == th["threshold"]


def test_transformer_directory_is_self_contained():
    model_dir = ROOT / "models" / "kazroberta_kk_profanity"
    if not model_dir.exists():
        pytest.skip("transformer checkpoint not trained in this checkout")
    required = {"config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json"}
    assert required <= {p.name for p in model_dir.iterdir()}

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True)
    enc = tok("қотақ", add_special_tokens=False)
    assert enc["input_ids"], "tokenizer produced no ids"


def test_lexicon_is_versioned_and_loaded():
    lex = json.loads((ROOT / "data/lexicon/profanity_kk.json").read_text(encoding="utf-8"))
    assert lex["version"] >= 2
    assert len(lex["terms"]) >= 18
    ids = [t["id"] for t in lex["terms"]]
    assert len(ids) == len(set(ids)), "duplicate term ids"
