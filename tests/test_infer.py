import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kazprof.infer import Detector  # noqa: E402


@pytest.fixture(scope="module")
def det():
    return Detector(backend="baseline")


def test_output_contract(det):
    out = det.predict("Тамағы дәмді екен, бірақ қызметі қотақ")
    assert set(out) >= {"label", "score", "threshold", "backend", "lang", "lexicon_hits"}
    assert out["label"] in (0, 1)
    assert 0.0 <= out["score"] <= 1.0
    assert out["backend"] == "baseline"


def test_clear_profanity_is_flagged(det):
    assert det.predict("Сігейін, тағы да дұрыс емес келді")["label"] == 1


def test_clean_review_is_not_flagged(det):
    assert det.predict("Көтерме бағамен алдым, тауар жақсы, сатушы сыпайы")["label"] == 0


def test_threshold_is_respected(det):
    text = "Доставка кешікті, но товар норм"
    score = det.score(text)
    assert 0.0 < score < 1.0
    low = Detector(backend="baseline", threshold=max(0.01, score - 0.05)).predict(text)["label"]
    high = Detector(backend="baseline", threshold=min(0.99, score + 0.05)).predict(text)["label"]
    assert low == 1 and high == 0


def test_explain_adds_fields(det):
    out = det.predict("қотақ", explain=True)
    assert "normalised" in out and "rule_feature" in out
    assert out["rule_feature"] > 0
