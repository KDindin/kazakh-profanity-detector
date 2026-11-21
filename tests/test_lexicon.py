import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from kazprof.lexicon import Lexicon  # noqa: E402

lex = Lexicon()

POSITIVE = [
    "қотақ", "Қотағымды жеме", "котак", "Қотақбас", "шешең амы", "Амынды шығарам",
    "сігейін", "сыгейн", "көтіңді қыс", "Котинды кыс", "боқ", "жәлеп", "қаншық",
    "нахуй", "бля", "Блә", "сука", "пиздец", "заебал", "далбаеб", "пидор",
]
NEGATIVE = [
    "Көтерме бағамен алдым, тауар жақсы",
    "Педикюрге бардым, қызмет жақсы",
    "Бокс қолғабын сатып алдым",
    "Амандық тілеймін",
    "Самал жел соғып тұр",
    "Тамағы дәмсіз, қызмет нашар",
    "Сатушы ақымақ сияқты",
    "Пікіріңізді балағат сөзсіз жазыңыз",
]


def test_positives_hit():
    for text in POSITIVE:
        assert lex.has_profanity(text), text


def test_negatives_do_not_hit():
    for text in NEGATIVE:
        assert not lex.has_profanity(text), text


def test_masked_spelling_hits():
    assert lex.has_profanity("с*ка, деньги верните")
    assert lex.has_profanity("қ о т а қ")


def test_squash_does_not_bridge_two_real_words():
    # "де" + "баланың" must not form "ебал"
    assert not lex.has_profanity("бізге орыс мектебі қажет емес, баланың білімі маңызды")
