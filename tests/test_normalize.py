import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from kazprof.normalize import is_in_scope, mask_pii, normalize  # noqa: E402


def test_masks_profile_links_and_phones():
    text = "пиши [**Ерасыл**](tg://user?id=1337) или +7 701 123 45 67"
    masked = mask_pii(text)
    assert "tg://" not in masked
    assert "1337" not in masked
    assert "701 123" not in masked


def test_folds_homoglyphs_only_inside_cyrillic_tokens():
    assert normalize("cука").norm == "сука"          # latin c
    assert normalize("пиздeц").norm == "пиздец"      # latin e
    assert normalize("ok top").norm == "ok top"      # latin-only token untouched


def test_collapses_stretched_letters():
    assert normalize("суууука").norm == "сука"
    assert normalize("Бляяять").norm == "блять"


def test_kazakh_and_ukrainian_are_distinguished():
    assert normalize("Бұл тауар өте жақсы екен").lang in {"kk", "mixed"}
    assert normalize("Якщо цікаво, можу розповісти більше").lang == "uk"
    assert not is_in_scope(normalize("Якщо цікаво, можу розповісти більше"))


def test_pure_russian_is_out_of_scope_but_mixed_is_in():
    assert normalize("Доставка долго, товар так и не пришёл").lang == "ru"
    assert is_in_scope(normalize("Доставка кешікті, товар жоқ"))


def test_leading_list_index_is_stripped():
    assert normalize("[1] Фильм ұнады").norm == "фильм ұнады"


def test_empty_and_none_are_safe():
    assert normalize("").norm == ""
    assert normalize(None).norm == ""
