"""Quick lexicon regression check: does every pattern fire on a positive control
and stay silent on a negative control?  Run: python scripts/10_lexicon_check.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from kazprof.lexicon import Lexicon  # noqa: E402

POSITIVES = {
    "kk_kotak": ["қотақ", "Қотағымды жеме", "қотағыңды", "котак", "Қотақбас"],
    "kk_am": ["ам", "Амынды шығарам", "шешең амы"],
    "kk_sig": ["сігейін", "Сігіп аламын", "сыгейн", "сігілш"],
    "kk_ket": ["көтіңді қыс", "Котинды кыс", "көтің"],
    "kk_bok": ["боқ", "Боқ сөз", "бок же"],
    "kk_zhalep": ["жәлеп"],
    "kk_kanshyk": ["қаншық", "каншык"],
    "ru_huy": ["хуйня", "нахуй керек", "похуй"],
    "ru_blya": ["бля", "блять", "Блә"],
    "ru_suka": ["сука", "сучка"],
    "ru_pizd": ["пиздец"],
    "ru_eb": ["заебал", "долбоеб", "далбаеб", "еблан"],
    "ru_nahuy": ["нахүй"],
    "ru_pidor": ["пидор", "Пидарас"],
}
NEGATIVES = [
    "Көтерме бағамен алдым, тауар жақсы",
    "Педикюрге бардым, қызмет жақсы",
    "Бокс қолғабын сатып алдым",
    "Амандық тілеймін",
    "Самал жел соғып тұр",
    "Тамағы дәмсіз, қызмет нашар",
    "Сатушы ақымақ сияқты",
    "Пікіріңізді балағат сөзсіз жазыңыз",
    "Доставка кешікті, тауар жоқ",
]

lex = Lexicon()
ok, bad = 0, 0
for term_id, examples in POSITIVES.items():
    for text in examples:
        ids = {h.term_id for h in lex.hits_text(text)}
        if term_id in ids:
            ok += 1
        else:
            bad += 1
            print(f"MISS  {term_id:12s} {text!r} -> {ids}")
for text in NEGATIVES:
    ids = {h.term_id for h in lex.hits_text(text)}
    if ids:
        bad += 1
        print(f"FP    {text!r} -> {ids}")
    else:
        ok += 1
print(f"passed {ok}, failed {bad}")
sys.exit(1 if bad else 0)
