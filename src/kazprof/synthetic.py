"""Template-based generation of Kazakh review sentences.

Why synthetic data exists here
------------------------------
Public Kazakh review data is small and almost never contains profanity: marketplaces
and 2GIS moderate it away. Real profanity survives only in unmoderated chat, which
gives ~360 real positives but almost no *review-shaped* positives. Templates close
that gap: they combine neutral review frames with profane material taken from real
comments, so the model also sees profanity inside review syntax.

Two slot types keep the sentences grammatical:

* ``{pn}`` -- profanity used as a noun / bare insult (қотақ, қотақбас, еблан)
* ``{pp}`` -- profanity used as a standalone exclamation or verb (сігейін, бля)

Every generated row carries ``synthetic: true`` and a ``template_id`` so the
synthetic share can be reported, filtered out, or excluded from evaluation.
"""
from __future__ import annotations

import itertools
import random
from dataclasses import dataclass

ITEMS = ["тауар", "телефон", "шаңсорғыш", "кроссовки", "крем", "наушник", "телевизор", "кофе", "кітап", "куртка"]
ITEMS_RU = ["товар", "телефон", "пылесос", "кроссовки", "крем", "наушники", "телевизор", "кофе", "книга", "куртка"]
ADJ = ["дәмді", "жақсы", "керемет", "сапалы", "арзан", "орташа"]
ADJ_BAD = ["нашар", "сапасыз", "жарамсыз", "тым қымбат"]

# Profane material copied from real mined comments, grouped by how it can be used.
PROFANE_NOUN = [
    "қотақ", "қотақбас", "боқ", "қаншық", "жәлеп",
    "хуйня", "пиздец", "еблан", "долбоеб", "сука", "далбаеб",
]
PROFANE_PHRASE = [
    "нахуй", "бля", "заебал", "ахуеть", "нахуй керек",
    "сігейін", "сігіл", "шешең амы", "көтіңді қыс", "амын шығарам", "сігіп тастайын",
]

# Kazakh frames.
FRAMES_KK = [
    ("kk_review_01", "{item} {adj} екен, бірақ {pn}."),
    ("kk_review_02", "{pp}, {item} үш күнде бұзылып қалды."),
    ("kk_review_03", "Доставка кешікті, {pp}, {item} әлі жоқ."),
    ("kk_review_04", "Сатушы {pn} болып шықты, {item} сынған."),
    ("kk_review_05", "{item} сапасы {pn} деп айтамын, ақшамды қайтарыңдар."),
    ("kk_review_06", "Бұл {item} {pn} екен, мүлдем ұнамады."),
    ("kk_review_07", "{item} ашқанда {pp} — іші бос."),
    ("kk_review_08", "Бағасы {pn} қымбат, {item} орташа."),
    ("kk_review_09", "Қызмет {pn}, {item} әкелген жоқ."),
    ("kk_review_10", "Рахмет, {item} {adj}, бірақ қаптама {pp} жыртылған."),
    ("kk_review_11", "{pp}, тағы да {item} дұрыс емес келді."),
    ("kk_review_12", "Бір жұлдыз, өйткені {pn}."),
    ("kk_review_13", "{item} туралы {pn} пікір — сапасыз."),
    ("kk_review_14", "Мен {pp}, ренжідім, {item} жартылай жұмыс істейді."),
    ("kk_review_15", "{pp} {item}, кепілдік те жоқ."),
    ("kk_review_16", "{item} {bad_adj} болып шықты, {pn}."),
    ("kk_review_17", "Екінші рет алмаймын, {pn}, {item} {bad_adj}."),
    ("kk_review_18", "Сатушыға жазсам {pp} деп жауап берді, {item} {bad_adj}."),
    ("kk_review_19", "{pp}, кассаға қайтару үшін кезек күттім, {item} {bad_adj}."),
    ("kk_review_20", "Фотоға сәйкес емес, {pn}, {item} {bad_adj}."),
]

# Mixed Kazakh-Russian frames, the way many KZ customers actually write.
# The connector is a slot: if every mixed frame used "но", the model would learn
# "но => profanity" instead of the profanity itself (observed in an audit).
CONNECTORS = ["но", "бірақ", "а", "и", ""]
FRAMES_MIX = [
    ("mix_review_01", "{item_ru} норм екен, {conn} упаковка {pn}."),
    ("mix_review_02", "Заказ келді, {conn} {pn} брак."),
    ("mix_review_03", "{pp}, деньги верните за {item_ru} деп жаздым."),
    ("mix_review_04", "Доставка {pp} долго болды, {item_ru} так и не пришёл."),
    ("mix_review_05", "Сапасы {pn}, больше не закажу {item_ru}."),
    ("mix_review_06", "{item_ru} супер екен, {conn} сервис {pn}."),
    ("mix_review_07", "Пишу отзыв, потому что {pp} деп ренжідім."),
    ("mix_review_08", "{pn}, вместо {item_ru} басқа зат келді."),
    ("mix_review_09", "{item_ru} {adj} екен, {conn} размер {pn} не тот."),
    ("mix_review_10", "Возврат {pp} жасамайды, {item_ru} сломан."),
    ("mix_review_11", "{conn} {item_ru} келді, сапасы {pn}."),
    ("mix_review_12", "Заказ {pp}, {item_ru} жоқ, деньги верните."),
]

# Clean mixed sentences that reuse the same vocabulary as the frames above.
# Without them, words like "доставка", "товар", "но" become profanity markers.
MIXED_CLEAN = [
    "Доставка кешікті, но товар норм.",
    "Заказ келді, но упаковка жыртылған, іші сау.",
    "Товар норм, но доставка долго болды.",
    "Сапасы жақсы, но размер не тот, қайтарамын.",
    "Доставка тез болды, товар күткендей, рахмет.",
    "Но цена қымбат, сапасы орташа.",
    "Курьер кешікті, но сыпайы болды, рахмет.",
    "Упаковка жыртылған, но ішіндегі зат сау.",
    "Заказ келді, но түсі фотоға сәйкес емес.",
    "Товар жақсы екен, но кепілдік жоқ.",
    "Доставка долго болды, но сатушы жауап берді.",
    "Сапасы орташа, но бағасы арзан.",
    "Заказды қайтардым, но ақшамды берді.",
    "Но сервис жақсы, келесіде тағы келемін.",
]

# Clean sentences built around the traps that cause false positives:
# insults without profanity, quoted words, and ordinary words hiding a profane substring.
FRAMES_NEGATIVE = [
    ("hard_neg_01", "Сатушы ақымақ сияқты әрекет етті, тауарды қайтардым."),
    ("hard_neg_02", "Сасық иіс шығады, бірақ дүкен қайтарып берді."),
    ("hard_neg_03", "Курьер кешікті, мен ренжідім, бірақ тауар жақсы."),
    ("hard_neg_04", "Бокс қолғабын сатып алдым, сапасы жақсы."),
    ("hard_neg_05", "Көтеріңкі көңіл-күймен келдім, бірақ тауар жоқ екен."),
    ("hard_neg_06", "Педикюрге бардым, қызмет өте жақсы, рахмет."),
    ("hard_neg_07", "Амандық тілеймін, тауар керемет келді."),
    ("hard_neg_08", "Самал жел соғып тұр, көңілім жақсы, тауар ұнады."),
    ("hard_neg_09", "Ақымақ емеспін, бірақ бұл тауардың нұсқаулығын түсінбедім."),
    ("hard_neg_10", "Итпұшық деп аталатын өсімдік туралы оқыдым, тауарға қатысы жоқ."),
    ("hard_neg_11", "Пікіріңізді балағат сөзсіз, сыпайы жазыңыз."),
    ("hard_neg_12", "Тауар нашар болды, ақшамды қайтардым, қызмет те нашар."),
    ("hard_neg_13", "Қаптамасы жыртылған, ішіндегі зат сау, бірақ ренжідім."),
    ("hard_neg_14", "Бағасы қымбат, сапасы орташа, екінші рет алмаймын."),
    ("hard_neg_15", "Көтерме бағамен алдым, тауар жақсы, сатушы сыпайы."),
    ("hard_neg_16", "Боксты жақсы көремін, спорттық киім сапалы келді."),
]

NEUTRAL_FRAMES = [
    "{item} {adj} екен, қаптамасы да тұтас, рахмет.",
    "{item} уақытында келді, сапасы {adj}, сатушы сыпайы.",
    "Бағасы {adj}, {item} күткендей болды, ұнады.",
    "{item_ru} {adj} екен, доставка тез болды.",
]


@dataclass
class Generated:
    text: str
    label: int
    template_id: str
    kind: str  # synthetic_positive | synthetic_negative


def _fill(template: str, rng: random.Random) -> str:
    text = template.format(
        item=rng.choice(ITEMS),
        item_ru=rng.choice(ITEMS_RU),
        adj=rng.choice(ADJ),
        bad_adj=rng.choice(ADJ_BAD),
        pn=rng.choice(PROFANE_NOUN),
        pp=rng.choice(PROFANE_PHRASE),
        conn=rng.choice(CONNECTORS),
    )
    text = " ".join(text.split())
    return text[0].upper() + text[1:]


def generate(n_pos: int, n_neg: int, seed: int = 42) -> list[Generated]:
    rng = random.Random(seed)
    out: list[Generated] = []
    seen: set[str] = set()

    pos_frames = [(t, f, "kk") for t, f in FRAMES_KK] + [(t, f, "mixed") for t, f in FRAMES_MIX]
    attempts = 0
    while len(out) < n_pos and attempts < n_pos * 40:
        attempts += 1
        tid, frame, _ = pos_frames[len(out) % len(pos_frames)]
        text = _fill(frame, rng)
        if text in seen:
            continue
        seen.add(text)
        out.append(Generated(text, 1, tid, "synthetic_positive"))

    neg: list[Generated] = []
    for i, text in enumerate(MIXED_CLEAN):
        if text in seen:
            continue
        seen.add(text)
        neg.append(Generated(text, 0, f"mixed_clean_{i:02d}", "synthetic_negative"))
    for tid, frame in FRAMES_NEGATIVE:
        text = frame.format(item=rng.choice(ITEMS), item_ru=rng.choice(ITEMS_RU))
        neg.append(Generated(text, 0, tid, "synthetic_negative"))
    while len(neg) < n_neg:
        tid = f"hard_neg_extra_{len(neg) % 4}"
        text = _fill(rng.choice(NEUTRAL_FRAMES), rng)
        if text in seen:
            continue
        seen.add(text)
        neg.append(Generated(text, 0, tid, "synthetic_negative"))

    out.extend(itertools.islice(neg, n_neg))
    rng.shuffle(out)
    return out
