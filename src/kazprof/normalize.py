"""Text normalisation shared by mining, training and inference.

The same function must be used everywhere, otherwise training/serving skew
appears and the saved tokenizer/vectoriser sees text it was never fitted on.

Two views are produced:

* ``norm``   -- lowercase, homoglyphs folded, PII masked, repeats collapsed.
                This is what the models read.
* ``squash`` -- ``norm`` with every non-letter removed, so that obfuscated
                spellings such as ``с*ка``, ``с у к а`` or ``қ.о.т.а.қ``
                collapse onto the plain word for lexicon matching.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

KK_SPECIFIC = set("әғқңөұүһі")
# Ukrainian shares "і" with Kazakh; these letters are uniquely Ukrainian and are
# the reliable way to keep Ukrainian comments out of a Kazakh task.
UK_SPECIFIC = set("їєґ")
UK_WORDS = {
    "що", "це", "буде", "дуже", "треба", "можна", "його", "вона", "вони",
    "мене", "тобі", "тільки", "також", "після", "через", "зараз", "щось",
    "якщо", "можу", "чи", "цей", "ця", "ці", "який", "яка", "які", "було",
    "були", "вже", "ще", "люди", "потрібно", "тому", "цього", "нього",
}

# Latin / digit characters that are routinely used to disguise Cyrillic words.
HOMOGLYPHS = str.maketrans(
    {
        "a": "а", "e": "е", "o": "о", "p": "р", "c": "с", "x": "х",
        "y": "у", "k": "к", "m": "м", "t": "т", "h": "н", "b": "в",
        "0": "о", "3": "з", "6": "б",
    }
)

RE_URL = re.compile(r"(https?://\S+|www\.\S+|tg://\S+|t\.me/\S+|\b\S+\.(?:kz|ru|com|net|org)\b)", re.I)
RE_MENTION = re.compile(r"@\w+")
RE_EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.]+\b")
RE_PHONE = re.compile(r"(?:\+?7|8)[\s\-()]?\d{3}[\s\-()]?\d{3}[\s\-]?\d{2}[\s\-]?\d{2}\b")
RE_LONGRUN = re.compile(r"\b\d{5,}\b")
RE_REPEAT = re.compile(r"(.)\1{2,}")
RE_SPACES = re.compile(r"\s+")
RE_LEADING_INDEX = re.compile(r"^\s*\[\d+\]\s*")

KK_WORDS = {
    "мен", "және", "бірақ", "өте", "жақсы", "бар", "жоқ", "емес", "деп",
    "бұл", "сол", "үшін", "керек", "өзі", "тамақ", "дәмді", "қызмет",
    "рахмет", "келемін", "ұнады", "бәрі", "қатты", "екен", "деген",
}
RU_WORDS = {
    "и", "что", "как", "очень", "это", "не", "на", "для", "был", "есть",
    "при", "все", "только", "меня", "мне", "там", "если", "или", "ещё",
    "спасибо", "хорошо", "плохо", "можно", "нет", "да",
}


@dataclass
class NormText:
    raw: str
    masked: str
    norm: str
    squash: str
    tokens: list[str] = field(default_factory=list)
    lang: str = "unknown"  # kk | ru | uk | mixed | unknown

    def as_dict(self) -> dict:
        return {"norm": self.norm, "squash": self.squash, "lang": self.lang}


def _fold_token(tok: str) -> str:
    """Fold Latin lookalikes and digits only inside tokens that are Cyrillic."""
    if not any("\u0400" <= ch <= "\u04ff" for ch in tok):
        return tok
    return tok.translate(HOMOGLYPHS)


def mask_pii(text: str) -> str:
    # markdown/telegram links that carry a display name and a profile id
    text = re.sub(r"\[([^\]]*)\]\([^)]*\)", " <user> ", text)
    text = RE_URL.sub(" <url> ", text)
    text = RE_EMAIL.sub(" <email> ", text)
    text = RE_PHONE.sub(" <phone> ", text)
    text = RE_MENTION.sub(" <user> ", text)
    text = RE_LONGRUN.sub(" <num> ", text)
    return text


def detect_lang(norm: str) -> str:
    """Coarse language guess for scope filtering (kk / ru / uk / mixed / unknown).

    Kazakh and Ukrainian both use ``і``; Ukrainian is therefore detected by its
    unique letters (ї, є, ґ) and function words first, and only then Kazakh is
    considered.
    """
    words = re.findall(r"[а-яәғқңөұүһіїєґ]+", norm)
    kk_specific = sum(1 for ch in norm if ch in KK_SPECIFIC and ch != "і")
    kk_i = norm.count("і")
    uk_specific = sum(1 for ch in norm if ch in UK_SPECIFIC)
    kk_words = sum(1 for w in words if w in KK_WORDS)
    ru_words = sum(1 for w in words if w in RU_WORDS)
    uk_words = sum(1 for w in words if w in UK_WORDS)

    kk_score = kk_specific * 4 + kk_words * 3 + (1 if kk_i and not uk_specific and not uk_words else 0)
    uk_score = uk_specific * 6 + uk_words * 3
    ru_score = ru_words * 3

    if uk_score > max(kk_score, ru_score):
        return "uk"
    if kk_score and ru_score and min(kk_score, ru_score) / max(kk_score, ru_score) > 0.5:
        return "mixed"
    if kk_score > ru_score:
        return "kk"
    if ru_score > kk_score:
        return "ru"
    return "unknown"


def is_in_scope(n: NormText) -> bool:
    """Kazakh or Kazakh-Russian mixed comment: the task's input domain."""
    return n.lang in {"kk", "mixed"}


def normalize(text: str) -> NormText:
    """Full pipeline: PII mask -> NFKC -> lowercase -> homoglyph fold -> clean."""
    if not isinstance(text, str):
        text = "" if text is None else str(text)
    raw = text
    text = RE_LEADING_INDEX.sub("", text)
    text = unicodedata.normalize("NFKC", text)
    masked = RE_SPACES.sub(" ", mask_pii(text)).strip()
    text = text.lower().replace("ё", "е")
    text = mask_pii(text)
    # keep letters, digits, spaces and light punctuation; drop emoji/symbols
    text = "".join(
        ch if (ch.isalnum() or ch in " .,!?;:-'\"()*") else " " for ch in text
    )
    pieces = []
    for tok in text.split():
        pieces.append(_fold_token(tok))
    norm = RE_SPACES.sub(" ", " ".join(pieces)).strip()
    norm = RE_REPEAT.sub(r"\1", norm)  # сууууупер -> супер
    tokens = [t for t in re.split(r"[\s.,!?;:\-\"'()]+", norm) if t]
    squash = "".join(ch for ch in norm if ch.isalpha())
    return NormText(raw=raw, masked=masked, norm=norm, squash=squash, tokens=tokens, lang=detect_lang(norm))


def is_kazakh_context(n: NormText, min_kk_letters: int = 1) -> bool:
    """True when the comment is Kazakh or Kazakh-Russian mixed (task scope)."""
    if n.lang in {"kk", "mixed"}:
        return True
    return sum(1 for ch in n.norm if ch in KK_SPECIFIC) >= min_kk_letters and n.lang != "ru"
