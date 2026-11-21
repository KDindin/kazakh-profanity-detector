"""Lexicon matching: the rule layer used for mining candidates and as a feature."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from .normalize import NormText, normalize

DEFAULT_LEXICON = Path(__file__).resolve().parents[2] / "data" / "lexicon" / "profanity_kk.json"
# Terms shorter than this are only matched on the word-level view, never on the
# squash view: without separators "ам" would fire inside ordinary words.
MIN_SQUASH_LEN = 4


@dataclass(frozen=True)
class Hit:
    term_id: str
    kind: str
    severity: str
    surface: str


def _squash_with_map(norm: str) -> tuple[str, list[int]]:
    """Remove non-letters, remembering which token every kept char came from."""
    chars: list[str] = []
    owners: list[int] = []
    token_idx = -1
    prev_letter = False
    for ch in norm:
        if ch.isalpha():
            if not prev_letter:
                token_idx += 1
            chars.append(ch)
            owners.append(token_idx)
            prev_letter = True
        else:
            prev_letter = False
    return "".join(chars), owners


def _token_lengths(norm: str) -> list[int]:
    return [len(t) for t in re.findall(r"[^\W\d_]+", norm, flags=re.UNICODE)]


class Lexicon:
    def __init__(self, path: Path | str | None = None) -> None:
        path = Path(path) if path else DEFAULT_LEXICON
        data = json.loads(path.read_text(encoding="utf-8"))
        self.version = data["version"]
        self.entries = data["terms"]
        self._compiled = [(e, re.compile(e["pattern"])) for e in self.entries]

    def _squash_hits(self, n: NormText) -> list[tuple[dict, str]]:
        """Match obfuscated spellings, rejecting matches that span real words.

        ``с*ка`` / ``с у к а`` survive squashing and hit; ``...де`` + ``буткыл...``
        accidentally forming ``деб`` does not, because the covered tokens are long.
        """
        squash, owners = _squash_with_map(n.norm)
        if not squash:
            return []
        lengths = _token_lengths(n.norm)
        out: list[tuple[dict, str]] = []
        for entry, rx in self._compiled:
            if entry.get("squash", True) is False:
                continue
            if len(entry["pattern"].replace("\\b", "")) < MIN_SQUASH_LEN:
                continue
            for m in rx.finditer(squash):
                start_tok, end_tok = owners[m.start()], owners[m.end() - 1]
                span = list(range(start_tok, end_tok + 1))
                # Allowed: a single token, or a word deliberately broken into
                # short pieces ("с у к а", "қ.о.т.а.қ"). A match that bridges two
                # real words ("де" + "баланың") is rejected.
                if len(span) > 1 and any(lengths[i] > 3 for i in span):
                    continue
                out.append((entry, m.group(0)))
                break
        return out

    def hits(self, n: NormText) -> list[Hit]:
        found: list[Hit] = []
        seen: set[str] = set()
        for entry, rx in self._compiled:
            m = rx.search(n.norm)
            if m:
                found.append(Hit(entry["id"], entry["kind"], entry["severity"], m.group(0)))
                seen.add(entry["id"])
        for entry, surface in self._squash_hits(n):
            if entry["id"] in seen:
                continue
            found.append(Hit(entry["id"], entry["kind"], entry["severity"], surface))
            seen.add(entry["id"])
        return found

    def hits_text(self, text: str) -> list[Hit]:
        return self.hits(normalize(text))

    def has_profanity(self, text: str) -> bool:
        return bool(self.hits_text(text))

    def rule_feature(self, n: NormText) -> float:
        """0.0-1.0 score: severe term = 1.0, moderate = 0.7, several hits add up."""
        hs = self.hits(n)
        if not hs:
            return 0.0
        weight = {"severe": 1.0, "moderate": 0.7, "mild": 0.4}
        best = max(weight.get(h.severity, 0.5) for h in hs)
        bonus = min(0.2, 0.05 * (len(hs) - 1))
        return min(1.0, best + bonus)
