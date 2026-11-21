"""Trace one comment end to end: raw text -> normalisation -> tokens -> score -> label.

    python scripts/11_trace.py "Тамағы дәмді екен, бірақ қызметі қотақ"
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kazprof.infer import Detector  # noqa: E402
from kazprof.lexicon import Lexicon  # noqa: E402
from kazprof.normalize import normalize  # noqa: E402


def main() -> int:
    text = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "Тамағы дәмді екен, бірақ қызметі қотақ"
    n = normalize(text)
    lex = Lexicon()
    det = Detector()

    print(f"1. raw            : {text!r}")
    print(f"2. PII-masked     : {n.masked!r}")
    print(f"3. normalised     : {n.norm!r}")
    print(f"4. tokens         : {n.tokens}")
    print(f"5. detected lang  : {n.lang}")
    print(f"6. lexicon hits   : {[(h.term_id, h.surface, h.severity) for h in lex.hits(n)]}")
    if det.backend == "transformer":
        from kazprof.encoding import encode_head_tail

        enc = encode_head_tail(det._tok, text)
        print(f"7. subword ids    : {enc['n_subwords']} (truncated={enc['truncated']})")
        print(f"8. first 12 ids   : {enc['input_ids'][:12]}")
    else:
        print("7. features       : TF-IDF word 1-2 + char_wb 2-5 fitted on train only")
    out = det.predict(text, explain=True)
    print(f"9. backend        : {det.backend}")
    print(f"10. P(profanity)  : {out['score']}")
    print(f"11. threshold     : {out['threshold']}")
    print(f"12. label         : {out['label']}  ({'profanity' if out['label'] else 'clean'})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
