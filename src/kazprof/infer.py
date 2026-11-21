"""Inference API: one comment in, one decision out.

The transformer is used when its saved directory exists, otherwise the TF-IDF
baseline is loaded, so the repository works on a machine that only has scikit-learn.

    from kazprof.infer import Detector
    d = Detector()
    d.predict("тауар жақсы, бірақ қызметі қотақ")
    # {'label': 1, 'score': 0.93, 'threshold': 0.42, ...}
"""
from __future__ import annotations

import json
from pathlib import Path

from .encoding import encode_head_tail
from .lexicon import Lexicon
from .normalize import normalize

ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "models" / "kazroberta_kk_profanity"
BASELINE_PATH = ROOT / "models" / "baseline_tfidf_lr.joblib"
THRESHOLD_PATH = ROOT / "models" / "threshold.json"
FINAL_PATH = ROOT / "models" / "final_model.json"


class Detector:
    def __init__(self, backend: str = "auto", threshold: float | None = None, model_dir: Path | None = None) -> None:
        self.lexicon = Lexicon()
        self.backend = backend
        self.threshold = threshold
        self._model = None
        self._tok = None
        self._baseline = None

        # The final model is whatever validation selected, as recorded by
        # scripts/07_evaluate.py. 'auto' obeys that file rather than guessing.
        final_choice = "transformer"
        if FINAL_PATH.exists():
            final_choice = json.loads(FINAL_PATH.read_text(encoding="utf-8")).get("final_model", "transformer")

        if backend == "auto":
            backend = final_choice if (model_dir or MODEL_DIR).exists() or final_choice == "baseline" else "baseline"

        if backend == "transformer" and (model_dir or MODEL_DIR).exists():
            self.backend = "transformer"
            self._load_transformer(model_dir or MODEL_DIR)
        elif BASELINE_PATH.exists():
            self.backend = "baseline"
            import joblib

            blob = joblib.load(BASELINE_PATH)
            self._baseline = blob["pipeline"]
            self.threshold = self.threshold if self.threshold is not None else blob["threshold"]
        else:
            raise FileNotFoundError(
                "No trained model found. Run scripts/05_train_baseline.py or scripts/06_train_transformer.py first."
            )

        if self.threshold is None and THRESHOLD_PATH.exists():
            self.threshold = float(json.loads(THRESHOLD_PATH.read_text())["threshold"])
        self.threshold = 0.5 if self.threshold is None else float(self.threshold)

    def _load_transformer(self, model_dir: Path) -> None:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self._tok = AutoTokenizer.from_pretrained(str(model_dir))
        if self._tok.pad_token is None:
            self._tok.pad_token = self._tok.eos_token
        self._model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
        self._model.eval()
        self._torch = torch

    def score(self, text: str) -> float:
        """Probability of class 1 (profanity present)."""
        if self.backend == "transformer":
            enc = encode_head_tail(self._tok, text)
            with self._torch.no_grad():
                ids = self._torch.tensor([enc["input_ids"]], dtype=self._torch.long)
                mask = self._torch.tensor([enc["attention_mask"]], dtype=self._torch.long)
                logits = self._model(input_ids=ids, attention_mask=mask).logits
                return float(self._torch.softmax(logits, dim=-1)[0, 1])
        return float(self._baseline.predict_proba([normalize(text).norm])[0, 1])

    def predict(self, text: str, explain: bool = False) -> dict:
        n = normalize(text)
        score = self.score(text)
        label = int(score >= self.threshold)
        out = {
            "text": text,
            "label": label,
            "score": round(score, 4),
            "threshold": round(self.threshold, 3),
            "backend": self.backend,
            "lang": n.lang,
            "lexicon_hits": [h.term_id for h in self.lexicon.hits(n)],
        }
        if explain:
            out["normalised"] = n.norm
            out["rule_feature"] = round(self.lexicon.rule_feature(n), 3)
            out["n_tokens"] = len(n.tokens)
        return out

    def predict_many(self, texts: list[str]) -> list[dict]:
        return [self.predict(t) for t in texts]
