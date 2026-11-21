# Saved models

| File | Committed? | Size | How to produce it |
|---|---|---|---|
| `baseline_tfidf_lr.joblib` | yes | 0.72 MB | `python scripts/05_train_baseline.py` |
| `kazroberta_kk_profanity/` | **no** — 318 MB, above GitHub's 100 MB file limit | — | `python scripts/06_train_transformer.py` (~20–25 min on CPU) |
| `threshold.json` | yes | <1 KB | `python scripts/07_evaluate.py` |

`kazroberta_kk_profanity/` contains everything inference needs:

```
config.json              model.safetensors (318 MB)
tokenizer.json           tokenizer_config.json
```

That is the complete set: `AutoTokenizer` loads offline from `tokenizer.json` alone
(verified with `local_files_only=True`), so the directory is self-contained.

The repository is still runnable without it: `kazprof.infer.Detector` falls back to
the committed TF-IDF baseline, which reproduces the baseline numbers in
`reports/metrics.json`. To get the transformer numbers, run `06` and `07`.

Publishing the 318 MB checkpoint: attach it to a GitHub **release** or push it to the
Hugging Face Hub; both keep it out of the git history.
