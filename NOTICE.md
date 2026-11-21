# Third-party sources and licences

This repository is **for educational, non-commercial use only** because one of its
data sources is licensed CC BY-NC-SA 4.0.

| Component | Source | Licence | What we do with it |
|---|---|---|---|
| `data/raw/telegram_kk_ru.txt` | [kurumikz/telegram-corpus-russian-kazakh](https://huggingface.co/datasets/kurumikz/telegram-corpus-russian-kazakh) | CC BY-NC-SA 4.0 | source of real profanity and clean chat comments |
| `data/raw/reviews_2gis_kk.csv` | [kkenbbi/kazakh_reviews_2gis](https://huggingface.co/datasets/kkenbbi/kazakh_reviews_2gis) | not stated on the dataset card | real Kazakh reviews (negatives) |
| `data/raw/kaspi_reviews.csv` | [datascience2201/kaspi_reviews1](https://huggingface.co/datasets/datascience2201/kaspi_reviews1) | Apache-2.0 (as stated) | real Kazakh product reviews (negatives) |
| `data/raw/swear_terms.jsonl`, `swear_sentences.jsonl` | [Rustem-Kaimolla/kazakh-swear-words](https://huggingface.co/datasets/Rustem-Kaimolla/kazakh-swear-words) | MIT | seed lexicon and 13 seed sentences |
| Base model | [kz-transformers/kaz-roberta-conversational](https://huggingface.co/kz-transformers/kaz-roberta-conversational) | see model card | fine-tuned checkpoint |
| Libraries | PyTorch, Transformers, scikit-learn, Matplotlib | BSD/Apache-2.0 | training and evaluation |

Attribution: the Telegram corpus and both review sets are redistributed here only
as *derived, masked, labelled* rows, not as raw dumps. Raw files are re-downloaded
from the original hosts by `scripts/01_fetch_sources.py`.

If you need a commercially usable version, remove the Telegram corpus
(`--exclude-source telegram_kk` in `04_build_dataset.py`) and retrain; positives then
come only from the seed sentences and templates, and the reported scores no longer
apply.

## AI assistance

Built with an AI coding agent (OpenCode / Claude). Section "AI assistance" in
`README.md` states which parts were AI-authored and which were human decisions.
