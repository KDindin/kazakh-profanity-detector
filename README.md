# Kazakh profanity detector

Binary classifier for Kazakh product reviews and comments: **0 = no Kazakh
profanity, 1 = Kazakh profanity present**, with a probability for every decision.

```console
$ python -m kazprof.predict "Тамағы дәмді екен, бірақ қызметі қотақ"
{"text": "Тамағы дәмді екен, бірақ қызметі қотақ", "label": 1, "score": 0.9153,
 "threshold": 0.5, "backend": "baseline", "lang": "kk", "lexicon_hits": ["kk_kotak"]}

$ python -m kazprof.predict "Көтерме бағамен алдым, тауар жақсы, сатушы сыпайы"
{"text": "Көтерме бағамен алдым, тауар жақсы, сатушы сыпайы", "label": 0, "score": 0.0479,
 "threshold": 0.5, "backend": "baseline", "lang": "kk", "lexicon_hits": []}
```

Scope: a comment written in **Kazakh**, or **mixed Kazakh–Russian** the way most
reviews in Kazakhstan are written. Pure-Russian comments are out of scope.
What counts as profanity, what counts only as an insult, and how masked words are
handled is defined in [`data/LABELING.md`](data/LABELING.md).

## Results

Evaluated once on the held-out test split (399 rows: 140 positives, 259 negatives),
after every setting and threshold was frozen.

| Model | Val F1 | Test precision | Test recall | Test F1 | Test n | Real-only F1 (n=343) |
|---|---|---|---|---|---|---|
| **TF-IDF + LR — shipped** | **0.945** | **0.955** | 0.900 | **0.927** | 399 | **0.902** |
| KazRoBERTa fine-tune (alternative) | 0.918 | 0.894 | **0.907** | 0.901 | 399 | 0.870 |

Confusion matrices (rows = truth, columns = prediction, class 1 = profanity):

```
TF-IDF + LR          [[253,   6],      KazRoBERTa      [[244,  15],
                      [ 14, 126]]                       [ 13, 127]]
```

Selection rule, fixed before the test run: highest F1 for class 1 on validation; if
two candidates are within 0.01, the smaller and faster model wins. The baseline wins
on validation (0.945 vs 0.918) and confirms it on test (0.927 vs 0.901), so no
tie-break is needed. The transformer is *not* better on real text either
(0.902 vs 0.870): with 1,201 training rows it overfits (training loss reaches 0.05)
and cannot beat a model that sees character n-grams directly.

Both numbers are reported because the negative result is informative: fine-tuning a
Kazakh base model is not automatically the right move at this data size. The baseline
is also 0.72 MB, trains in 40 seconds and its decisions can be read feature by feature.

Comparison with a model that only applies the rule layer: precision 1.000, recall 0.901
on validation — it never fires on a clean row, but misses one profanity in ten. That is
the trade-off the learned model is meant to improve.

Details: [`reports/metrics.json`](reports/metrics.json),
[`reports/confusion_matrix.png`](reports/confusion_matrix.png),
[`reports/predictions_test.csv`](reports/predictions_test.csv),
[`reports/threshold_table.csv`](reports/threshold_table.csv).

## Data

2,000 labelled rows, **700 positives / 1,300 negatives**, 1,820 Kazakh + 180
Kazakh–Russian mixed, 282 synthetic (14.1 %, every row flagged), split
1,201 / 400 / 399 by near-duplicate cluster, seed 42.

Full source list, access instructions, split sizes, rejected datasets and the
privacy rules are in [`data/DATASET_CARD.md`](data/DATASET_CARD.md). Short version:

| Source | Real rows used | Role |
|---|---|---|
| Kaspi.kz reviews (`datascience2201/kaspi_reviews1`, apache-2.0) | 99 | product-review negatives |
| 2GIS reviews (`kkenbbi/kazakh_reviews_2gis`) | 126 | venue-review negatives |
| Telegram corpus (`kurumikz/telegram-corpus-russian-kazakh`, CC BY-NC-SA) | 1,478 | real profanity + clean chat comments |
| Kazakh swear-words list (`Rustem-Kaimolla/kazakh-swear-words`, MIT) | 15 sentences | seed positives, lexicon |
| Templates (`src/kazprof/synthetic.py`) | 282 | profanity inside review syntax |

A negative review is **not** profanity: «Өте нашар, ақшасына тұрамайды» is a 1-star
review and label 0. An insult without obscenity («ақымақ», «сасық») is also label 0.
Masked spellings (`с*ка`, `қ.о.т.а.қ`, `с у к а`), Latin homoglyphs and Kazakh words
written in Russian orthography (`котинды`, `сыгейн`) are label 1.

Labeled examples (the full rule set and ten examples are in
[`data/LABELING.md`](data/LABELING.md#labelled-examples)):

| Comment | Label | Why |
|---|---|---|
| `Тауар жақсы екен, қызметі қотақ` | **1** | Kazakh obscenity |
| `Бля, кроссовки үш күнде бұзылды` | **1** | mat inside a Kazakh review |
| `с*ка, деньги верните` | **1** | masked mat |
| `Котинды кыс` | **1** | Kazakh obscenity in Russian spelling |
| `Сатушы ақымақ сияқты әрекет етті` | **0** | insult, no obscenity |
| `Өте нашар випті алғанбыз, мұздай` | **0** | negative review only |
| `Көтерме бағамен алдым, тауар жақсы` | **0** | homograph `көтер` |
| `Пікіріңізді балағат сөзсіз жазыңыз` | **0** | talks about profanity, contains none |

Labels inherited from other datasets were not trusted: the public swear-word list marks
`ақымақ` (fool) as offensive, we relabel it 0; the 2GIS file ships a *sentiment* label
and the Kaspi file a star rating — neither is used, every row was re-labelled with the
rules above (see [`data/DATASET_CARD.md`](data/DATASET_CARD.md#labels-inherited-from-other-datasets)).

## Model

**Shipped: TF-IDF (word 1–2 + char_wb 2–5) + logistic regression**, `class_weight=balanced`,
C=8 chosen by validation, vocabulary and IDF fitted on the training split only,
threshold 0.50 tuned on validation. It is selected because it wins the pre-registered
validation comparison and is 0.72 MB / 40 s to train / readable feature by feature
(`scripts/13_explain_features.py` prints which n-grams moved a decision).

**Alternative: fine-tuned `kz-transformers/kaz-roberta-conversational`** (83.5M
parameters, Kazakh conversational base) with a 2-way classification head. It was
trained and evaluated under the same protocol and lost to the baseline on validation
and on test (0.901 vs 0.927 F1): at 1,201 training rows it overfits and its contextual
advantage never materialises. It is kept because the comparison is part of the answer,
not because it ships.

Why a contextual model was tried at all: Kazakh is agglutinative and code-switched
(`қотағыма`, `сігейін`, `нахуй` are one family; `көтерме`, `педикюр`, `бокс` are
ordinary words that share letter sequences), and pretraining should in principle
supply that language model. Why it did not win: the classic model already gets the
character-level view, and 1,201 rows are not enough to fine-tune 83M parameters.

| Setting (transformer) | Value |
|---|---|
| Objective | weighted cross-entropy (neg 0.77 / pos 1.43) |
| Optimiser | AdamW, lr 2e-5, weight decay 0.01, 10 % warmup, grad-clip 1.0 |
| Batch / epochs | 16 / 4, checkpoint by validation F1 |
| Max length | 128 subwords, **94 head + 32 tail** (profanity often sits at the end) |
| Padding | dynamic per batch (median comment is ~25 subwords) |
| Seed | 42 |

Text pipeline (identical in training and inference, `src/kazprof/`):

1. `normalize.py` — NFKC, lowercase, `ё→е`, PII masking, Latin/digit homoglyph fold,
   repeated-letter collapse (`сууука → сука`), punctuation trimmed to a light set.
2. `encoding.py` — BPE; long comments cut head+tail; unknown words fall back to BPE
   subwords, so `котинды` still decomposes into seen pieces.
3. `lexicon.py` — 20-pattern rule layer (word view + separator-free "squash" view),
   used for mining, candidate review and as an explainable signal; it does not decide
   the label.

How the shipped model handles the edge cases the task asks about:

| Case | Handling |
|---|---|
| Long comment | TF-IDF has no length limit: all tokens and character n-grams are used; the transformer keeps the first 94 and last 32 subwords |
| Punctuation | trimmed to `.,!?;:-'"()*` and kept for word boundaries; the `*` is kept because it appears in masked profanity (`с*ка`) |
| Unknown word | word vocabulary (train-only, `min_df=2`) ignores it, but character 2–5-grams still fire — `котинды` and `қотағыма` share n-grams with seen forms; the transformer falls back to BPE subwords |
| Emoji / URLs / @mentions | removed or replaced with `<url>`, `<user>`, so they neither leak PII nor add noise |
| Empty / very short text | normaliser returns an empty string, vectoriser yields a zero vector, score equals the bias (≈0.03) → label 0 |
| Latin lookalikes | folded to Cyrillic only inside tokens that already contain Cyrillic, so `cука` becomes `сука` but English `ok` is untouched |

## Evaluation

One pass on the held-out test set, after all settings were frozen. Precision,
recall and F1 are reported **for the profanity class (1)**, with the confusion
matrix and the number of test examples (399). Because 14 % of the test set is
synthetic, metrics are also broken down for the real-only subset (343 rows).

What the errors mean in moderation terms:

* **False negative** — profanity stays published under a product page. Reputation
  and brand-safety cost; usually discovered late by a user report. The shipped model
  has 14 of them (recall 0.90).
* **False positive** — a clean, often angry review is hidden or queued for manual
  review. Trust and review-volume cost, plus support work. The shipped model has 6.

`reports/threshold_table.csv` lists precision/recall at thresholds 0.20–0.85 so the
team can move the operating point without retraining: e.g. auto-hide only when
precision is high, send the middle band to humans.

### Error analysis

[`reports/error_analysis.md`](reports/error_analysis.md) inspects four mistakes with
the actual n-grams that drove each decision:

1. **FP** — `Заказ келді, но түсі фотоға сәйкес емес.` scores 0.68 from
   `word__заказ келді` (+0.32), the opener of a *profane* template frame. Template
   leakage at frame level; `но` itself is now negative (−0.31) after the fix.
2. **FP** — `У нас қалың мал` scores 0.72 from generic chat n-grams, not profanity:
   short chat messages are over-represented among positives, so register acts as a proxy.
3. **FN** — `Бляя ауылда үй неткен керемет` scores 0.23: `char__ бля` is the strongest
   positive feature (+0.23) but `керемет` cancels it (−0.39). Positive-sentiment shortcut.
4. **FN** — `Билігіміз орыстарға көт жалай берсін…` scores 0.09: `көт` is seen
   (+0.12), but `осыны` (−0.30) and `қазақ` (−0.21) outvote it; `көт жалау` never
   appears as a token in training (agglutination).

A measured follow-up (`scripts/12_rule_override_analysis.py`): OR-ing the model with
"the lexicon found a severe term" raises test F1 from 0.927 to 0.975. It is **not
shipped**, because the positives were mined with that same lexicon, which makes the
gain partly circular — an important caveat rather than a result.

## Limitations

* 35 % of positives are generated from templates, so recall on generated syntax is
  optimistic; the real-only test block is the honest number.
* **Mining bias.** Positives were found with the same lexicon that the error analysis
  uses as a ceiling, so both models look better than they would on independently
  labelled reviews — and a rule override looks better still.
* **Register confound.** 443 of 458 real positives are Telegram chat while most
  negatives are reviews, so the model can partly separate *chat register* from
  *review register* instead of profanity (see false positive 3 in the error analysis).
* Telegram chat is not a marketplace review; Kaspi/2GIS are. Domain shift is real.
* Intent is not modelled: mat used as an intensifier («Ебать, суық екен») is labelled
  1 by the definition, and quoted profanity is labelled 1 too.
* Latin/leet transliteration beyond single-character homoglyphs is not covered.
* Hate speech, threats and spam without profanity are out of scope.
* The Telegram source is CC BY-NC-SA 4.0 → **non-commercial use only**.

## Repository layout

```
data/
  LABELING.md            labeling rules, revisions, labelled examples
  DATASET_CARD.md        sources, counts, splits, privacy, rejected datasets
  lexicon/profanity_kk.json   20-pattern rule layer
  processed/             train.jsonl, val.jsonl, test.jsonl, dataset_stats.json
scripts/
  01_fetch_sources.py    download the four public sources
  02_mine.py             mine profanity candidates from the Telegram corpus
  03_build_pool.py       merge sources, PII mask, label by rule, filter scope
  04_build_dataset.py    balance, dedupe, near-dup clusters, 60/20/20 split
  05_train_baseline.py   TF-IDF + logistic regression
  06_train_transformer.py fine-tune KazRoBERTa
  07_evaluate.py         tune on validation, evaluate once on test
  08_error_analysis.py   FP/FN candidate list for the written analysis
  09_lexicon_check.py    pattern regression check (46 controls)
  10_show_examples.py    print labelled rows per split
  11_trace.py            end-to-end trace of one comment
  12_rule_override_analysis.py  measured rule-override experiment (analysis only)
  13_explain_features.py which n-grams moved a baseline decision
  14_pii_audit.py        scans every published text for PII patterns
  15_record_env.py       writes reports/training_env.json
  16_verify_splits.py    proves no near-duplicate pair crosses a split
tests/                   normaliser, lexicon, dataset, inference and artifact tests
src/kazprof/
  normalize.py  encoding.py  lexicon.py  synthetic.py  infer.py  predict.py
models/                baseline_tfidf_lr.joblib, threshold.json, (transformer/)
reports/               metrics.json, confusion_matrix.png, predictions_test.csv,
                       threshold_table.csv, transformer_train.json, training_env.json,
                       rule_override_analysis.json, error_analysis.md
```

## Reproduce

```bash
python -m venv .venv && . .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .                                   # makes `python -m kazprof...` work
pip install pytest                                 # only needed for the test suite
python scripts/01_fetch_sources.py
python scripts/02_mine.py
python scripts/03_build_pool.py
python scripts/04_build_dataset.py
python scripts/05_train_baseline.py
python scripts/06_train_transformer.py            # ~20-25 min on CPU (4 epochs)
python scripts/07_evaluate.py
python scripts/15_record_env.py
python -m pytest tests -q                          # 26 checks
```

Windows shortcut: `powershell -ExecutionPolicy Bypass -File run_all.ps1`.
Linux/macOS: `bash run_all.sh`.

## Verification

Every claim in this README is checkable without retraining:

```bash
python -m pytest tests -q                 # 26 tests: normaliser, lexicon, dataset,
                                          # inference contract, train-only vocabulary,
                                          # self-contained tokenizer
python scripts/09_lexicon_check.py         # 46 pattern controls (37 must fire, 9 must not)
python scripts/14_pii_audit.py             # 0 PII findings over 2,399 published texts
python scripts/16_verify_splits.py         # max cross-split similarity 0.849 < 0.85
python scripts/11_trace.py "Тамағы дәмді екен, бірақ қызметі қотақ"
python scripts/13_explain_features.py "Бляя ауылда үй неткен керемет"
```

`reports/training_env.json` records the Python version, every dependency version, all
seeds, the dataset shape and the exact hyper-parameters of the two reported runs.

Determinism: every split, generation and training run uses seed 42; the dataset build
is deterministic given the same raw files. The transformer checkpoint is not committed
(318 MB, over GitHub's file limit) — see [`models/README.md`](models/README.md).

## AI assistance

This project was built in an AI-assisted session (OpenCode with a Claude-based agent).
Being explicit about the division of work, because the assignment asks for it:

| Produced by the agent | Set or owned by the author |
|---|---|
| all scripts and modules, the normaliser, training loop, plots, tests | the task interpretation and the decision to scope "Kazakh + Kazakh–Russian mixed" |
| first draft of the lexicon patterns | approving which sources to use, and rejecting the machine-generated review set |
| documentation drafts (README, dataset card, labeling rules) | the labelling policy: obscenity = 1, insults = 0, negative ≠ profanity |
| proposing the two candidate models and the selection rule | the decision to ship the baseline and to present the transformer as the production option |
| enumerating candidate mistakes for the error analysis | reading the mined positives, judging ambiguous cases, writing the conclusions |
| debugging (Ukrainian leakage, squash false positives, PII links, train-only vocabulary test) | publishing the repository and defending every claim live |

The author is expected to be able to explain and defend every number and every rule
in this repository; the table is a record of who did what, not a transfer of
responsibility.

## Licence

CC BY-NC-SA 4.0 — educational, non-commercial use only. See [`LICENSE`](LICENSE),
[`NOTICE.md`](NOTICE.md).
