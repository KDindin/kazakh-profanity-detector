# Defense cheat-sheet

Everything an examiner is likely to ask, with the answer and the file that proves it.

## 0. Requirements map

| Requirement | Where it is answered |
|---|---|
| Public data, sources, access, labeling method, counts, distribution | `data/DATASET_CARD.md` (§Sources, §At a glance); `scripts/01_fetch_sources.py` |
| Labeling rules: mixed language, masks, spelling variants, ambiguity, insult ≠ profanity | `data/LABELING.md` (8 sections + 10 labelled examples) |
| Synthetic examples clearly identified | `synthetic` and `template_id` fields on every row of `data/processed/*.jsonl` |
| Separate splits, duplicate removal, near-duplicates kept together, sizes, seed | `DATASET_CARD.md` §Split construction; `scripts/04`, `scripts/16_verify_splits.py` |
| Vocabulary / preprocessing fitted on training data only | `tests/test_model_artifacts.py::test_vocabulary_was_fitted_on_train_only` |
| Model choice, text → input, long comments, punctuation, unknown tokens | `README.md` §Model (incl. the edge-case table) |
| Model, dependencies, settings, seed recorded | `reports/training_env.json` |
| Saved model + tokenizer needed for inference | `models/` (`baseline_tfidf_lr.joblib`, `kazroberta_kk_profanity/`) |
| Precision, recall, F1 for class 1, confusion matrix, n | `reports/metrics.json`, `reports/confusion_matrix.png` |
| What FP and FN mean, which errors matter | `README.md` §Evaluation |
| ≥3 inspected mistakes with reasons and fixes | `reports/error_analysis.md` |
| Trace one comment raw → prediction | `scripts/11_trace.py` |
| Small change / new examples | `DEFENSE.md` §6 |
| AI assistance acknowledged | `README.md` §AI assistance |

## 1. Trace one comment from raw text to prediction

Best demo is the script, not the CLI:

```bash
python scripts/11_trace.py "Тамағы дәмді екен, бірақ қызметі қотақ"
```

| Step | Code | What it does to the example |
|---|---|---|
| 1. normalise | `src/kazprof/normalize.py` | NFKC, lowercase, `ё→е`, PII mask, homoglyph fold (`c→с`, `0→о`), repeated letters collapse, punctuation trimmed → `тамағы дәмді екен, бірақ қызметі қотақ` |
| 2. features | `05_train_baseline.py` | word 1–2 + char_wb 2–5 TF-IDF, IDF from training split |
| 3. model | `models/baseline_tfidf_lr.joblib` | logistic regression → P(profanity) = 0.8978 |
| 4. decision | `models/threshold.json` | `label = 1 if score ≥ 0.50` → 1 |
| 5. rule view | `src/kazprof/lexicon.py` | reports `kk_kotak`; used for mining and explainability, not for the decision |
| 6. transformer path (alternative) | `src/kazprof/encoding.py` | same normalised text → BPE → 94 head + 32 tail → KazRoBERTa head |

## 2. Dataset in one paragraph

2,000 labelled rows: 700 positives / 1,300 negatives; 1,820 Kazakh and 180
Kazakh-Russian mixed; 282 synthetic (14.1 %) and marked. Real reviews come from
Kaspi.kz (99) and 2GIS (126); real profanity comes from an unmoderated Telegram
corpus (1,478 rows, 443 in-scope positives that were each read before keeping) plus
15 hand-written seed sentences. Templates put real profane phrases into review
frames. Splits 1,201 / 400 / 399 by near-duplicate cluster (char 3–5-gram cosine
≥ 0.85), seed 42. Labels follow `data/LABELING.md`: obscenity = 1, insults and
negative opinions = 0.

## 3. Model and training objective

**Shipped:** TF-IDF (word 1–2 + char_wb 2–5) + logistic regression, `class_weight=balanced`,
C=8 and threshold 0.50 both chosen on validation; vocabulary/IDF fitted on train only.
Why: it wins the pre-registered validation comparison (0.945 vs 0.918) and the test
comparison (0.927 vs 0.901), is 0.72 MB, trains in 40 s and every decision can be
traced to n-grams.

**Alternative:** fine-tuned `kz-transformers/kaz-roberta-conversational`, 83.5M
parameters, trained on Kazakh conversational text — the closest public base model to
review language. Objective: weighted cross-entropy (negatives 0.77, positives 1.43),
AdamW lr 2e-5, weight decay 0.01, 10 % warmup, batch 16, 4 epochs, grad-clip 1.0,
seed 42, max length 128 (94 head + 32 tail), dynamic per-batch padding, checkpoint by
validation F1. It was trained and evaluated under the same protocol and **lost**
(validation 0.918 vs 0.945, test 0.901 vs 0.927 F1).

Why a transformer was tried: mat is heavily inflected and code-switched (`қотағыма`,
`сігейін`, `нахуй`), and character n-grams also fire on ordinary words (`көтерме`,
`педикюр`). Why it did not win: 1,201 training rows, training loss falls to 0.05, and
the classic model already has the character-level view.

**Rule layer** (20 regex patterns, word + separator-free views): used to mine
candidates, to relabel after review, and to explain a decision. It is not part of the
shipped decision.

## 4. Metrics: what the numbers mean here

* **Precision (class 1)** = of all comments the model flags, the share that really
  contain profanity. Low precision = innocent reviews hidden or punished.
* **Recall (class 1)** = of all truly profane comments, the share caught. Low recall
  = profanity stays visible.
* **F1** = harmonic mean, the single number used to compare settings.
* **Confusion matrix** rows = truth, columns = prediction. `FN` (true 1, pred 0) is
  profanity that slipped through; `FP` (true 0, pred 1) is a clean review flagged.

| Error | Moderation meaning | Cost |
|---|---|---|
| False negative | swear word stays under a product page | reputational, brand-safety, often cheap to fix later (report button) |
| False positive | a normal review is hidden or sent to manual review | author trust, lost review volume, support load |

For a marketplace the usual choice is **high precision with moderate recall**
(auto-hide only confident cases, queue the rest), which is why the threshold table
in `reports/threshold_table.csv` is shipped: it shows what precision costs in recall.
The shipped threshold maximises validation F1 — a deliberate middle ground, not a
production policy.

## 5. Mistakes to discuss (all four are in `reports/error_analysis.md`)

Name the mechanism, not just the example:

1. **Template leakage** — `Заказ келді, но түсі фотоға сәйкес емес.` scores 0.68 from
   `word__заказ келді` (+0.32): the opener is shared with a profane frame. The connector
   `но` itself is now negative (−0.31) after the audit fix, so the frame opener is the
   remaining leak.
2. **Register confound** — `У нас қалың мал` scores 0.72 from generic chat n-grams
   (`char__ у `, `char__с `), not from profanity: positives are chat-heavy.
3. **Dilution / positive-sentiment shortcut** — `Бляя ауылда үй неткен керемет`:
   `char__ бля` is the strongest positive feature (+0.23), but `керемет` (−0.39)
   cancels it. One token cannot win against a long praise sentence.
4. **Agglutination** — `Билігіміз орыстарға көт жалай берсін…`: `көт` is seen (+0.12),
   but `осыны` (−0.30) and `қазақ` (−0.21) outvote it; `көт жалау` is not a training token.

Fixes, in the order I would try them: generate clean negatives from the *same* frames
with the profane slot removed; morpheme augmentation of profane stems;
domain-balanced sampling; a character-level model with wider n-grams.

## 6. Small live change (rehearse this)

Change the operating point without retraining:

```python
from kazprof.infer import Detector
d = Detector(threshold=0.75)          # precision-first
d.predict("Көтерме бағамен алдым, тауар жақсы")
```

Or run a new comment through the baseline instead of the transformer:

```bash
python -m kazprof.predict --backend baseline "Сігейін, тауар келмеді"
```

Or add a missed word to the rule layer and re-mine:

```bash
# data/lexicon/profanity_kk.json -> add {"id": "kk_new", "pattern": "..."}
python scripts/02_mine.py && python scripts/03_build_pool.py && python scripts/04_build_dataset.py
```

## 7. Questions with short answers

**Why not just a profanity word list?** Measured on validation it gets precision 1.000
and recall 0.901 — it never fires on a clean row, but misses one profanity in ten,
because it cannot see `котинды`, `с*ка`, `бляя`, `далбаеб` unless a pattern was written
for them. The rule layer is kept for mining, relabelling and explainability; the
learned model makes the decision.

**Why ship the baseline and not the fine-tuned transformer?** Pre-registered rule:
highest validation F1, and within 0.01 prefer the smaller model. Baseline 0.9446 vs
transformer 0.9176, and test agrees (0.927 vs 0.901 F1; real-only 0.902 vs 0.870). The
transformer is a negative result: at 1,201 rows, fine-tuning 83M parameters overfits and
does not beat character n-grams. The baseline also trains in 40 s and is explainable.

**Why is the test set not used for tuning?** Any choice made after seeing test
results leaks information and inflates the reported score. Threshold and C were
chosen on validation; test was evaluated once, after both models were frozen.

**Why does the dataset have synthetic positives?** Public Kazakh reviews are
moderated, so profanity there is nearly absent (0 hits in 1,250 candidate reviews,
0 in 2,116 2GIS/Kaspi rows). Real profanity exists only in chat. Templates combine
the two. The cost is optimistic recall on generated syntax, which is why
`test_real_only` is reported next to the overall number.

**What about mixed-language comments?** In scope: a Kazakh review with one Russian
mat token is labelled 1. Pure Russian comments were excluded (about 15.9k
candidates) so the model is not silently a Russian profanity detector.

**What is not solved?** Intent (insult vs intensifier), Latin/leet transliteration
beyond homoglyphs, hate speech without profanity, and Russian-only comments.

**Reproduce everything:**

```bash
python scripts/01_fetch_sources.py
python scripts/02_mine.py
python scripts/03_build_pool.py
python scripts/04_build_dataset.py
python scripts/05_train_baseline.py
python scripts/06_train_transformer.py
python scripts/07_evaluate.py
```
