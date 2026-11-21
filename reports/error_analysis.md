# Error analysis — shipped model (TF-IDF + logistic regression, threshold 0.50)

Test split: 399 rows, 140 positives, 259 negatives.
Confusion matrix: `[[253, 6], [14, 126]]` → 126 TP, 14 FN, 6 FP, 253 TN.
Precision 0.954, recall 0.900, F1 0.927 for the profanity class.

The six false positives and fourteen false negatives are listed verbatim in
[`error_candidates.md`](error_candidates.md). Four are analysed below; each was
reproduced with `scripts/13_explain_features.py`, which prints the n-grams that moved
the decision, so the explanation is measured, not guessed.

---

## Mistake 1 — false positive caused by a template prefix

**Text:** `Заказ келді, но түсі фотоға сәйкес емес.` **Score:** 0.676 → predicted 1, truth 0.
(This row is a synthetic clean negative — the model was *not* trained on it, it sits in test.)

| feature | weight | contribution |
|---|---|---|
| `word__заказ келді` | +0.881 | +0.324 |
| `word__сәйкес емес` | +0.534 | +0.196 |
| `word__но` | −0.313 | −0.095 |

The sentence carries no profanity; the score comes from `заказ келді`, the opening of a
*profane* mixed frame (`Заказ келді, {conn} {pn} брак`). The model keys on the shared
prefix instead of on the profane slot.

**Mechanism — template leakage at frame level.** This is the residue of a bug found
during the audit: every mixed frame used the connector `но`, and the model had learned
`но ⇒ profanity` (weight +2.3). The generator now varies the connector and adds clean
mixed negatives, so `но` flipped to −0.31 — but the frame opener is still shared between
the two classes.

**Fix:** never reuse an opener across classes — generate clean negatives from the *same*
frames with the profane slot removed, and diversify openers (12+ per class). That is a
data-generation change, not a modelling change.

## Mistake 2 — false positive: the model learned the chat register

**Text:** `У нас қалың мал` (*"we have bride-price livestock"*) **Score:** 0.717 → predicted 1, truth 0.

The top contributors are generic chat fragments — `char__ у ` +0.084, `char__с ` +0.071,
`char__ мал` +0.069 — and the negatives that would have suppressed them (`char__лың`,
`char__қа`) are too weak. No profanity is present; `мал` ("livestock", also a mild insult)
is enough to tip a short message.

**Mechanism — register confound + short text.** 443 of 458 real positives are Telegram
chat, most negatives are reviews, so chat-shaped character patterns act as a proxy for
profanity, and a 4-word message has nothing else to dilute them.

**Fix:** domain-balanced sampling (equal chat:review ratio per class) and a minimum-length
guard in production (short comments → manual queue).

## Mistake 3 — false negative: the word is seen but outvoted

**Text:** `Бляя ауылда үй неткен керемет` (*"fuck, the village house is amazing"*)
**Score:** 0.230 → predicted 0, truth 1.

| feature | weight | contribution |
|---|---|---|
| `char__ бля` | +2.114 | +0.230 |
| `char__ля` | +1.711 | +0.176 |
| `word__керемет` | −0.393 | **−0.393** |

The profanity *is* detected — `char__ бля` is the single strongest positive feature — but
`керемет` ("great") cancels it, and the bias term (−1.76) does the rest.

**Mechanism — dilution in L2-normalised TF-IDF plus a positive-sentiment shortcut.** The
model has learned "praise words ⇒ clean", the mirror image of the labelling rule
"negative review ≠ profanity" that we were careful to enforce.

**Fix:** augment positives with stretched spellings (`бляя`, `бляяя`), and add the
lexicon severity as a feature so one severe hit cannot be outvoted by sentiment words.

## Mistake 4 — false negative: agglutination and rare token

**Text:** `Билігіміз орыстарға көт жалай берсін, бірақ қазақ халқы әрқашан осыны есте сақтайды`
**Score:** 0.088 → predicted 0, truth 1.

`char__көт` +0.116 and `char__ көт` +0.116 are present, but `word__осыны` −0.300 and
`word__қазақ` −0.214 dominate. `көт жалау` ("ass-licking") never appears as a training
token; only character n-grams carry the evidence, and they are weak.

**Mechanism:** Kazakh agglutination plus a long formal sentence. Training has `көтін`,
`көтің`, `көті` — not the bare `көт` — and the sentence contains many common words with
negative weights.

**Fix:** morpheme-aware augmentation (attach the eight most frequent case suffixes to each
profane stem) or a character-level model with wider n-grams. The transformer sees subword
pieces and gains recall on exactly this pattern, though it loses precision elsewhere.

---

## What a rule override would change (analysis, not shipped)

OR-ing the model with "the lexicon found a severe term"
(`scripts/12_rule_override_analysis.py`) takes the test set from F1 0.927 to **0.975**
(precision 0.959, recall 0.993), fixing 13 of the 14 misses with no new false positives;
validation agrees (0.945 → 0.990).

It is **not** shipped, for one reason: the positives in this dataset were *mined with that
same lexicon*, so a rule that re-finds them is partly circular. On independently labelled
reviews the gain would be smaller and the recall optimistic. The analysis is kept as
evidence of the failure mode, not as a result to report.

## Summary of what the numbers hide

| Hidden effect | Where it shows up |
|---|---|
| Template leakage through shared frame openers | false positives on clean mixed reviews (`Заказ келді, но …`) |
| Register/domain confound (chat vs review) | false positives on short clean chat (`У нас қалың мал`) |
| Agglutination / unseen inflections | false negatives on `көт жалау`, `сігілгенімді` |
| Single-token dilution in long text | false negatives with one strong word in a positive sentence |
| Positive-sentiment shortcut | false negatives where praise words outvote the profanity |
| Lexicon-derived positives | optimistic recall for both models, and the circularity above |
