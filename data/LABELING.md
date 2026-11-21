# Labeling rules — Kazakh profanity

The task is **one comment or review in, one binary label out**:

* `1` — the text contains at least one Kazakh or Russian profane expression, as used by Kazakh speakers;
* `0` — no profanity.

Everything below was applied by the same person for every row, and the rule that
decided a row is recorded in the row's `source`/`origin` fields. Where a rule was
changed while labeling (the lexicon grew twice), the change is listed at the end.

## 1. What counts as profanity

| Category | Examples (roots, not an exhaustive list) | Label |
|---|---|---|
| Kazakh obscenity | қотақ, қотақбас, ам, сігу/сігу family, боқ, көт, қаншық, жәлеп | 1 |
| Russian mat used in a Kazakh/mixed comment | хуй, пизда, блядь/бля, ебать family, сука, мудak, пидор | 1 |
| Obscene insult | далбаеб, еблан, долбоеб | 1 |
| Insult **without** obscenity | ақымақ, сасық, итпұшық, қоқыс, арсыз, жынды, есер, даун | **0** |
| Strong negative opinion | «тамағы дәмсіз», «қызмет нашар», «ақшамды қайтарыңдар» | **0** |

The distinction between an insult and profanity is deliberate. The public
`Rustem-Kaimolla/kazakh-swear-words` list marks `ақымақ` (fool) and `сасық`
(stinky) as `offensive`; we relabel them **0** because they carry no obscenity.
That is a revision of the source labels, and it is the reason the "insult" rows in
our dataset exist: a model that flags every angry review is useless.

## 2. Negative review is not profanity

«Өте нашар випті алғанбыз, мұздай» — 1 star, no profanity → **0**.
«Масқара, ақшасына тұрамайды, алмаңыздар» — angry, still no obscenity → **0**.

## 3. Masked and obfuscated spellings

Any masking that a reader would still understand as profanity is labelled **1**:

* `с*ка`, `бл**ь`, `қ.о.т.а.қ`, `с у к а` (letters separated by punctuation or spaces);
* Latin homoglyphs (`cука`, `пиздeц` with Latin `e`), digit substitutions (`0` for `о`);
* stretched letters (`сууука`, `бляяя`);
* Kazakh words typed in Russian orthography (`котак`, `котинды`, `сыгейн`, `нахүй`).

The normaliser maps all of these onto the same token before the model sees them,
and the rule layer matches both the word view and a "squashed" view with all
separators removed.

## 4. Mixed-language comments

The input domain is a Kazakh comment **or a Kazakh–Russian mixed comment**, which
is how most reviews in Kazakhstan are actually written. The label follows the
content, never the majority language:

* `Тамағы дәмді, бірақ қызметі қотақ` → **1** (one profane token in a Kazakh review);
* `Доставка кешікті, но упаковка хуйня` → **1** (mixed frame + mat);
* `Доставка кешікті, тауар жоқ` → **0**.

Comments written **entirely** in Russian with no Kazakh token are out of scope and
were dropped from the dataset (about 15,900 Telegram candidates). A Russian mat
word inside a Kazakh/mixed comment is in scope.

## 5. Quotes, mentions and meta-discussion

* The comment reproduces the word, even as a quotation: `«әкеннің аузын сігейін» деп жазды` → **1**.
  A moderation queue sees the word regardless of intent.
* The comment talks about profanity without writing any: `Пікіріңізді балағат сөзсіз жазыңыз` → **0**.

## 6. Homographs and substrings (the false-positive traps)

`көтерме баға`, `педикюр`, `бокс қолғабы`, `амандық`, `самал жел`, `көтеріңкі көңіл-күй`
contain letter sequences that look like profanity roots but are ordinary words → **0**.
The lexicon uses word boundaries and a dense-token rule for the squash view so
`де` + `баланың` cannot accidentally form a profane string.

## 7. Out of scope (documented, not labelled)

Hate speech, threats, sexual harassment and spam are separate moderation classes.
If such a text contains no profanity it is **0** here, even when it should be
removed from a marketplace for another reason.

## 8. Ambiguous cases and how they were decided

| Case | Decision | Reason |
|---|---|---|
| Obscene word used literally, not as an insult (`көтім ауырып тұр`) | **1** | the label asks whether the obscenity is present, not whether it is directed at someone |
| Mild obscenity `боқ` ("shit") | **1**, severity `moderate` | still obscene vocabulary; severity is recorded so a stricter policy can filter it |
| Pun or wordplay on a profane root (`Америка – ам ерік ә`) | **1** | a reader and a moderation filter both see the word |
| Insult + profanity in one sentence | **1** | profanity dominates; the insult is irrelevant once the obscenity is present |
| Profanity written in Latin (`suka`, `qotaq`) | **1 by rule**, but excluded from the dataset | the model has no tokenisation for it; augmenting would be needed before claiming coverage |
| Kazakh word that *looks* profane after transliteration (`сасық`, `қоқыс`, `көтерме`) | **0** | ordinary lexicon; the rule layer uses word boundaries to avoid them |
| A slur used as a joke between friends | **1** | intent is not modelled; the definition is presence-based |
| Truncated message that ends mid-word (`Требую до деп`) | kept whatever its label was, flagged in the error analysis | corpus artefact, not a labelling decision |

## Labelled examples

| Text | Label | Why |
|---|---|---|
| `Тауар жақсы екен, қызметі қотақ` | 1 | Kazakh obscenity |
| `Бля, кроссовки үш күнде бұзылды` | 1 | mat inside a Kazakh review |
| `Сігейін, тағы да дұрыс емес келді` | 1 | Kazakh obscenity |
| `Сатушы ақымақ сияқты әрекет етті` | 0 | insult, no obscenity |
| `Өте нашар випті алғанбыз, мұздай` | 0 | negative review only |
| `Көтерме бағамен алдым, тауар жақсы` | 0 | homograph `көтер` |
| `Педикюрге бардым, қызмет жақсы` | 0 | homograph `педик` |
| `Пікіріңізді балағат сөзсіз жазыңыз` | 0 | meta-discussion, no word |
| `с*ка, деньги верните` | 1 | masked mat |
| `Котинды кыс` | 1 | Russian-orthography Kazakh obscenity |

## How labeling was actually performed

1. A rule pass (lexicon + normaliser) labelled every candidate; it produced 448
   in-scope Telegram hits (420 Kazakh, 28 mixed) out of 870,759 corpus lines.
2. Every in-scope positive was read and kept or dropped. Kept rows are the
   `telegram_kk` and `swear_sentences_seed` positives; rows judged to be false hits
   were dropped (for example `педикюр` before the word-boundary fix, and Ukrainian
   comments that the first Kazakh detector misclassified because both languages use `і`).
3. A random sample of 120 negatives was read to check for missed profanity; it found
   three (`Амынды шығарамын`, `далбан`, `Блә`), each of which led to a lexicon fix
   and a re-run of the whole pipeline.
4. Synthetic rows were generated, never hand-written, and are marked.

## Revisions made during labeling

| Round | Change | Reason |
|---|---|---|
| 1 | `педик` matched `педикюр` | false positive on manicure reviews |
| 2 | squash matching bridged two words (`де`+`баланың` → `ебал`) | false positives; added token-density rule |
| 3 | added `боқ`, `көт` | real obscenities missing from the source list |
| 4 | added `ынды` to the `ам` pattern | `Амынды шығарамын` was labelled 0 |
| 5 | added `далбаеб/далбаб`, `блә` | misspellings of `долбоеб` and `бля` |
| 6 | Ukrainian detection (ї/є/ґ + function words) | a 60-row Ukrainian segment was misread as Kazakh |
| 7 | dropped 3 of 63 seed sentences (`Щщс`, `667`, `Салдақы`) | not profane |
| 8 | added `сіге`, `пизля`, `підорас` spellings | test error analysis found 4 rows labelled 0 that do contain profanity; whole pipeline re-run |
