# Building your own golden set

The golden set (`eval/golden_set.jsonl`) is the list of questions the agent is scored on, each with the expected answer and the place in the transcript that supports it. Its quotes match the text of one transcript provider. If you pull from a different provider, or choose different calls, write your own golden set with the steps below. None of these steps make model calls.

## 1. Choose the calls

Edit `corpus/calls.json`. Each entry is one call:

```json
{"ticker": "AMD", "label": "2026Q2", "fiscal_year": 2026, "fiscal_quarter": 2}
```

- `label` names the files (`AMD_2026Q2.json`) and is the `quarter` in golden-set rows.
- `fiscal_year` and `fiscal_quarter` are the period the call reports, in the company's own fiscal calendar. For companies whose fiscal year does not follow the calendar year, they can differ from the calendar quarter.
- `frozen` is the date after which no new calls are added, so results stay comparable.

Then pull the transcripts (see the README, "Evaluation corpus"). Each file in `corpus/v2/raw/` is JSON with a `data` list of turns:

```json
{"speakerName": "Jean Hu", "speakerRole": "CFO", "startSeconds": 1329.01, "text": "..."}
```

`speakerName` and `startSeconds` can be `null`. With another provider, adapt `scripts/transcript_source.py` so it writes this shape.

## 2. Write the questions

Aim for several questions per call and a spread of kinds:

| `kind` | Asks | Example |
|---|---|---|
| `fact` | A reported number or event | What revenue did AMD report for Q2 2026? |
| `reason` | Why management says something happened | Why did gross margin expand? |
| `guidance` | What management expects next | What revenue did AMD guide for Q3 2026? |
| `assumption` | What management assumes behind the outlook | What does the outlook assume about export rules? |
| `not_discussed` | A topic the call never covers (see step 4) | What dividend did AMD declare? |

Ask about what management said, not about the stock. The project gives no investment advice, so the golden set has no "should I buy" questions.

## 3. Find the passage for an answerable question

List the turns of a call, then search for the words that answer your question:

```bash
python eval/find_passage.py AMD 2026Q2 --list
python eval/find_passage.py AMD 2026Q2 "gross margin"
```

Each match prints a `passage` object with the turn number, speaker, role and start second already filled in, and the matching sentence as the quote. Shorten the quote to the words that carry the answer. It must stay word for word from the transcript and long enough to appear in only one turn. Keep it under 200 characters: the golden set is public and the transcripts are not.

A full answerable row (one line in the file):

```json
{"id": "AMD-01", "ticker": "AMD", "quarter": "2026Q2", "answerable": true, "kind": "fact",
 "question": "What revenue did AMD report for Q2 2026, and how did it grow?",
 "expected_answer": "A record $11.5 billion, up 50% year over year and 13% sequentially.",
 "key_facts": ["$11.5 billion", "50% year over year", "13% sequentially"],
 "passage": {"turn": 3, "speaker": "Jean Hu", "role": "CFO", "start_seconds": 1329,
             "quote": "Revenue increased 50% year-over-year and 13% sequentially to a record $11.5 billion"}}
```

- `expected_answer`: what a correct answer says, in your own words.
- `key_facts`: the short facts the judge looks for. A "correct" answer has all of them, a "partial" one has some.

## 4. Write the unanswerable questions

Each unanswerable question names a ticker and quarter in the corpus and asks about a topic that call never covers. The right answer is "not discussed", with no citation. These questions measure whether the agent declines to answer or makes something up.

Choose a topic, then search the whole call with every way the topic could be phrased:

```bash
python eval/find_passage.py AMD 2026Q1 --absent "dividend|buyback|repurchas|capital return|payout"
```

If there are any hits, the call discusses the topic: pick another. With no hits, the command prints the `absence_check` object for the row:

```json
{"id": "NA-04", "ticker": "AMD", "quarter": "2026Q1", "answerable": false, "kind": "not_discussed",
 "question": "What dividend did AMD declare on its Q1 2026 call?",
 "expected_answer": "Not discussed on the call. The answer should say so and cite nothing.",
 "key_facts": [], "passage": null,
 "absence_check": {"pattern": "dividend|stock split|capital return|payout", "hits": 0}}
```

## 5. Check the set

```bash
python eval/check_golden.py                                   # eval/golden_set.jsonl vs corpus/v2/raw/
python eval/check_golden.py --golden my_set.jsonl --corpus path/to/raw
```

The checker verifies every row against the transcripts: required fields, unique ids, that each passage's speaker, role and start second match its turn, that each quote is word for word in that turn and in no other, and that each absence pattern has no hits. It prints the problems and the number of questions per ticker and kind, and exits with code 1 if any row fails.

The checker cannot tell whether the expected answer is right. Read each passage and check the expected answer against it by hand. A model can help draft questions, but a golden set whose answers a model wrote is grading the model against itself.

## 6. Run the evaluation

See the docstring of `eval/run_eval.py`. A run makes model calls on Amazon Bedrock and writes its scores to `eval/results/`.

`tests/fixtures/` has a small invented call and two rows in this format, used by the tests and useful as a minimal example.
