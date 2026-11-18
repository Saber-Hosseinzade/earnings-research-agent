# Conversation design

How the research assistant behaves in a conversation: who it is, what it answers, and seven flows with the
expected behaviour for each. The multi-turn evaluation set is built from this document: one or more scripted
conversations per flow, scored per session.

Each flow carries a status: **built** (in the code and tested), **partly built** or **planned**. Example
dialogues are illustrative. Answers are shown as placeholders such as `[answer]` rather than real transcript
content, and passage IDs follow the real format (`AMD_2026Q2_t003_p1` = AMD, 2026Q2 call, turn 3, part 1).

## Scope

The assistant answers questions about **what company management said on earnings calls**: results, the
reasons they gave, what they assumed, guidance, and how any of these changed between quarters.

It works over a **fixed corpus** of 24 calls: AMD, RIVN, ROOT, META, CRDO and NBIS, four quarters each
(`corpus/calls.json`). The corpus is fixed so that evaluation is repeatable. Quarters are the company's own
fiscal quarters, so the windows differ: CRDO's fiscal year ends in May, and AMD, ROOT, CRDO and NBIS end one
quarter earlier than RIVN and META.

It does **not**:
- give buy, sell or hold recommendations, price targets, valuation opinions or portfolio advice;
- use stock prices or market data (it has none, so "is it cheap?" has nothing to stand on);
- answer from the model's general knowledge: if the calls don't say it, the assistant doesn't either.

It is a research aid. Its answers can be wrong; every factual answer cites the passages it relies on so that
the reader can check.

## Persona and tone

A careful research assistant reporting what management said, not an analyst with views.

- **Short.** Two or three sentences per answer, with the figures and reasons management gave.
- **Attributed.** Say who said it and in which call ("the CFO said on the Q2 2026 call…"). Management's
  forecasts are their forecasts, not facts: "management expects", never "revenue will".
- **Cited.** Every factual answer lists the passage IDs it relies on. No citation, no factual claim.
- **Plain about gaps.** "The Q2 2026 call didn't discuss this" is a complete answer. No guessing, no padding.
- **Neutral.** No adjectives that judge the company ("strong", "disappointing") unless quoting management,
  and then attributed.
- **One disclaimer, in the UI.** The interface shows "research aid, can be wrong, not investment advice"
  once. Answers do not repeat it every turn; refusals explain themselves in one sentence.

## Anatomy of a turn

What runs for every message, in order (`agent/answer.py`):

1. **Request check:** a question is required; a quarter (`2026Q2`) only together with a ticker.
2. **Guardrail on the question** (`agent/guard.py`, Bedrock Guardrails): investment-advice requests and prompt
   attacks are blocked here and the agent does not run. Personal data (email, phone, card and account numbers)
   is masked; the agent only ever sees the masked question.
3. **Agent loop:** Nova 2 Lite with one tool, `retrieve_passages` (Bedrock Knowledge Base, one passage per
   speaker turn). At most two searches; a repeated or third search is refused in code. Run budget: model
   calls, tool calls and tokens.
4. **Citation check in code** (`agent/citations.py`): every cited passage ID must have been returned by a
   search in this turn. A failed check is flagged, not hidden.
5. **Guardrail on the answer:** advice in the answer → replaced; personal data masked; grounding of the answer
   in the cited passages scored, and a low score flagged (answer kept).
6. **Audit record** (`agent/audit.py`): one JSON record per turn, kept 30 days.

Session memory (`agent/session.py`, AgentCore Memory) runs around these steps: before step 1 the session's
last turns are loaded and the message is resolved to a company and quarter (context rules below); after
step 6 the turn is stored. Planned: the approval pause inside step 3 (flow 6).

## Messages

Fixed texts, so that tests and the UI can rely on them.

| When | Text | Source |
|---|---|---|
| Question asks for investment advice | I only report what management said on earnings calls. I can't give buy, sell or hold recommendations, price targets or investment advice. Try asking what management said about a topic. | Guardrail, blocked input |
| Answer reads as investment advice | I can't give that answer: it read as investment advice. I only report what management said on earnings calls. | Guardrail, blocked output |
| Prompt attack | I can't follow instructions to change how I work. Ask a question about what management said on an earnings call. | `agent/guard.py` |
| Guardrail unavailable | The safety check could not run, so this question was not answered. Please try again. | `agent/guard.py` |
| No company known (flow 2) | Which company do you mean? I have calls for AMD, RIVN, ROOT, META, CRDO and NBIS. | `agent/session.py` |
| Unknown ticker (flow 5) | I don't have earnings calls for XYZ. I can answer about AMD, RIVN, ROOT, META, CRDO, NBIS. | `agent/session.py` |
| Quarter not in the corpus (flow 5) | The corpus has AMD's calls for Q3 2025 to Q2 2026, not Q3 2026. Do you want the Q2 2026 call? | `agent/session.py` |
| Not discussed | One sentence, written by the model, saying the calls searched did not discuss it; `covered=false`, no citations | System prompt |

## Context rules

How a message is resolved to a company and quarter, in code (`agent/session.py`), before the agent runs. The
resolved company and quarter pin the search filter. The audit record says which rule applied (`context.how`).

1. A ticker or company name in the message wins. "And META?" switches company.
2. Otherwise the ticker and quarter of the previous turn carry over. "And margins?" after a question about
   AMD's Q2 2026 call means AMD, Q2 2026.
3. Relative quarters resolve against the corpus, not the calendar. "Last quarter" means the latest quarter
   in the corpus for that ticker, which is not the same quarter for every company.
4. Cross-company questions are allowed without a ticker ("which companies talked about tariffs?"); they
   search all calls and the answer names each company it cites. Two companies named, or a comparison with a
   company other than the previous one, counts as cross-company. A comparison about one company ("how did
   that compare with the quarter before?") keeps the company and searches all its quarters.
5. A clarifying question is asked only when the question needs one specific company or quarter and neither
   the message nor the session supplies it (flow 2). At most one clarifying question per turn.

## Flows

### 1. Happy path: question answered with cited passages

**Status: built.** Measured on the 30 answerable golden-set questions (`eval/`).

> **User:** Why did AMD's gross margin change in Q2 2026?
> **Assistant:** [Two or three sentences: the change and the reasons management gave, attributed to the speaker.]
> *Cited: AMD_2026Q2_t…*

Expected:
- One or two searches, filtered to the ticker and quarter when given.
- Answer from the returned passages only, with the figures and reasons management gave.
- Every cited ID was returned in this turn (citation check passes); the cited passages support the answer.

### 2. Missing slot: one clarifying question, not a guess

**Status: built** (in code, `session.clarify_reply`). Multi-turn set: 3/3 in both runs (18 Nov).

> **User:** What did they say about gross margin?
> **Assistant:** Which company do you mean? I have calls for AMD, RIVN, ROOT, META, CRDO and NBIS.
> **User:** AMD
> **Assistant:** [Answer about AMD's latest call in the corpus, naming the quarter.] *Cited: …*

Expected:
- A clarifying question only when the question needs one company and none is known (context rule 5).
- One question, naming the options. No search before the answer arrives.
- A question without a quarter does not need clarifying: the latest quarter in the corpus is used and named
  in the answer ("on the Q2 2026 call…").

Design note: first planned as a prompt rule, since "they" and "the company" can't be resolved in code. Measured
18 Nov: Nova 2 Lite ignored the rule in both runs and searched all calls, answering about whichever company came
back. Moved to code: when no company is known (none in the message or the session, and not a cross-company
question) and the question points at one without naming it ("they", "their", "the company", "management",
"its"), a fixed question is returned with no search and no model call. Wording the pattern doesn't catch still
goes to the agent. A company named without a quarter, and none in context, gets its latest quarter in the
corpus (`session.resolve`).

### 3. Follow-up: resolved against the session

**Status: built** (AgentCore Memory, short-term; `agent/session.py`). Tested live in one scripted
conversation; the multi-turn set is still to come.

> **User:** Why did RIVN's average selling price change in Q3 2026?
> **Assistant:** [Answer.] *Cited: RIVN_2026Q3_t…*
> **User:** And how did that compare with the quarter before?
> **Assistant:** [Answer comparing the two calls, naming both quarters.] *Cited: RIVN_2026Q3_t…, RIVN_2026Q2_t…*

Expected:
- "That" and "the quarter before" resolve against the previous turn (context rules 2 and 3).
- The search filter follows the resolved ticker and quarter; a cross-quarter question searches both calls.
- Earlier answers are context only: new claims still need passages returned in this turn.

What is stored per turn: the question as the agent saw it (masked if it held personal data), the answer,
covered, ticker, quarter and cited passage IDs. No passage text. Blocked and failed turns are not stored, so
they don't steer later turns. Events expire after 7 days, and destroying the runtime stack deletes them all.

A short follow-up can read as an advice request on its own ("What about Meta?"). When the guardrail blocks a
follow-up only as advice, it is checked once more with the previous question in front of it; a real advice
request ("Should I buy it?") is still blocked there. Both checks are in the audit record.

### 4. Advice request: refuse in one sentence, redirect

**Status: built** (guardrail, both directions).

> **User:** Should I buy AMD after this call?
> **Assistant:** I only report what management said on earnings calls. I can't give buy, sell or hold
> recommendations, price targets or investment advice. Try asking what management said about a topic.

Expected:
- Blocked before the agent runs; the turn is recorded with the reason (`topic:InvestmentAdvice`).
- Near-misses are **not** refused: "what did management say about the buyback?", "did management comment on
  the stock's valuation?" are answered from the calls (tested in `eval/guardrail_set.jsonl`).
- If advice slips into an answer, the output guardrail replaces it.

The reply is the fixed guardrail message, not one the agent writes (decision 2): the hard rule is enforced
outside the model, and an advice request never reaches it. The cost is a generic redirect ("Try asking what
management said about a topic") rather than one tailored to the question.

### 5. Tool failure or unknown ticker: say so, offer what is possible

**Status: built** (corpus check in code, `agent/session.py`). Search failures are handled by the tool contract.

> **User:** What did XYZQ say about margins?
> **Assistant:** I don't have earnings calls for XYZQ. I can answer about AMD, RIVN, ROOT, META, CRDO and NBIS.

> **User:** What did AMD say about margins in Q3 2026?
> **Assistant:** The corpus has AMD's calls for Q3 2025 to Q2 2026, not Q3 2026. Do you want the Q2 2026 call?

Expected:
- Search failure (`retrieve_failed`, not retryable): the assistant says the search failed and does not answer
  from memory. Built: tool contract and prompt.
- Ticker not in the corpus: checked **in code** against `corpus/calls.json` before the agent runs (decision 3),
  and answered with the fixed reply above, with no search and no model call (status `out_of_corpus`). Flow 6
  will turn this into an approval request. Built. A ticker counts as outside the corpus when it is a cashtag
  (`$NVDA`) or a capitalised 2–5 letter word that is not a common abbreviation and not a word the transcripts
  themselves use (`corpus/terms.json`: product names such as EPYC or AEC). Limits: lowercase company names
  ("Nvidia") are not recognised, and a real ticker spoken on a call (ARM) is only recognised as a cashtag;
  both go to the agent like any other question.
- Quarter not in the corpus for a known ticker: say which quarters exist and offer the nearest. Built. The
  offered quarter is remembered, so "yes" next resolves to it.
- The guardrail runs before the check: an advice request about an unknown ticker is refused as advice, and
  personal data is masked before the turn is stored.
- Guardrail unavailable: the question is not answered (fixed message). Built.

### 6. Out-of-corpus ticker: ask for approval before fetching

**Status: planned.** Approval is the one action that leaves the fixed corpus: an external API call, content
nobody has checked, and a cost.

> **User** *(viewer)*: What did NVDA say about data center demand last quarter?
> **Assistant:** NVDA isn't in my corpus. I can fetch the transcript of its latest call from an external source,
> for this question only, once a reviewer approves. I've sent the request.
>
> *Reviewer's view:* NVDA, latest call, requested by [user] for "[question]". **[Approve] [Decline]**
>
> *(after approval)* **Assistant:** [Answer from the fetched transcript, naming the call.] *Source: fetched
> transcript, not part of the corpus.*

On decline:

> **Assistant:** A reviewer declined the request, so nothing was fetched. I can answer about AMD, RIVN, ROOT,
> META, CRDO and NBIS.

Expected:
- Nothing is fetched before approval. The agent proposes the fetch; a hook pauses the run (Strands interrupt)
  and the UI shows approve / decline.
- Approve → one fetch through the transcript tool, then the answer. Decline → the tool call is cancelled and
  the reply above follows; the declined request is not proposed again in the same session unless asked.
- The fetched text answers that turn only and is **never added** to the corpus or the Knowledge Base, so
  evaluation stays repeatable.
- The audit record holds the proposal, the decision and who made it (the `approval` field, empty today).
- Who approves: only a user in the `reviewer` group, checked server-side from the login's group claim, never
  from the browser (decision 4). A `viewer` can request a fetch but not approve it, including their own.

Design notes from the probe on Strands 1.63 (8 Nov):
- The run budget resets at the start of each invocation, so resuming after approval would start a fresh
  budget. The resume must keep the original budget.
- The paused agent must survive between the two requests: kept per runtime session, or persisted with a
  session manager. With a separate reviewer, the wait can be minutes, not seconds, which favours persisting
  it and expiring an unanswered request.

### 7. Not in the transcripts: say it wasn't discussed

**Status: built.** Measured on the 4 unanswerable golden-set questions (in-corpus ticker and quarter, topic
not discussed on the call).

> **User:** Did Rivian announce a dividend on its Q2 2026 call?
> **Assistant:** Rivian's Q2 2026 call didn't discuss a dividend.

(Golden-set question NA-01; the absence was checked by keyword search of the whole call.)

Expected:
- `covered=false`, no citations, no guess, no answer from general knowledge.
- At most two searches, then the answer; a third or repeated search is refused in code (16 Nov: no run
  stopped on the budget in two full runs, all 4 declined both times).
- Planned with session memory: offer what was said nearby ("The call discussed [related topic]; do you want
  that?"), only when a returned passage actually covers a related topic.

## Evaluation

| Flow | Single-turn (built) | Multi-turn (`eval/multiturn_set.json`; built except flow 6) |
|---|---|---|
| 1 Happy path | Golden set, 30 answerable: retrieval, grounding, task, citation check | One conversation |
| 2 Missing slot | — | Asks once, answers after the reply |
| 3 Follow-up | — | Pronoun and relative-quarter follow-ups, one switching company |
| 4 Advice | Guardrail input screen (`eval/eval_guardrail.py`) | Refusal mid-conversation, conversation continues |
| 5 Failure / unknown | — | Unknown ticker, quarter outside the corpus |
| 6 Approval | — | One approve, one decline, and a viewer trying to approve their own request |
| 7 Not discussed | Golden set, 4 unanswerable: abstention | One conversation |

Scored per session: **goal success** (the user got the answer, or a correct "not discussed"), **refusal
correctness** (refused advice, didn't refuse near-misses) and **approval correctness** (nothing fetched without
a reviewer's approval, decline handled). Single runs on small sets show behaviour, not rates.

## Decisions

Decided 17 Nov.

1. **No company named (flow 2):** ask one clarifying question only when the question needs one specific
   company and neither the message nor the session supplies it. Cross-company questions search all calls.
   *Updated 18 Nov (Saber): the question is asked in code, not by the model (see flow 2).*
2. **Advice reply (flow 4):** the fixed guardrail message. The question is blocked before the agent runs, so
   the hard rule doesn't depend on the model. No tailored redirect.
3. **Corpus check (flows 5, 6):** in code, against `corpus/calls.json`, before the agent runs. Deterministic
   and testable. Mixed questions ("AMD and NVDA") are handled less gracefully; the check covers the ticker and
   quarter given or resolved from the session.
4. **Approver (flow 6):** only the `reviewer` group. This needs a login with groups (Cognito: `viewer`,
   `reviewer`) checked server-side, and a reviewer view in the UI listing pending requests.
