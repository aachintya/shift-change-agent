# Evaluation report

## Method

There are two layers of evidence.

**1. Automated tests (`tests/`, 68 tests).** These use real Postgres, a
scripted `FakeLLM` and a clock frozen at 17:00 IST. Each one asserts on
database state (bookings, seat counts, request rows), not only on reply text.

**2. Evaluation set (`eval/cases.jsonl`, 43 cases).** Each case is a short
script: employee messages, manager decisions, service restarts, injected tool
faults, and "someone else took the seat". `eval/run.py` runs every case on
fresh sample data and writes `eval/results.jsonl` with
`id, input, expected, actual, pass, notes`.

Judging is **deterministic**. The runner compares structured outcomes: the
tool chosen, the final booking in the database, the request status, the HTTP
status, notifications, required and forbidden phrases in the reply, and the
LLM call count (at most 4). There is no LLM-as-judge, so results can be
rerun for free and never depend on a quota.

There are two modes:

- **Offline (`python -m eval.run`).** The model's answer is scripted per step.
  This tests everything *around* the LLM: rules, isolation, approval and
  restart, failures, and routing. It also runs inside `pytest`, as
  `tests/test_eval.py`, so a failing case fails the test suite.
- **Real (`python -m eval.run --real`).** The scripts are ignored and a real
  model reads the messages. This additionally tests *understanding* (tool
  selection, slot extraction) and measures latency. `fail-01` and `fail-02`
  stay scripted, because they simulate a provider outage or garbage output that
  a real model can't be asked to produce.

## Results (offline mode)

| Category | Cases | Pass | What it covers |
|---|---|---|---|
| `iso` cross-company isolation | 5 | 5 | identity claimed in text, colleague's data, prompt injection, no token, other company's manager |
| `cut` company rules | 6 | 6 | same request → different outcome per company, no override for Globex, full cab, cancelling inside the cut-off |
| `res` resume after restart | 5 | 5 | approve or reject after restart, cab filled while waiting, status while pending, decide twice |
| `sel` tool selection | 12 | 12 | roster / options / confirm / status / small talk / option number / off-topic / AM pickup / cancel for leave / policy per company / greeting by name |
| `amb` ambiguity and follow-ups | 9 | 9 | "a bit late" → options, "make it 8:30 instead", pickup-or-drop question, no cab at that time, plain "no" / "2" read by code, "on leave tomorrow" |
| `fail` failures | 6 | 6 | LLM quota, garbage output, tool timeouts, seat taken before "yes", empty input |
| **Total** | **43** | **43** | |

Tests: `68 passed`. Offline latency, without the LLM: p50 about 14 ms, p95
about 18 ms per message, which shows the code path adds almost nothing.

Concurrency (the bonus) is covered in `tests/test_concurrency.py` rather than
the eval set, because it needs real parallel threads:

- Two employees hit the last seat at the same instant: one `APPLIED`, one `CAB_FULL`, and the seat count ends at capacity.
- One employee double-submits: one `APPLIED`, one `ALREADY_DONE`, exactly 1 request row, and the seat counts move by exactly 1.
- One employee double-submits a request that needs approval: exactly one PENDING row.
- A booking is moved by another request between our read and our write: the result is `CHANGED` with the cab the booking is really on, never a confirmation of the cab that was asked for, and fresh options are offered.
- One employee double-submits a cancellation: one `CANCELLED`, one `ALREADY_DONE`, and exactly one seat freed.

## Results (real LLM)

Model: `qwen/qwen3.8-27b` on Groq's free tier, temperature 0. Command:
`python -m eval.run --real`. Raw output is in `eval/results_real.jsonl`.
This run is on the final code (20 Nov 2026).

| Category | Pass | Notes |
|---|---|---|
| iso | 5/5 | |
| cut | 6/6 | |
| res | 5/5 | |
| sel | 12/12 | |
| amb | 8/9 | `amb-05` (provider timeout, see below) |
| fail | 6/6 | |
| **Total** | **42/43** | |
| latency | p50 356 ms, p95 613 ms, max 3124 ms (n=64) | target: p95 < 5000 ms |

**What failed:** `amb-05` ("8:30 drop" → "no, leave it"). Groq timed out
after 3 s on the second message. This is a provider failure, not a wrong
decision: Arjun got the plain "couldn't process your message, nothing has
changed" reply and his booking was untouched. It is also the 3124 ms maximum.
We kept it as a failure instead of rerunning.

**Model choice.** `qwen/qwen3.8-27b` on Groq: about 560 tokens and 0.1 s per
call. The `gpt-oss` models spend hidden reasoning tokens and hit the free
tier's 8,000 tokens/minute limit quickly. The eval waits 9 s between
messages for the same reason (not counted in latency).

## Known failures and risks

These are the cases we expect a real model to get wrong sometimes. Each one
has a safety net in code:

- **"make it 8 instead" after an *options list*.** The model may choose
  `raise_change_request` with `time=20:00` instead of `check_cab_options`.
  Safety net: if the time is not one of the offered options,
  `raise_change_request` falls back to showing options (`agent.confirm`).
- **Ambiguous AM/PM** ("move my pickup to 8" when the current pickup is 10 AM
  could mean either). The model resolves it to a time. Safety net: the employee
  sees the exact time in the proposal and must say yes.
- **No day stated, with bookings today and tomorrow.** The code picks the next
  upcoming trip in that direction. Correct most of the time, and the proposal
  names the day.
- **A declined escalation** ("no, don't send it"). This relies on the model
  returning `none`. If it misreads that as "yes", a pending request is created.
  It is visible to the employee and the manager can reject it.

## Limitations

- The eval set is hand-written (43 cases). It is not a sample of real
  traffic, and the phrasings are clean English. Hinglish and typos
  ("kal ka drop 8 baje") are not covered yet.
- In offline mode, the `sel` category tests routing, not the model's
  judgement. The real-LLM run is the true test of tool selection.
- The restart test restarts the app inside the same Python process: the pool
  is closed and the graphs are rebuilt. In the demo we restart the real
  container with `docker compose restart api`.

## A trade-off we made, and why

**We used templates for every reply instead of letting the LLM phrase them.**
LLM-written replies would sound more natural. But a confident made-up
confirmation is worse than no answer. A template can only say "Done,
you're on the 8:30 PM cab AC-2030" if that row was just committed. It also
cuts the LLM calls per message from 2 to 1, which is most of our latency
headroom. We accepted a more robotic tone in exchange for replies that are
always correct.
