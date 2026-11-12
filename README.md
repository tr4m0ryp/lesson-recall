# lesson-recall: long-term memory for Claude Code

A Claude Code plugin that learns from your past sessions. When you correct the
agent, it writes the lesson down as a small card. When a later prompt needs that
lesson, the card comes back. When nothing fits, it stays silent.

## The problem

Coding agents forget everything between sessions. You explain that the tests
need a flag, that a helper already exists, that a deploy goes to staging first,
and next week you explain it again.

The usual fix is retrieval: store everything, embed the prompt, fetch the k most
similar memories. In my measurements that part already works. On a synthetic
benchmark the right memory is in the top 10 every time, but it is picked first
only 0.6194 of the time, and a larger k does not change that. The right memory
is retrieved and then out-ranked by its lookalikes, or it was never written
down. **The work is in the write path**, not in the retriever.

## The idea

Decide at write time what a memory is and which memories it competes with.
At read time, stay silent unless a card clearly fits, and let the model that is
already reading the prompt do the picking.

```
session transcript
  -> cut into units          (every stretch is read; corrections only decide the cuts)
  -> one Haiku call per unit (0 to 2 lesson cards, each tied to a verbatim quote)
  -> write decision          (new card against its nearest cards: duplicate,
                              supersedes, coincides, or interferes and gets linked)
  -> append-only event log   (nothing is merged or deleted; every decision can be undone)

your prompt
  -> BM25 shortlist of 3     (trigger and referent count twice, lesson once)
  -> local encoder gate      (cosine against a threshold fitted on your own past prompts)
  -> 0 to 3 cards injected   (as observations that may not apply)
```

**The lesson card.** A referent (what it is about), a lesson (one or two
sentences, hedges kept), a trigger (the kind of request that should recall it),
a scope (this project, or everywhere), and evidence (session id, byte offset, a
verbatim quote). A card whose quote cannot be found in the session is dropped.

**The write side runs in the background.** A hook on Stop, PreCompact and
SessionEnd hands the session to a detached pass that reads only what was added
since its last visit. No prompt ever waits for it.

**The read side makes no model call.** A hook on every prompt ranks the cards
with BM25 over the cards in scope, and a local encoder (bge-small-en-v1.5 through
ONNX) gates the top 3 by cosine. Both thresholds are quantiles of your own
replayed prompts, refitted as the store grows.

**Cards can follow you across devices.** Optionally, each device pushes its own
append-only log to a private GitHub repository of yours and reads the others.
Two devices never write the same file, so a sync can lose a race but never conflict.

### How it works, and why

Each design choice comes from a question about human memory, and each one has
a number behind it. In short: plain retrieval fetches the right memory and then
loses it to lookalikes; the brain spends its effort when a memory is written.
So cards are lessons with a verbatim quote, lookalikes are told apart by a
decision when a card is stored, a contradicted card is retired only after a
second confirming call, and recall stays silent unless an encoder gate fitted
on your own prompts is cleared. [SETUP.md](./SETUP.md#how-it-works-and-why)
walks through it with one formula per idea, including the ideas that failed.

## Install

```
/plugin marketplace add tr4m0ryp/lesson-recall
/plugin install lesson-recall@lesson-recall
/lesson-recall:setup
```

The third command has Claude Code walk you through [SETUP.md](./SETUP.md): plan
the backfill and confirm its Haiku calls with you, fill the store from past
sessions, install the encoder, show you real cards, switch recall on, and
optionally set up the private backup. Needs python3 3.10 or newer and the
`claude` CLI, on Linux or macOS. Until setup switches recall on, the plugin runs
in shadow mode: new sessions already produce cards, nothing is injected.

## How well it works

All numbers below come from the author's own Claude Code sessions, unless they
are marked synthetic.

### The write side

```
13,922 session files scanned
    53 qualify (31 live, 22 from an archive)
   756 units read (282 surprise episodes, 474 calm chunks)
   380 held no lesson
   478 cards proposed, 8 dropped because the quote was not found verbatim (1.7%)
    57 absorbed as duplicates
   413 cards stored
```

Extraction cost 1.6905 USD for the 756 units and the write decisions 0.7992 USD,
as reported by `claude -p`. The write decision judged 2,135 pairs: 1,787
INTERFERENCE (1,036 with a named distinguishing condition), 270 DEFER, 64
DUPLICATE, 12 COINCIDENCE, 2 SUPERSESSION. Both supersessions failed the second
confirming call. Zero cards are retired.

### The read side

On 320 sampled past prompts, labelled by Haiku:

| Gate | Fires | Top card relevant | At least one shown card relevant | Shown cards relevant |
|---|---|---|---|---|
| None | every prompt | 0.371 | | |
| BM25 score | 102 of 320 (31.9%) | 0.510 | 0.627 | 0.488 of 207 |
| BM25 shortlist, encoder cosine (shipped) | 97 of 320 (30.3%) | 0.701 | 0.722 | 0.688 of 160 |

The encoder gate shows 110 relevant cards against 101, with 47 fewer cards in
total. Top-card precision rises by +0.191, paired bootstrap 95% interval
[0.086, 0.300]. Over 906 labelled pairs, the cosine separates relevant from
irrelevant cards with an AUC of 0.793, against 0.676 for the BM25 score.

**Latency.** The read hook took 82 ms at 172 cards, with 31 to 52 ms of scoring
at 413 cards. When the encoder runs it costs 615 ms inside the hook and 0.75 s
wall; when the BM25 pre-gate stops the prompt first, 27.5 ms and 0.09 s. The
pre-gate lets 70% of prompts through. The write hook returns after 77 ms.

### Picking among candidates

The least exciting idea worked best. On the synthetic benchmark, one LLM call
that compares the top 5 candidates picked the right memory 0.9444 of the time
[0.8918, 0.9752], against 0.6194 for picking by similarity. It was measured at
k=5 only, on a 180-query subsample, and costs about $0.014 per query as its own
call. The plugin makes no such call: it hands the cards to the session model,
which is already reading the prompt.

### What I tried that did not work

**Moving memories apart.** The brain gives two similar inputs codes that sit
further apart than the inputs were (pattern separation). I moved stored vectors
apart at write time: 0 of 14 configurations cleared the bar, the best gained
+0.83 points against a 5-point bar (synthetic). A symmetric push with a shared
normaliser cannot reorder a pair of two competitors; the proof holds only under
those premises (symmetric push, shared normaliser, same dimension, two
competitors). The plugin separates lookalikes with a write-time decision
instead. The directional variant, which pushes only the new memory away, is not in the plugin. [More in SETUP.md](./SETUP.md#how-it-works-and-why).

**Surprise as a filter.** Reading only the episodes where the user corrected
the agent would have dropped half the lessons: flagged episodes held a lesson in
149 of 281 (53.0%), calm spans in 32 of 60 (53.3%). Every stretch is read now.

### How these numbers were measured

- **Sessions:** the author's own Claude Code sessions, from the funnel above.
- **Labels:** by Haiku, one call per prompt.
- **Gate choice:** the encoder and its card text were chosen among 18 variants on
  the same 320 prompts; the lowest variant measured 0.608.
- **Encoder:** English (bge-small-en-v1.5).

## What leaves your machine

- **Transcript excerpts go to Haiku** through your own Claude login, after
  credential-shaped text is redacted. These calls use your plan's usage.
- **The encoder is downloaded once** from PyPI and Hugging Face, checked against
  pinned sha256 digests.
- **Cards go to your own private repository**, and only if you turn on sync.
- **Nothing goes to the plugin author.**

The store lives in `~/.local/share/claude-recall` (or `$RECALL_HOME`). Switch
with `python3 <plugin root>/recall/cli.py mode on|shadow|off`; undo steps are
at the end of [SETUP.md](./SETUP.md).

## License

lesson-recall is **source-available**, licensed under the [PolyForm Noncommercial License 1.0.0](./LICENSE), **not** an OSI open-source license.

- **You may** use, modify, fork, and share lesson-recall freely for any **noncommercial** purpose, as long as you keep the copyright and `Required Notice:` lines (see [`NOTICE`](./NOTICE)) and credit *"lesson-recall by Keygraph, Inc."*
- **You may not** sell it, bundle it into a paid product, or run it as a paid/hosted service **without a commercial license**.

Copyright (c) 2026 Keygraph, Inc. Commercial licensing enquiries: see [`NOTICE`](./NOTICE).
