# Setting up lesson recall

This file is written for the agent (Claude Code) that sets the plugin up in the
user's session. Read the first section to explain the plugin to the user if
they ask, then run the steps in order. Each step ends with a check. If a check
fails, stop and tell the user what failed.

## How it works, and why

The brain is the design motivation here, not a claim that this plugin
implements the brain. Where a block borrows only a name, it says so.
Synthetic means a generated benchmark corpus; real means the author's own Claude Code sessions.

**Standard retrieval (RAG).** Score every stored memory $`d_i`$ against the
question $`q`$ with an encoder $`e`$, keep the $`k`$ best, and paste them in:

```math
s_i = \cos\big(e(q), e(d_i)\big), \qquad C(q) = \operatorname{top-}k_i\; s_i, \qquad \text{answer} = \mathrm{LLM}\big(q + C(q)\big)
```

In words: store everything, fetch the k most similar, and hope the right one wins.

**The problem we measured.** Two different probabilities hide in that line:
P(right memory is in the top k) and P(right memory wins among its lookalikes).
On a synthetic corpus (120 episodes, 360 queries, 33 clusters of lookalikes,
plain vector search with MiniLM), the first is 0.958 at k=5 and reaches 1.0000
at k=10. The second is 0.619 and does not move with k (0.6194 plain, 0.6361
BM25, at every k). A larger k cannot fix this. The loss is competition between
similar memories, so the fix belongs in the write path. This was measured on a
synthetic corpus, not on real sessions.

So we looked at how human memory handles this. One block per idea.

**1. Store lessons, not transcripts.** The brain keeps what an episode taught,
not a recording of it. A background pass cuts each session into units and one
Haiku call per unit writes at most 2 cards, each a referent (what it is about),
a lesson (hedges kept) and a trigger (the request that should recall it). The
correction to the gist idea: summaries invent (5.02% invented-belief rate in
the research), so every card carries a quote from the session and is dropped
unless that quote is found verbatim:

```math
\text{keep}(c) = \mathbf{1}\big[\,\text{quote}(c) \subseteq \text{unit}(c)\,\big]
```

On real sessions 8 of 478 proposed cards failed this check (1.7%).

**An idea that did not hold: surprise as a filter.** The plan was to read only
surprising stretches (a user correction, an interrupt, a burst of tool errors).
On real sessions, flagged episodes held a lesson in 149 of 281 (53.0%) and calm
spans in 32 of 60 (53.3%). Surprise now decides where to cut; every stretch is read.

**2. Separate lookalikes at write time.** The dentate gyrus gives two similar
inputs codes that sit further apart than the inputs did (pattern separation).
The plugin separates with a decision instead of geometry. A new card meets its
up to 5 nearest active cards (BM25, at least 3 shared terms), and Haiku answers
two bits per pair: same referent $`\rho`$ and same value $`\nu`$:

```math
\text{class}(\rho,\nu) = \begin{cases} \text{DUPLICATE} & (1,1) \\ \text{SUPERSESSION} & (1,0) \\ \text{COINCIDENCE} & (0,1) \\ \text{INTERFERENCE} & (0,0) \\ \text{DEFER} & \text{either bit unsure} \end{cases}
```

INTERFERENCE keeps both cards and links them when the judge names the
condition that tells them apart; recall then shows "not to be confused with".
On 2,135 real judged pairs: 1,787 INTERFERENCE (1,036 with a named condition),
270 DEFER, 64 DUPLICATE, 12 COINCIDENCE, 2 SUPERSESSION. The classes are the Haiku judge's labels.

**2b. Moving the vectors apart: tried, not shipped.** The literal version of
pattern separation moves stored vectors apart at write time. We tried a
symmetric transform on frozen MiniLM vectors (synthetic corpus): 0 of 14
configurations cleared the bar, the best gained +0.83 points against a 5-point
bar. Theorem 1 says why. For two competitors (m=2), a symmetric write-time
transform $`T`$ with a normaliser shared across the pair cannot change which
one wins:

```math
\arg\max_{i \in \{1,2\}} \cos\big(e(q), T(d_i)\big) = \arg\max_{i \in \{1,2\}} \cos\big(e(q), d_i\big)
```

Disambiguation stayed at 0.8056 at every strength. The result is closed only
inside its premises: symmetric push, shared normaliser, same dimension, m=2.
The brain's version is directional: it moves the NEW memory away from its
stored neighbours. That asymmetric variant sits outside the theorem and was
never properly tested (one attempt fell from 0.806 to 0.694 at m=2). It is not
in this plugin.

**3. Update instead of piling up.** In reconsolidation a recalled memory that
meets a mismatch is updated, not duplicated. A SUPERSESSION retires the older
card only after a second Haiku call confirms that the newer one makes it
obsolete; the old card is marked superseded, never deleted, and `restore`
undoes it. A DUPLICATE raises the old card's count and keeps the new wording.
The count is shown ("seen 3x") but does not change ranking. On real sessions
both proposed supersessions failed the confirming call: zero cards retired.

**4. Recall by cue and by place.** A memory comes back when the cue matches how
it was stored (encoding specificity) and more easily in the same context. BM25
scores the trigger and referent twice and the lesson once; the encoder embeds
the whole card (`referent. trigger lesson`). Project cards fire inside their
project on any device (and in a parent folder at least two levels below home),
general cards everywhere:

```math
\text{pool}(q) = \{\, c : \text{scope}(c) = \text{general} \;\lor\; \text{project}(c) \subseteq \text{cwd}(q) \,\}
```

**5. Stay silent by default.** Most prompts should recall nothing. BM25 takes
the top 3 in-scope cards; the encoder runs only when the best BM25 score clears
$`\theta_{\text{pre}}`$ with 2 or more matched terms; card $`i`$ is shown only if

```math
\cos\big(e(q), e(c_i)\big) \ge \theta, \qquad \theta = \text{the quantile of replayed past prompts at which a share } \rho = 0.33 \text{ fire}
```

Each prompt's own session is held out of the replay, $`\theta_{\text{pre}}`$
lets 70% of prompts through, and both refit when the pool has grown by 15%.
On 320 real past prompts labelled by Haiku, the top card is relevant in 0.701
of fires (30.3% fire) against 0.510 for a BM25-only gate (31.9% fire), and
0.371 with no gate. The encoder and card text were chosen among 18 variants on the same 320 prompts (the lowest scored 0.608); labels come from Haiku; the encoder adds about 0.75 s on the prompts that pass the pre-gate.

**6. Let a comparison choose.** The least exciting idea is the one that worked
best: one LLM call that compares the candidates (llm_pick):

```math
\text{pick} = \mathrm{LLM}\big(q, \text{top-}k\big) \qquad \text{versus} \qquad \text{pick} = \arg\max_i s_i
```

On the synthetic corpus it reached 0.9444 [0.8918, 0.9752] against 0.6194,
closing 86.1% of the headroom. It was measured at k=5 only, on a 180-query
subsample, and costs about $0.014 per query as a dedicated call, which is why
it stayed a baseline. The brain-inspired rival, recollection veto
(recall-to-reject), lost: 0.8833 against 0.9444, paired difference -6.11
[-9.89, -3.08] (lost a comparison, not proven closed). The plugin makes no
extra call: the cards go to the session model, which is already reading the
prompt, marked as observations that may not apply. The 0.9444 was measured with a dedicated call at k=5.

**Side by side.** RAG takes the argmax over everything stored, at read time.
This plugin decides at write time what a memory is and which memories it
competes with; at read time similarity finds the candidates, a comparison
chooses among them, and silence is allowed.

## What leaves the machine

- **Transcript excerpts go to Haiku** through the user's own `claude` login, after
  credential-shaped text is redacted. This uses the user's Claude plan usage.
- **The encoder is downloaded once** (step 4) from PyPI and Hugging Face.
- **Cards go to the user's own private repository**, and only if they enable sync
  (step 7).
- **Nothing goes to the plugin author.**

Everything else stays in the store, `${RECALL_HOME:-$HOME/.local/share/claude-recall}`.

## Before you start

Your shell tool does not keep variables between calls. Start every command with
the plugin root, which the setup skill gave you:

```bash
export CLAUDE_PLUGIN_ROOT="<plugin root>"
```

Without the skill, find it with
`ls -d "${CLAUDE_CONFIG_DIR:-$HOME/.claude}"/plugins/cache/lesson-recall/lesson-recall/*/ | tail -1`.
Below, `RECALL` stands for `python3 "$CLAUDE_PLUGIN_ROOT/recall/cli.py"`; type it out in full.

The plugin is **on from the moment it is installed**: the background pass writes
cards from new turns of live sessions, through Haiku calls on the user's own
Claude login, and matching cards are injected. Tell the user this first. Until
the steps below have run, the store only knows sessions held since the install,
and the gate runs on keyword scores alone.

## Step 1: prerequisites

```bash
python3 --version; git --version; command -v claude || ls ~/.local/bin/claude; command -v gh && gh auth status
```

**Check:**
- **python3 3.10 or newer** is required (tested on 3.12). Missing or older: tell the
  user the plugin stays inert until they install it (macOS: `brew install python`;
  Debian/Ubuntu: `sudo apt install python3 python3-venv`). Stop here.
- **git** is needed only for sync (step 7). Missing: skip step 7.
- **claude** must resolve, because cards are written by `claude -p --model haiku`.
- **gh** (GitHub CLI, logged in) is optional and only used by step 7.
- Runs on Linux and macOS.

## Step 2: plan the backfill (no model call)

```bash
python3 "$CLAUDE_PLUGIN_ROOT/recall/cli.py" backfill --dry-run
```

This counts, without calling any model, the sessions under
`${CLAUDE_CONFIG_DIR:-~/.claude}/projects`, how many qualify (2 or more typed
user turns, an opening prompt not shared by 5 or more sessions, not started in
print mode or through the Agent SDK), how many units that is, and the Haiku calls.

Show the user `sessions_qualify`, `units_to_process`, `haiku_calls_extract` and
`haiku_calls_judge_at_most`, and say plainly: **these calls run through your own
Claude login and count against your plan's usage.** For scale, on the author's
sessions 756 units cost 1.6905 USD to extract and 0.7992 USD to judge, as
reported by `claude -p`.

Ask for a go-ahead. If the user wants fewer calls, plan the N newest sessions
and show the new numbers:

```bash
python3 "$CLAUDE_PLUGIN_ROOT/recall/cli.py" backfill --dry-run --limit 20
```

**Check:** the user said yes to a specific command. If `sessions_qualify` is 0,
skip to step 4; cards will come from new sessions.

## Step 3: run the backfill in the background

```bash
mkdir -p "${RECALL_HOME:-$HOME/.local/share/claude-recall}/log"
nohup python3 "$CLAUDE_PLUGIN_ROOT/recall/cli.py" backfill --limit 20 \
  > "${RECALL_HOME:-$HOME/.local/share/claude-recall}/log/backfill.out" 2>&1 &
```

Drop `--limit 20` if the user approved all sessions. Use your shell tool's
background mode if it has one. For scale, the author's 756 units ran in two
batches: 7.3 and 9.2 minutes of extraction, 17 and 30.8 minutes of serial write decisions.

**Watch:** `tail -n 4 "${RECALL_HOME:-$HOME/.local/share/claude-recall}/log/backfill.out"`.
It prints the sessions line, the units line, an `extracted:` line, and finishes
with a `written:` line.

**Resume:** run the same command again. Finished units are skipped and cached
extractions are reused, so no unit is extracted twice; only the judging of a
unit cut off mid-decision repeats.

**Check:** the last line starts with `written:`.

## Step 4: install the encoder and fit the gate

```bash
python3 "$CLAUDE_PLUGIN_ROOT/recall/cli.py" dense install
```

This creates a private venv inside the store and installs `onnxruntime==1.31.1`,
`tokenizers==0.23.3` and `numpy==2.5.4` from PyPI (version floors when no pinned
wheel fits the Python). It downloads `model_optimized.onnx` and `tokenizer.json`
of bge-small-en-v1.5 (a 66 MB local encoder) from
`huggingface.co/Qdrant/bge-small-en-v1.5-onnx-Q`, pinned to one revision, and
refuses any file whose sha256 differs from the pinned value. It then fits both
gate thresholds on the user's past prompts, which takes minutes of CPU.

**Check:** the last line is JSON with `"dense": true` and `"model_ready": true`.
If the install fails, run `python3 "$CLAUDE_PLUGIN_ROOT/recall/cli.py" dense off`
then `python3 "$CLAUDE_PLUGIN_ROOT/recall/cli.py" calibrate`; recall then gates on
BM25 alone, which is less precise.

## Step 5: show the user what was learned

```bash
python3 "$CLAUDE_PLUGIN_ROOT/recall/cli.py" stats
python3 "$CLAUDE_PLUGIN_ROOT/recall/cli.py" cards --limit 8
```

Then pick 2 or 3 requests the user really makes, using the cards' "applies
when" lines, and show what each would recall from the project it belongs to:

```bash
python3 "$CLAUDE_PLUGIN_ROOT/recall/cli.py" query "<a realistic request>" --cwd "<project directory>"
```

`fires: True` means the card would be injected; `fires: False` means the prompt
stays silent. Both are normal.

**Check:** `stats` shows `cards` above 0, and the user has seen real cards.

## Step 6: confirm recall is on

```bash
python3 "$CLAUDE_PLUGIN_ROOT/recall/cli.py" mode on
```

**Check:** the printed config has `"mode": "on"`. From the next prompt on,
matching cards arrive as a `<recalled-lessons>` block.

## Step 7: optional private backup and sync

Ask the user whether they want it. Cards quote session text, so the repository
**must be private**. With the GitHub CLI logged in:

```bash
python3 "$CLAUDE_PLUGIN_ROOT/recall/cli.py" sync init
```

This reuses `<gh user>/claude-recall-cards`, or creates it with
`gh repo create --private`, checks that it is private, refuses otherwise, and
pushes branch `main`. Each device writes only its own log under `events/`;
`cards/` holds a readable markdown view. For another private repository, add
`--remote URL`; when gh cannot see it, add `--assume-private` only after the
user confirms it is private.

**Check:** the JSON ends with `"state": "pushed"` (or `"clean"` with no cards
yet). Show the user `python3 "$CLAUDE_PLUGIN_ROOT/recall/cli.py" sync status`
and confirm the repository URL with them. Later syncs run in the background.

## Step 8: verify, and how to undo

**Verification checklist:**
- `stats` shows `"mode": "on"`, the gate (`dense` or `bm25`), and cards.
- `tail -n 3 "${RECALL_HOME:-$HOME/.local/share/claude-recall}/log/read.jsonl"`
  shows one line per prompt after the next prompt.
- With sync: `sync status` shows `"ready": true` and no `"error"` in `last`.

**To undo:**
- **Stop injecting, keep learning:** `python3 "$CLAUDE_PLUGIN_ROOT/recall/cli.py" mode shadow`
- **Stop everything:** `python3 "$CLAUDE_PLUGIN_ROOT/recall/cli.py" mode off`
- **Uninstall:** `/plugin uninstall lesson-recall@lesson-recall`
- **Delete the store** (cards, logs, encoder): `rm -rf "${RECALL_HOME:-$HOME/.local/share/claude-recall}"`
- **Delete the backup:** the user deletes `<gh user>/claude-recall-cards` on GitHub themselves.

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `sync init` says gh is not installed or not logged in | No GitHub CLI login | `gh auth login`, or `--remote URL`; recall works without sync |
| sync reports credential-shaped text | A card quotes something token-like | `sync scrub`, then `sync now` |
| sync says the branch holds a different log for this device | Two machines share a hostname, or a reinstall | `sync init --device NAME`, or `sync init --adopt` after a reinstall |
| `read.jsonl` lines carry `dense_error` | Encoder missing or broken | `dense install` again; BM25 decides meanwhile |
| Nothing is ever recalled | Mode is not on, or few cards | `mode on`; check `stats`, then `calibrate` |
