# Setting up lesson recall

This file is written for the agent (Claude Code) that sets the plugin up in the
user's session. Read the first section to explain the plugin to the user if
they ask, then run the steps in order. Each step ends with a check. If a check
fails, stop and tell the user what failed.

## How it works, and why

lesson-recall is the Claude Code plugin that implements **MemBrain**, an
approach to agent memory that decides at write time which memories compete and
stays silent at read time unless one clearly fits. It implements the part of
the MemBrain research that survived its measurements. The full explanation,
with the math and the plots, is in the README section
[How it works, and why](./README.md#how-it-works-and-why). Read it there if the
user wants depth. The short version, for explaining it in a few sentences:

- **The problem.** Standard retrieval (RAG) sums over the k most similar memories.
  On a synthetic benchmark the right memory is in the top 10 every time
  (recall 1.0000 at k=10) but ranked first only 0.6194 of the time, at every k.
  The loss is competition between lookalike memories, so the fix belongs in the
  write path, which is where MemBrain works. This was measured on a synthetic
  corpus, not on real sessions.
- **Lessons, not transcripts.** One Haiku call per unit writes at most 2 cards;
  a card is dropped unless its quote is found verbatim (8 of 478 dropped, 1.7%).
- **A write decision for lookalikes.** A new card meets its up to 5 nearest
  cards; two bits (same referent, same value) give DUPLICATE, SUPERSESSION,
  COINCIDENCE, INTERFERENCE or DEFER. Nothing is merged or deleted.
- **Supersession needs a second confirming call.** On real sessions both
  proposed supersessions failed it: zero cards retired.
- **Recall by cue and scope, silent by default.** BM25 (trigger and referent
  count twice) shortlists 3 in-scope cards; the MemBrain gate (a local encoder) shows a card
  only above a threshold fitted so that a share 0.33 of the user's own past
  prompts would fire. Top card relevant in 0.701 of fires against 0.510 for a
  BM25-only gate (320 real prompts, labels by Haiku, gate
  chosen among 18 variants on them).
- **The session model picks.** One comparative LLM call picked the right memory
  0.9444 of the time against 0.6194 by similarity (synthetic, k=5, 180-query
  subsample). The plugin makes no extra call: the session model, which already reads the prompt, makes the pick.
- **What failed.** Moving stored vectors apart symmetrically cannot reorder a
  pair (proven inside its premises; 0 of 14 configurations cleared the bar),
  surprise did not select lessons (53.0% against 53.3%), and the recollection
  veto lost to the comparative call (0.8833 against 0.9444).

## What leaves the machine

- **Transcript excerpts go to Haiku** through the user's own `claude` login, after
  credential-shaped text is redacted. This uses the user's Claude plan usage.
- **The encoder is downloaded once** (step 5) from PyPI and Hugging Face.
- **Cards go to the user's own private repository**, and only if they enable sync
  (step 8).
- **Nothing goes to the plugin author.**

Everything else stays in the store, `${RECALL_HOME:-$HOME/.local/share/claude-recall}`.

## What is fitted to you, and what is a default

- **Fitted to the user's own data:** the stop words (`tune`, from their past
  prompts); the gate thresholds `gate_score`, `dense_pre` and `dense_gate`
  (`calibrate`, quantiles of their replayed prompts); `link_score` (`calibrate`,
  so that half of the stored "not to be confused with" links are shown). The
  thresholds refit by themselves once the store has grown by 15%; re-run
  `tune` now and then as the user writes more prompts.
- **Set for the user in step 2:** `correction_cues`, `card_language`, and
  whether the encoder runs (`dense`).
- **Defaults chosen by the author, not fitted to anyone:** a third of past
  prompts should fire (`gate_rate` 0.33), the pre-gate lets 70% through,
  stop words at 10% of prompts, BM25 with k1 = 1.2 and b = 0.75, trigger and
  referent counted twice, 3 cards shown, 5 neighbours per write decision, units
  of at most 14,000 characters, prompts under 24 characters skipped.
- **Fallbacks from the author's sessions, used only until `calibrate` runs:**
  `gate_score` 10.5, `dense_pre` 8.0, `dense_gate` 0.68, `link_score` 25.
- **Language-specific parts:** the encoder (English; for other languages setup
  uses the keyword gate), the suffix trimming in the keyword index (English),
  and the prompt floor, counted in characters (24).

All of these live in `config.json` in the store; `cli.py config show` lists them.

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
- **git** is needed only for sync (step 8). Missing: skip step 8.
- **claude** must resolve, because cards are written by `claude -p --model haiku`.
- **gh** (GitHub CLI, logged in) is optional and only used by step 8.
- Runs on Linux and macOS.

## Step 2: adapt it to this user (no model call)

```bash
python3 "$CLAUDE_PLUGIN_ROOT/recall/cli.py" tune
```

This reads the user's past typed prompts, the same ones the gate is later fitted
on, and calls no model. It learns the user's **stop words**: every term found in
at least 10% of their prompts (config `stop_df`), at most 150. Such a term is in
almost every prompt, so it cannot tell cards apart, whatever its language. With
fewer than 100 prompts the built-in list stays, which is English plus Dutch only.
It also prints a **language estimate**: the share of prompts in which at least
15% of the terms are common English function words. That is a heuristic, not a
classifier. Re-running it is safe: the same prompts give the same result.

Show the user `top_stopwords`, `non_ascii_token_share`, `english_prompt_share`
and `language_estimate`. You can read their language from the stop words
yourself; trust that over the heuristic. Then make three decisions:

1. **Correction cues.** These are the words that mark a user turn as a
   correction (defaults: English and Dutch). For each other language the user
   writes in, add three to five words they would type to stop or correct an agent:

   ```bash
   python3 "$CLAUDE_PLUGIN_ROOT/recall/cli.py" config add correction_cues "falsch"
   ```

   Cues are regular expressions, matched case-insensitively as whole words in
   the first 400 characters of a user turn; in Chinese, Japanese or Thai they
   match anywhere. Do not spend long on this. Cues only decide where a session
   is cut into units, and every stretch is read either way: on the author's
   sessions flagged and calm stretches held a lesson at the same rate (53.0%
   and 53.3%).
2. **Card language.** Cards are written in English by default. If most prompts
   are in another language, write the cards in that language, so their words
   match the prompts that should recall them:

   ```bash
   python3 "$CLAUDE_PLUGIN_ROOT/recall/cli.py" config set card_language "German"
   ```
3. **Gate.** The encoder in step 5 is English only. If `language_estimate` is
   `mostly not English`, skip it: in step 5 run `dense off` and `calibrate`
   instead of `dense install`, and tell the user why. Recall then gates on the
   keyword score alone (on English prompts: top card relevant in 0.510 of fires,
   against 0.701 with the encoder).

**Check:** `tune` printed `"stopwords": "learned"` (or says why it kept the
built-in list), and `python3 "$CLAUDE_PLUGIN_ROOT/recall/cli.py" config show`
shows the cues and card language you set.

## Step 3: plan the backfill (no model call)

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
skip to step 5; cards will come from new sessions.

## Step 4: run the backfill in the background

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

## Step 5: install the encoder and fit the gate

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

If step 2 found the prompts mostly not English, run this instead and skip the check below:

```bash
python3 "$CLAUDE_PLUGIN_ROOT/recall/cli.py" dense off
python3 "$CLAUDE_PLUGIN_ROOT/recall/cli.py" calibrate
```

**Check:** the last line is JSON with `"dense": true` and `"model_ready": true`.
If the install fails, run `python3 "$CLAUDE_PLUGIN_ROOT/recall/cli.py" dense off`
then `python3 "$CLAUDE_PLUGIN_ROOT/recall/cli.py" calibrate`; recall then gates on
BM25 alone, which is less precise.

## Step 6: show the user what was learned

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

## Step 7: confirm recall is on

```bash
python3 "$CLAUDE_PLUGIN_ROOT/recall/cli.py" mode on
```

**Check:** the printed config has `"mode": "on"`. From the next prompt on,
matching cards arrive as a `<recalled-lessons>` block.

## Step 8: optional private backup and sync

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

## Step 9: verify, and how to undo

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
