---
name: setup
description: Set up lesson recall on this machine (backfill past sessions into lesson cards, install the local encoder, switch recall on, optional private backup). Use when the user asks to set up, install, configure or finish installing lesson-recall.
disable-model-invocation: true
---

# Set up lesson recall

Read `${CLAUDE_PLUGIN_ROOT}/SETUP.md` in full, then follow it top to bottom.

## Tell the user this before you start

Explain in your own words, in five or six lines, what is about to happen.
Open with one sentence on what they installed: lesson-recall is the Claude Code
plugin that implements MemBrain, a memory that decides at write time which
memories compete and stays silent unless one clearly fits.

1. **A workflow analyses all of their past Claude Code sessions.** It reads every
   transcript under their Claude config folder. Nothing is sent anywhere at this stage.
2. **It filters out the sessions worth learning from.** A session qualifies when
   a person actually worked in it: at least two typed user turns, not a scripted
   or headless run, not one of many sessions that open with the same template.
3. **It converts the qualifying sessions into lesson cards.** Each stretch of a
   session goes to Haiku once, which writes 0 to 2 cards: what the lesson is
   about, the lesson, and the kind of request that should bring it back. A card
   is kept only if its supporting quote is found word for word in the session.
4. **It builds the retrieval architecture on top of the cards.** Every new card
   is compared with its nearest existing cards and filed as new, a repeat, a
   replacement or a related lookalike. Then a keyword index shortlists cards per
   prompt, and a small local encoder decides whether any is close enough to show.
   The thresholds are fitted on the user's own past prompts.
5. **It adapts to the user's language.** Stop words are learned from their own
   past prompts, correction cues and the card language are set for the languages
   they write in, and the English-only encoder is skipped when most prompts are
   not English.

Then state the two facts the user must know: the Haiku calls in step 3 run
through their own Claude login and count against their plan's usage, and the
plugin has been on since it was installed, so new sessions are already being
read and matching cards are already injected.

- The plugin is installed at `${CLAUDE_PLUGIN_ROOT}`. SETUP.md's commands start with `export CLAUDE_PLUGIN_ROOT=...`; use that path there.
- Run the check after each step before moving on. Stop and tell the user if a check fails.
- Step 2 (adapt to the user) makes no model call; do it before the backfill, because the cues and card language shape the cards.
- Step 3 spends the user's own Claude plan usage: show the plan numbers and get the user's go-ahead (or a `--limit`) before running the backfill.
- Step 8 (backup to a private repository) is optional: ask the user whether they want it.
