---
name: setup
description: Set up lesson recall on this machine (backfill past sessions into lesson cards, install the local encoder, switch recall on, optional private backup). Use when the user asks to set up, install, configure or finish installing lesson-recall.
disable-model-invocation: true
---

# Set up lesson recall

Read `${CLAUDE_PLUGIN_ROOT}/SETUP.md` in full, then follow it top to bottom.

- The plugin is installed at `${CLAUDE_PLUGIN_ROOT}`. SETUP.md's commands start with `export CLAUDE_PLUGIN_ROOT=...`; use that path there.
- Run the check after each step before moving on. Stop and tell the user if a check fails.
- Step 2 spends the user's own Claude plan usage: show the plan numbers and get the user's go-ahead (or a `--limit`) before running the backfill.
- Step 7 (backup to a private repository) is optional: ask the user whether they want it.
