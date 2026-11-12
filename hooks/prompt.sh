#!/usr/bin/env bash
# prompt.sh -- UserPromptSubmit hook: MemBrain, read side.
# Ranks the stored lesson cards against the prompt. When the best card clears
# the gate, the top few are injected as background notes and the session model
# decides whether one applies. No model call is made here; with the encoder
# installed it reranks the shortlist locally. In shadow mode (the default until
# setup switches it on) the hook only logs what it would have injected.
# Switch: python3 "${CLAUDE_PLUGIN_ROOT}/recall/cli.py" mode on|shadow|off
# Output: JSON additionalContext, or nothing. Silent, and always exits 0.

[ -n "${RECALL_INNER:-}" ] && exit 0   # inside the plugin's own headless model call
ROOT="${CLAUDE_PLUGIN_ROOT:-$(cd "$(dirname "$0")/.." 2>/dev/null && pwd)}"
[ -f "$ROOT/recall/cli.py" ] || exit 0
PY="${RECALL_HOME:-$HOME/.local/share/claude-recall}/venv/bin/python"   # present once the encoder is installed
[ -x "$PY" ] || PY=$(command -v python3 2>/dev/null) || exit 0
LIMIT=$(command -v timeout || command -v gtimeout)   # macOS ships neither; the hook timeout still applies
${LIMIT:+"$LIMIT" 4} "$PY" "$ROOT/recall/cli.py" hook 2>/dev/null
exit 0
