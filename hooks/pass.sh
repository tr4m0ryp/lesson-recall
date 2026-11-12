#!/usr/bin/env bash
# pass.sh -- Stop / PreCompact / SessionEnd hook: MemBrain, write side.
# Hands the session to a background pass that reads only what was added since
# its last visit, and turns each closed episode into lesson cards (one Haiku
# call per unit, through the user's own claude login). The pass detaches
# itself, so this hook returns after one Python start. Stop passes are spaced
# per session; PreCompact and SessionEnd always run, and SessionEnd also closes
# episodes that are still open. Mode off stops all of it.
# Silent, and always exits 0.

[ -n "${RECALL_INNER:-}" ] && exit 0   # inside the plugin's own headless model call
ROOT="${CLAUDE_PLUGIN_ROOT:-$(cd "$(dirname "$0")/.." 2>/dev/null && pwd)}"
[ -f "$ROOT/recall/cli.py" ] || exit 0
PY="${RECALL_HOME:-$HOME/.local/share/claude-recall}/venv/bin/python"   # present once the encoder is installed
[ -x "$PY" ] || PY=$(command -v python3 2>/dev/null) || exit 0
LIMIT=$(command -v timeout || command -v gtimeout)   # macOS ships neither; the hook timeout still applies
${LIMIT:+"$LIMIT" 5} "$PY" "$ROOT/recall/cli.py" pass >/dev/null 2>&1
exit 0
