"""One headless Haiku call, stripped of tools, hooks, MCP servers and session persistence.

The call runs from a neutral directory with user settings skipped, and with RECALL_INNER set,
so none of the user's hooks, this plugin's included, fire inside it. It goes through the
user's own `claude` login and counts against that plan's usage.
"""
import json
import os
import re
import shutil
import subprocess
import time

from . import paths

GUARDS = {"RECALL_INNER": "1"}   # set inside our own calls; the hooks exit at once when they see it


PRINT_FLAGS = {"-p", "--print", "--output-format", "--input-format"}


class LLMError(RuntimeError):
    pass


class Refused(LLMError):
    """The model declined the content. Retrying cannot help, so callers skip the item."""


def automated():
    """True inside our own headless calls and inside any print-mode claude run.

    Scripted callers (commit-message writers, CI jobs, SDK agents) must never get cards injected
    or have their transcripts mined. The check reads the nearest claude ancestor's arguments.
    """
    if any(os.environ.get(k) for k in GUARDS):
        return True
    pid = os.getppid()
    for _ in range(8):
        try:
            argv, parent = _process(pid)
        except (OSError, ValueError, IndexError, subprocess.SubprocessError):
            return False
        for i, word in enumerate(argv[:2]):   # `claude ...` or `node /path/claude ...`
            if os.path.basename(word) == "claude":
                return bool(PRINT_FLAGS & set(argv[i + 1:]))
        if parent <= 1:
            return False
        pid = parent
    return False


def _process(pid):
    """(argv, parent pid) of a process: from /proc where it exists, else from ps (macOS)."""
    if os.path.isdir("/proc/self"):
        with open(f"/proc/{pid}/cmdline", "rb") as f:
            argv = f.read().decode(errors="replace").split("\0")
        with open(f"/proc/{pid}/stat") as f:
            return argv, int(f.read().rsplit(")", 1)[1].split()[1])
    out = subprocess.run(["ps", "-o", "ppid=,args=", "-p", str(pid)], capture_output=True, text=True, timeout=2)
    parent, _, args = out.stdout.strip().partition(" ")
    return args.split(), int(parent)


def claude_bin():
    """The claude binary: on PATH, else the native install, which a hook's PATH may not include."""
    return shutil.which("claude") or os.path.expanduser("~/.local/bin/claude")


def _json_block(text):
    """Return the first JSON object found in a model reply."""
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    start = text.find("{")
    if start < 0:
        raise LLMError("no JSON object in reply")
    obj, _ = json.JSONDecoder().raw_decode(text[start:])
    return obj


def ask(system, prompt, timeout=120, tries=3):
    """Send one prompt, return (parsed JSON reply, cost in USD)."""
    paths.ensure()
    cmd = [claude_bin(), "-p", "--model", "haiku", "--max-turns", "1", "--tools", "",
           "--no-session-persistence", "--strict-mcp-config", "--mcp-config", '{"mcpServers":{}}',
           "--disable-slash-commands", "--output-format", "json",
           "--setting-sources", "project", "--system-prompt", system]
    env = dict(os.environ, **GUARDS)
    last = "no attempt"
    for attempt in range(tries):
        try:
            out = subprocess.run(cmd, input=prompt, capture_output=True, text=True,
                                 timeout=timeout, cwd=paths.RUN, env=env)
            reply = json.loads(out.stdout)
            if reply.get("is_error"):
                if "/legal/aup" in str(reply.get("result")):
                    raise Refused(str(reply.get("result"))[:120])
                raise LLMError(str(reply.get("result"))[:200])
            return _json_block(reply["result"]), float(reply.get("total_cost_usd") or 0.0)
        except Refused:
            raise
        except (subprocess.TimeoutExpired, ValueError, KeyError, LLMError) as exc:
            last = f"{type(exc).__name__}: {exc}"[:300]
            time.sleep(4 * (attempt + 1))
    raise LLMError(last)
