"""Read a session JSONL into a flat list of turns, each tagged with its byte offset."""
import json
import re

from ..core.redact import redact

SKIP = ("<command-", "<local-command", "Caveat:", "<system-reminder>", "<task-notification",
        "This session is being continued")
INTERRUPT = "[Request interrupted"
_ENTRY = re.compile(r'"entrypoint":"([^"]*)"')
_REMINDER = re.compile(r"<system-reminder>.*?</system-reminder>", re.S)


def _result_text(block):
    c = block.get("content")
    if isinstance(c, list):
        return "\n".join(b.get("text", "") for b in c if isinstance(b, dict) and b.get("type") == "text")
    return c if isinstance(c, str) else json.dumps(c, ensure_ascii=False)


def _tool_line(name, inp):
    if not isinstance(inp, dict):
        return f"{name}: {str(inp)[:160]}"
    for key in ("command", "file_path", "notebook_path", "pattern", "description", "query", "url"):
        if inp.get(key):
            return f"{name}: {str(inp[key])[:240]}"
    return f"{name}: {json.dumps(inp, ensure_ascii=False)[:160]}"


def _turns_of(entry, off):
    msg = entry.get("message")
    if not isinstance(msg, dict) or entry.get("isMeta") or entry.get("isSidechain"):
        return
    role, content, ts = msg.get("role"), msg.get("content"), entry.get("timestamp") or ""
    blocks = [{"type": "text", "text": content}] if isinstance(content, str) else (content or [])
    for b in blocks:
        if not isinstance(b, dict):
            continue
        kind = b.get("type")
        if kind == "text":
            text = (b.get("text") or "").strip()
            if role == "user":
                if text.startswith(INTERRUPT):
                    yield {"off": off, "kind": "interrupt", "text": "[user interrupted]", "ts": ts}
                    continue
                text = _REMINDER.sub("", text).strip()
                if not text or text.startswith(SKIP):
                    continue
                yield {"off": off, "kind": "user", "text": redact(text), "ts": ts}
            elif text:
                yield {"off": off, "kind": "assistant", "text": redact(text), "ts": ts}
        elif kind == "tool_use":
            yield {"off": off, "kind": "tool", "ts": ts,
                   "text": redact(_tool_line(b.get("name", "?"), b.get("input")))}
        elif kind == "tool_result":
            yield {"off": off, "kind": "result", "ts": ts, "err": bool(b.get("is_error")),
                   "text": redact((_result_text(b) or "")[:1200])}


def parse(path, start=0):
    """Return the turns found from byte offset `start`. A half-written last line is left out."""
    turns = []
    with open(path, "rb") as fh:
        fh.seek(start)
        while True:
            off = fh.tell()
            raw = fh.readline()
            if not raw or not raw.endswith(b"\n"):
                break
            if b'"type":"user"' not in raw and b'"type":"assistant"' not in raw:
                continue
            try:
                entry = json.loads(raw)
            except ValueError:
                continue
            if entry.get("type") in ("user", "assistant"):
                turns.extend(_turns_of(entry, off))
    return turns


def headless(path, lines=40):
    """True when the transcript was written by print mode or the Agent SDK (`claude -p`, SDK agents).

    Claude Code stamps every entry with the entrypoint that started the session: `cli` for an
    interactive terminal, `sdk-cli`, `sdk-ts` or `sdk-py` for scripted runs. Scripted sessions
    are neither mined for cards nor given any.
    """
    try:
        with open(path, "r", errors="replace") as fh:
            for _, line in zip(range(lines), fh):
                hit = _ENTRY.search(line)
                if hit:
                    return hit.group(1).startswith("sdk")
    except OSError:
        pass
    return False


def user_turns(path, limit=None):
    """Cheap count of typed user turns, the first prompt, and whether the session is headless."""
    n, first, scripted = 0, "", None
    with open(path, "r", errors="replace") as fh:
        for line in fh:
            if scripted is None:
                hit = _ENTRY.search(line)
                if hit:
                    scripted = hit.group(1).startswith("sdk")
            if '"type":"user"' not in line or '"tool_result"' in line:
                continue
            try:
                d = json.loads(line)
            except ValueError:
                continue
            if d.get("type") != "user" or d.get("isMeta") or d.get("isSidechain"):
                continue
            c = (d.get("message") or {}).get("content")
            if isinstance(c, list):
                c = " ".join(b.get("text", "") for b in c if isinstance(b, dict) and b.get("type") == "text")
            if not isinstance(c, str):
                continue
            c = c.strip()
            if not c or c.startswith(SKIP) or c.startswith(INTERRUPT):
                continue
            n += 1
            first = first or re.sub(r"\s+", " ", c)[:60]
            if limit and n >= limit:
                break
    return n, first, bool(scripted)
