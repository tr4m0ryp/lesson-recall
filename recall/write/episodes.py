"""Cut a turn list into episodes and render each as a bounded window.

Surprise episodes come first. The stretches between them are tiled with calm chunks, because
a control run found a lesson in calm spans as often as in flagged ones (32 of 60 against 149
of 281). The detector therefore decides where to cut, not what to read.

A surprise event is a user correction, an interrupt, or a burst of tool errors. The regex is
only a candidate generator: the extractor may still answer that an episode holds no lesson.
An episode runs from the request before its first event to the first calm user turn after
its last event. Until that calm turn exists the episode is open and no card is written.
Two events share an episode only when the second arrives before the first has closed, so a
parse that starts at any calm user turn cuts the same episodes as a parse of the whole file.
"""
import hashlib
import re

CORR = re.compile(
    r"\b(no[,. ]|not what|wrong|stop|don'?t|instead|again|still (broken|not|fail)|revert|undo|"
    r"why did you|you (forgot|missed|broke)|nee[,. ]|niet|fout|verkeerd|opnieuw|terug)\b", re.I)
CLIP = {"user": 1500, "assistant": 700, "tool": 200, "result": 300, "interrupt": 40}
BURST, SPAN = 3, 6   # BURST failed tool results within the last SPAN results
CALM_STEPS, CALM_TURNS = 3, 120   # a calm chunk closes after this many requests or turns


def events(turns):
    """Indices of surprise turns, with their kind."""
    found, recent, seen_user, quiet_until = [], [], False, -1
    for i, t in enumerate(turns):
        if t["kind"] == "user":
            if seen_user and CORR.search(t["text"][:400]):
                found.append((i, "correction"))
            seen_user = True
        elif t["kind"] == "interrupt":
            found.append((i, "interrupt"))
        elif t["kind"] == "result":
            recent = (recent + [bool(t.get("err"))])[-SPAN:]
            if sum(recent) >= BURST and i > quiet_until:
                found.append((i, "errors"))
                recent, quiet_until = [], i + SPAN
    return found


def _open_index(turns, i):
    """The request that led to event i: the last typed user turn strictly before it."""
    for j in range(i - 1, -1, -1):
        if turns[j]["kind"] == "user":
            return j
    return 0


def _close_index(turns, i, event_ix):
    """First calm user turn after event i: not an event, not the reply to an interrupt."""
    after_interrupt = turns[i]["kind"] == "interrupt"
    for j in range(i + 1, len(turns)):
        kind = turns[j]["kind"]
        if kind == "interrupt":
            after_interrupt = True
        elif kind == "user":
            if j not in event_ix and not after_interrupt:
                return j
            after_interrupt = False
    return None


def cut(turns, sid, final=False):
    """Return episodes: dicts with id, lo, hi, events, closed, ts."""
    evs = events(turns)
    event_ix = {i for i, _ in evs}
    out = []
    for i, kind in evs:
        lo, close = _open_index(turns, i), _close_index(turns, i, event_ix)
        hi = close if close is not None else len(turns) - 1
        if out and lo < out[-1]["hi"]:
            ep = out[-1]
            ep["hi"], ep["closed"] = max(ep["hi"], hi), close is not None
            ep["events"].append((i, kind))
            continue
        out.append({"lo": lo, "hi": hi, "closed": close is not None, "events": [(i, kind)]})
    for ep in out:
        first = turns[ep["events"][0][0]]
        ep["id"] = hashlib.sha1(f"{sid}:{first['off']}".encode()).hexdigest()[:16]
        ep["ts"], ep["off"] = first["ts"], first["off"]
        ep["closed"] = ep["closed"] or final
    return out


def _chunk(turns, sid, lo, hi, closed):
    off = turns[lo]["off"]
    return {"lo": lo, "hi": hi, "closed": closed, "events": [], "ts": turns[lo]["ts"], "off": off,
            "id": hashlib.sha1(f"{sid}:calm:{off}".encode()).hexdigest()[:16]}


def calm(turns, sid, eps, final=False):
    """Tile the user turns outside the surprise episodes `eps` with calm chunks.

    A chunk starts where the previous unit ended and closes at a later user turn, so the
    cut depends only on turns at or after its start. The trailing chunk stays open.
    """
    is_user = [t["kind"] == "user" for t in turns]
    out, start = [], next((i for i, u in enumerate(is_user) if u), None)
    for ep in eps + [None]:
        if start is None:
            break
        stop = ep["lo"] if ep else len(turns) - 1
        lo, steps = start, 0
        for g in (i for i in range(start + 1, stop + 1) if is_user[i]):
            steps += 1
            if steps >= CALM_STEPS or g - lo >= CALM_TURNS:
                out.append(_chunk(turns, sid, lo, g, True))
                lo, steps = g, 0
        if stop > lo:
            out.append(_chunk(turns, sid, lo, stop, bool(ep) or final))
        start = ep["hi"] if ep and is_user[ep["hi"]] else None
    return out


def spans(turns, sid, final=False):
    """Every unit the extractor reads: surprise episodes and calm chunks, in file order."""
    eps = cut(turns, sid, final)
    return sorted(eps + calm(turns, sid, eps, final), key=lambda e: (e["lo"], e["hi"]))


def _clipped(turn, is_event):
    cap = 2000 if is_event else CLIP[turn["kind"]]
    if turn["kind"] == "result" and turn.get("err"):
        cap = 500
    text = turn["text"]
    return text if len(text) <= cap else text[:cap] + " [...]"


def render(turns, ep, budget=14000):
    """Return (window text, {label: turn}) under a character budget.

    The opening request, every event and the closing turn are always kept. The remaining
    budget goes to the turns nearest an event, so a long episode loses its middle first.
    A calm chunk has no event: there the user turns and failed results take that role.
    """
    ev = [i for i, _ in ep["events"]]
    anchors = set(ev) | {ep["lo"], ep["hi"]}
    span = range(ep["lo"], ep["hi"] + 1)
    focus = ev or [i for i in span if turns[i]["kind"] == "user" or turns[i].get("err")]
    text = {i: _clipped(turns[i], i in ev) for i in span}
    keep, used = set(anchors), sum(len(text[i]) for i in anchors)
    for i in sorted((i for i in span if i not in anchors), key=lambda i: min(abs(i - e) for e in focus)):
        if used + len(text[i]) > budget:
            break
        keep.add(i)
        used += len(text[i])
    lines, labels, prev, n = [], {}, ep["lo"] - 1, 0
    names = {"user": "USER", "assistant": "ASSISTANT", "tool": "TOOL", "result": "RESULT",
             "interrupt": "USER"}
    for i in span:
        if i not in keep:
            continue
        if i - prev > 1:
            lines.append(f"[... {i - prev - 1} turns omitted ...]")
        n, prev = n + 1, i
        labels[n] = turns[i]
        tag = names[turns[i]["kind"]] + (" ERROR" if turns[i].get("err") else "")
        lines.append(f"[{n}] {tag}: {text[i]}")
    return "\n".join(lines), labels


def norm(s):
    return re.sub(r"\s+", " ", s).strip().lower()
