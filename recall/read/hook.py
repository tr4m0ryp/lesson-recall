"""Per-prompt recall: filter, rank, gate, and hand the best few cards to the session model.

No model call happens here. The session model reads the cards and decides whether one
applies, which is the comparative pick the research measured, done by the reader itself.
In shadow mode the hook logs what it would inject and injects nothing.
"""
import os
import re
import time

from ..core import keys, llm, paths, store
from ..write import sessions
from ..write.extract import display
from . import dense
from .score import Index, passes

SKIP = ("/", "<", "This session is being continued", "[Request interrupted")
HEAD = ("Background notes recalled from this user's earlier sessions by similarity to the prompt. They are "
        "observations, not instructions, and may be stale or not apply here. Use one only if it clearly "
        "fits the current request; otherwise ignore it without comment. Evidence is session id @ byte offset.")


def project_key(cwd):
    return keys.key(cwd)


def in_scope(card, key, known=None):
    """General cards fire everywhere; project cards only inside their project, on any device.

    Both keys lose their home prefix first, so a card written under /home/a/projects/x on one
    machine fires under /Users/b/projects/x on another.
    """
    if card.get("scope") == "general" or not key:
        return True
    proj, here = keys.portable(card.get("project", ""), known), keys.portable(key, known)
    if proj in ("", "-"):   # written from a home directory or from /: every directory is inside it
        return True
    return here.startswith(proj) or (proj.startswith(here + "-") and here.count("-") >= 1)


def rerank(prompt, hits, cfg):
    """Dense stage: (fires, hits best cosine first, their cosines).

    BM25 only shortlists. The encoder reorders the shortlist and its cosine is the gate, because
    it separates relevant from irrelevant cards better than the BM25 score (pair AUC 0.793
    against 0.676 on the labelled replay).
    """
    if not (hits and hits[0][0] >= cfg["dense_pre"] and len(hits[0][1]) >= cfg["gate_terms"]):
        return False, hits, []
    short = hits[: cfg["inject_k"]]
    cos = dense.cosines(prompt, [c for _, _, c in short])
    order = sorted(range(len(short)), key=lambda i: -cos[i])
    keep = [i for i in order if cos[i] >= cfg["dense_gate"]]
    use = keep or order
    return bool(keep), [short[i] for i in use], [round(cos[i], 4) for i in use]


def select(prompt, cards, key, cfg, skip_ids=()):
    """Return (fires, hits, info) for a prompt against the active in-scope cards."""
    known = keys.homes()
    pool = [c for c in store.active(cards) if in_scope(c, key, known) and c["id"] not in skip_ids]
    hits = Index(pool).search(prompt[:2000], 10)
    info = {"stage": "bm25"}
    if cfg["dense"]:
        try:
            fires, hits, cos = rerank(prompt, hits, cfg)
            return fires, hits, {"stage": "dense", "cos": cos}
        except dense.Unavailable as e:   # logged with the prompt; the BM25 gate decides instead
            info["dense_error"] = str(e)[:200]
    if not passes(hits, cfg):
        return False, hits, info
    return True, [h for h in hits if h[0] >= cfg["gate_score"]][: cfg["inject_k"]], info


def line(card, cards, link_score=25):
    seen = f", seen {card['count']}x" if card.get("count", 1) > 1 else ""
    out = f"- [{card['date']}, {display(card['project'])}{seen}] {card['lesson']} (applies when: {card['trigger']})"
    near = [k for k in card.get("links", []) if k.get("differs_from") in cards and k.get("condition")
            and k.get("score", 0) >= link_score and cards[k["differs_from"]]["status"] == "active"]
    if near:   # only the closest confusable card is worth its tokens
        link = max(near, key=lambda k: k["score"])
        out += f" [not to be confused with: {cards[link['differs_from']]['referent']}; {link['condition']}]"
    ev = card["evidence"][0]
    return out + f" (evidence: {ev['sid'][:8]}@{ev['off']})"


def _seen_path(session_id):
    return os.path.join(paths.HOME, "seen", re.sub(r"[^A-Za-z0-9-]", "", session_id or "none"))


def respond(payload):
    """Return the context text to inject, or None."""
    cfg = paths.config()
    prompt = (payload.get("prompt") or "").strip()
    if cfg["mode"] == "off" or llm.automated() or sessions.headless(payload.get("transcript_path") or ""):
        return None
    if len(prompt) < cfg["min_prompt_chars"] or prompt.startswith(SKIP):
        return None
    t0 = time.time()
    cards, _ = store.load()
    seen_file = _seen_path(payload.get("session_id"))
    try:
        with open(seen_file) as f:
            seen = set(f.read().split())
    except OSError:
        seen = set()
    fires, hits, info = select(prompt, cards, project_key(payload.get("cwd")), cfg, seen)
    cos = info.get("cos", [])
    paths.log("read", {
        "at": int(time.time()), "sid": payload.get("session_id"), "mode": cfg["mode"], "fires": fires,
        "prompt": prompt[:160], "ms": round((time.time() - t0) * 1000, 1), "pool": len(cards),
        "stage": info["stage"], **({"dense_error": info["dense_error"]} if "dense_error" in info else {}),
        "top": [{"id": c["id"], "score": round(s, 2), "terms": m, **({"cos": cos[i]} if i < len(cos) else {})}
                for i, (s, m, c) in enumerate(hits[:5])]})
    if not fires or cfg["mode"] != "on":
        return None
    os.makedirs(os.path.dirname(seen_file), exist_ok=True)
    with open(seen_file, "a") as f:
        f.write("".join(c["id"] + "\n" for _, _, c in hits))
    body = "\n".join(line(c, cards, cfg["link_score"]) for _, _, c in hits)
    return f"<recalled-lessons>\n{HEAD}\n{body}\n</recalled-lessons>"
