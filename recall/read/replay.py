"""Calibrate the gate offline: replay historical user prompts through the read side.

Cards mined from a prompt's own session are held out, so a prompt cannot recall the lesson
it produced. `replay` reports the share of prompts that fire at each threshold and a sample;
`calibrate` sets the threshold from a target firing rate.
"""
import os
import random
import time

from ..core import paths, store
from ..write import select as stage0, sessions
from . import dense
from .hook import SKIP, select


def prompts(files):
    """Historical prompts the hook would have scored: (session id, project key, text)."""
    floor = paths.config()["min_prompt_chars"]
    for path in files:
        for turn in sessions.parse(path):
            text = turn["text"].strip()
            if turn["kind"] == "user" and len(text) >= floor and not text.startswith(SKIP):
                yield os.path.basename(path)[:-6], stage0.project(path), text


def _tops(cards, cfg):
    """Best (score, matched terms), prompt and BM25 shortlist per historical prompt, own session held out."""
    files, _ = stage0.candidates(cfg["extra_roots"])
    by_sid = {}
    for c in cards.values():
        for e in c["evidence"]:
            by_sid.setdefault(e["sid"], set()).add(c["id"])
    loose = dict(cfg, gate_score=0, gate_terms=0, dense=False)
    for sid, project, text in prompts(files):
        _, hits, _ = select(text, cards, project, loose, by_sid.get(sid, ()))
        yield (hits[0][0], len(hits[0][1]), text, hits) if hits else (0.0, 0, text, [])


def fit_dense(rows, cfg, rate):
    """Fit the pre-gate to `dense_pass` and the cosine gate to `rate`, both as quantiles."""
    ok = sorted((r for r in rows if r[3] and r[1] >= cfg["gate_terms"]), key=lambda r: -r[0])
    ok = ok[: int(cfg["dense_pass"] * len(rows)) + 1]
    cut = int(rate * len(rows))
    if cut >= len(ok):
        return None
    cards = {c["id"]: c for r in ok for _, _, c in r[3]}
    vec = dict(zip(cards, dense.vectors(list(cards.values()))))
    best = sorted((max(float(vec[c["id"]] @ q) for _, _, c in r[3]) for r, q in
                   zip(ok, dense.embed([r[2][:dense.encoder.PROMPT_CHARS] for r in ok]))), reverse=True)
    return {"dense_pre": round(ok[-1][0], 2), "dense_gate": round(best[cut], 4)}


def fit_links(cards, share):
    """link_score such that `share` of the stored differs_from links clear it, or None without links.

    The link score is a raw BM25 score, whose scale grows with the store, so a fixed value would
    carry the author's store into every other one.
    """
    scores = sorted((k["score"] for c in store.active(cards) for k in c.get("links", [])
                     if k.get("differs_from") and k.get("condition") and "score" in k), reverse=True)
    if not scores:
        return None
    return round(scores[min(len(scores) - 1, max(0, int(share * len(scores)) - 1))], 3)


def calibrate(rate=None):
    """Set gate_score so that `rate` of historical prompts fire, and record the pool size.

    The best score among more cards is higher, so a fixed threshold fires more often as the
    store grows: 37.5% of prompts at 172 cards became 55.3% at 413. The gate is therefore a
    quantile of the replayed scores, refitted when the pool has grown.
    """
    cfg = paths.config()
    rate = rate or cfg["gate_rate"]
    cards, _ = store.load()
    rows = list(_tops(cards, cfg))
    ok = sorted((s for s, m, _, _ in rows if m >= cfg["gate_terms"]), reverse=True)
    cut = int(rate * len(rows))
    if not rows or cut >= len(ok):
        return None
    score = round(ok[cut], 2)
    out = {"at": int(time.time()), "prompts": len(rows), "rate": rate, "gate_score": score,
           "cards": len(store.active(cards)), "was": cfg["gate_score"]}
    link = fit_links(cards, cfg["link_share"])
    if link is not None:
        out["link_score"] = link
    if cfg["dense"]:
        try:
            out.update(fit_dense([(s, m, t, h[: cfg["inject_k"]]) for s, m, t, h in rows], cfg, rate) or {})
        except dense.Unavailable as e:
            out["dense_error"] = str(e)[:200]
    paths.set_config(gate_rate=rate, gate_cards=len(store.active(cards)),
                     **{k: out[k] for k in ("gate_score", "dense_pre", "dense_gate", "link_score") if k in out})
    paths.log("calibrate", out)
    return out


def replay(thresholds=(8, 10, 12, 14, 16, 18), sample=12, seed=0):
    cfg = paths.config()
    cards, _ = store.load()
    rows = list(_tops(cards, cfg))
    out = {"prompts": len(rows), "fires": {}}
    for th in thresholds:
        out["fires"][th] = sum(1 for s, m, _, _ in rows if s >= th and m >= cfg["gate_terms"])
    random.Random(seed).shuffle(rows)
    out["sample"] = [
        {"score": round(s, 2), "terms": h[0][1], "prompt": t[:200], "card": h[0][2]["lesson"][:200]}
        for s, m, t, h in rows if h and s >= cfg["gate_score"] and m >= cfg["gate_terms"]][:sample]
    return out
