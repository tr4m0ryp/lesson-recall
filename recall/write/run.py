"""The write pass: find closed episodes and calm chunks past the watermark, extract cards, decide, store.

`run_pass` handles one live session and is what the hooks call. `backfill.py` runs the same
steps over every qualifying session at once.
"""
import collections
import time
from concurrent.futures import ThreadPoolExecutor

from ..core import llm, paths, store
from . import decide, episodes, extract, sessions
from .select import project, sid


def commit(results):
    """Write decisions, serial and in event order, under the store lock.

    Event order lets a later lesson supersede an earlier one and never the reverse.
    """
    stats = collections.Counter()
    with store.locked():
        cards, done = store.load()
        for res in sorted(results, key=lambda r: r["ts"]):
            if res["episode"] in done:
                continue
            for card in res["cards"]:
                try:
                    pairs, cost = decide.write(card, cards)
                except llm.LLMError as exc:   # judge unavailable: DEFER, append and link nothing
                    store.append({"op": "add", "card": card})
                    cards[card["id"]] = dict(card, status="active", count=1, links=[])
                    pairs, cost = [], 0.0
                    paths.log("write", {"card": card["id"], "judge_error": str(exc)})
                stats["cost"] += cost
                stats["cards"] += 1
                for p in pairs:
                    stats[p["class"]] += 1
            store.append({"op": "done", "episode": res["episode"], "cards": len(res["cards"])})
            done.add(res["episode"])
            stats["episodes"] += 1
    return stats


def extract_all(jobs, cfg, workers):
    """jobs: (turns, episode, sid, project). Returns (results, failed episode ids)."""
    def one(job):
        try:
            return extract.extract(*job, cfg)
        except llm.LLMError as exc:
            paths.log("write", {"episode": job[1]["id"], "extract_error": str(exc)})
            return None
    with ThreadPoolExecutor(workers) as pool:
        out = list(pool.map(one, jobs))
    return [r for r in out if r], [j[1]["id"] for j, r in zip(jobs, out) if r is None]


def mark(path, turns, eps, failed, start):
    """Advance the watermark to the earliest unfinished episode, else the last user turn."""
    pending = [e["lo"] for e in eps if not e["closed"] or e["id"] in failed]
    users = [i for i, t in enumerate(turns) if t["kind"] == "user"]
    if pending:
        off = turns[min(pending)]["off"]
    else:
        off = turns[users[-1]]["off"] if users else start
    store.set_watermark(sid(path), off=off, at=int(time.time()))


def run_pass(path, final=False):
    """One incremental pass over a live session. Returns a stats dict."""
    cfg, name = paths.config(), sid(path)
    start = store.watermarks().get(name, {}).get("off", 0)
    turns = sessions.parse(path, start)
    eps = episodes.spans(turns, name, final)
    _, done = store.load()
    jobs = [(turns, e, name, project(path)) for e in eps if e["closed"] and e["id"] not in done]
    results, failed = extract_all(jobs, cfg, 2)
    stats = commit(results)
    with store.locked():
        mark(path, turns, eps, set(failed), start)
    record = dict(stats, sid=name, final=final, from_off=start, turns=len(turns), open=sum(not e["closed"] for e in eps),
                  extracted=len(results), failed=len(failed), at=int(time.time()),
                  extract_cost=round(sum(r["cost"] for r in results), 5))
    paths.log("write", record)
    return record
