"""Backfill: run the write pass over every qualifying session at once, or only plan it.

Extraction runs in parallel, write decisions serially and in event order. The run is
resumable: each extraction result is cached per unit (`cache/<unit>.json`), each finished
unit is recorded as a `done` operation in the event log, and each session keeps a watermark.
A rerun after an interruption skips finished units and reuses cached extractions, so it makes
no model call twice for the same unit; only the judging of a unit cut off mid-decision repeats.
"""
import os
import time

from ..core import paths, store
from . import episodes, extract, run, sessions
from .select import project, sid, survey

STALE_S = 6 * 3600   # a session untouched this long is treated as ended


def _chosen(found, limit):
    """The qualifying sessions to read: all of them, or the `limit` most recently modified."""
    files = found["files"]
    if limit:
        files = sorted(files, key=os.path.getmtime, reverse=True)[:limit]
    return files


def _units(files, done):
    """(parsed sessions, jobs): every closed unit not yet finished, per session."""
    parsed, jobs = {}, []
    for path in files:
        final = time.time() - os.path.getmtime(path) > STALE_S
        turns = sessions.parse(path)
        eps = episodes.spans(turns, sid(path), final)
        parsed[path] = (turns, eps)
        jobs += [(turns, e, sid(path), project(path)) for e in eps if e["closed"] and e["id"] not in done]
    return parsed, jobs


def plan(roots=None, limit=None):
    """What a backfill would do, without calling any model. Returns a JSON-ready dict.

    Extraction makes one Haiku call per unit that is not cached and whose window reaches the
    extractor's minimum length. Each stored card then gets at most one judging call (none when
    it has no near neighbour), and a proposed supersession one more confirming call.
    """
    cfg = paths.config()
    found = survey(roots or cfg["extra_roots"])
    files = _chosen(found, limit)
    _, done = store.load()
    parsed, jobs = _units(files, done)
    calls = cached = short = 0
    for turns, ep, _, _ in jobs:
        if extract.cached(ep["id"]):
            cached += 1
        elif len(episodes.render(turns, ep, cfg["window_chars"])[0]) < extract.MIN_WINDOW:
            short += 1
        else:
            calls += 1
    return {
        "sessions_scanned": found["scanned"],
        "sessions_excluded": {"headless": found.get("headless", 0), "under_2_user_turns": found.get("too_short", 0),
                              "templated_opener": found.get("templated", 0)},
        "sessions_qualify": found["qualify"],
        "sessions_selected": len(files),
        "units_total": sum(len(eps) for _, eps in parsed.values()),
        "units_already_done": sum(e["closed"] and e["id"] in done for _, eps in parsed.values() for e in eps),
        "units_open": sum(not e["closed"] for _, eps in parsed.values() for e in eps),
        "units_to_process": len(jobs),
        "units_cached": cached,
        "units_too_short": short,
        "haiku_calls_extract": calls,
        "haiku_calls_judge_at_most": 2 * (calls + cached),
        "haiku_calls_confirm": "one per proposed supersession; rare",
        "note": "Extract calls are exact. Judge calls: at most one per new card, at most 2 cards per unit, "
                "none for a card without a near neighbour. Most units yield no card. Every call runs "
                "through your own claude login and counts against your plan's usage.",
    }


def backfill(roots=None, limit=None, workers=4, say=print):
    """Convert every qualifying session (or the `limit` newest) in one batch. Safe to stop and rerun."""
    cfg = paths.config()
    found = survey(roots or cfg["extra_roots"])
    files = _chosen(found, limit)
    say(f"sessions: {found['scanned']} scanned, {found['qualify']} qualify, {len(files)} selected")
    _, done = store.load()
    parsed, jobs = _units(files, done)
    say(f"units: {sum(len(e) for _, e in parsed.values())} cut, {len(jobs)} to extract")
    results, failed = run.extract_all(jobs, cfg, workers)
    say(f"extracted: {len(results)} ok, {len(failed)} failed, "
        f"{sum(len(r['cards']) for r in results)} cards, {sum(r['dropped'] for r in results)} dropped")
    stats = run.commit(results)
    with store.locked():
        for path, (turns, eps) in parsed.items():
            run.mark(path, turns, eps, set(failed), 0)
    record = dict(stats, sessions=len(parsed), failed=len(failed), at=int(time.time()), backfill=True,
                  extract_cost=round(sum(r["cost"] for r in results), 5))
    paths.log("write", record)
    say(f"written: {dict(stats)}")
    return record
