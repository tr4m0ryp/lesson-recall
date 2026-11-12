"""Append-only card store.

Cards are never edited or deleted. Every change is one operation appended to the event log
(add, bump, supersede, restore, link, done); the current cards are the replay of that log.
A bump keeps the full text of the card it absorbed, so no decision loses information and
each one can be undone by a later line.

Each device appends only to its own log. Sync copies the other devices' logs into remote/,
and the replay merges all of them by timestamp, so no two machines ever write the same file.
"""
import contextlib
import fcntl
import glob
import heapq
import json
import os
import time

from . import keys, paths


@contextlib.contextmanager
def locked():
    """Serialise writers across the parallel sessions that share the store."""
    paths.ensure()
    with open(paths.LOCK, "w") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)


def append(op):
    op.setdefault("at", time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    paths.ensure()
    with open(paths.EVENTS, "a", encoding="utf-8") as f:
        f.write(json.dumps(op, ensure_ascii=False) + "\n")


def _ops(path, device):
    try:
        f = open(path, encoding="utf-8")
    except FileNotFoundError:
        return
    with f:
        for line in f:
            try:
                op = json.loads(line)
            except ValueError:
                continue
            yield op.get("at", ""), device, op


def logs():
    """(device, path) of every event log: this device's first, then the synced ones."""
    own = keys.device()
    found = [(own, paths.EVENTS)]
    for path in sorted(glob.glob(os.path.join(paths.REMOTE, "*.jsonl"))):
        name = os.path.basename(path)[:-6]
        if name != own:
            found.append((name, path))
    return found


def load(path=None):
    """Replay the event logs. Returns (cards by id, set of finished episode ids).

    Adds are applied first: a decision on one device may name a card another device wrote,
    and clocks differ, so an operation can sort ahead of the add it refers to.
    """
    cards, done = {}, set()
    streams = [_ops(path, "")] if path else [_ops(p, d) for d, p in logs()]
    ops = list(heapq.merge(*streams, key=lambda row: row[0]))
    for _, device, op in ops:
        if op.get("op") == "add":
            card = dict(op["card"])
            card.setdefault("status", "active")
            card.setdefault("count", 1)
            card.setdefault("links", [])
            card.setdefault("evidence", [])
            if device:
                card.setdefault("device", device)
            cards.setdefault(card["id"], card)
    for _, _, op in ops:
        kind = op.get("op")
        if kind == "bump" and op["id"] in cards:
            card = cards[op["id"]]
            card["count"] += 1
            card["evidence"].append(op["evidence"])
            card.setdefault("absorbed", []).append(op.get("card", {}).get("lesson", ""))
        elif kind == "supersede" and op["id"] in cards:
            cards[op["id"]]["status"] = "superseded"
            cards[op["id"]]["links"].append({"superseded_by": op["by"]})
            if op["by"] in cards:
                cards[op["by"]]["links"].append({"supersedes": op["id"]})
        elif kind == "restore" and op["id"] in cards:
            cards[op["id"]]["status"] = "active"
            for c in cards.values():
                c["links"] = [k for k in c["links"] if k.get("supersedes") != op["id"]
                              and not (c["id"] == op["id"] and "superseded_by" in k)]
        elif kind == "link" and op["id"] in cards and op["other"] in cards:
            extra = {"condition": op.get("condition", ""), "score": op.get("score", 0)}
            cards[op["id"]]["links"].append(dict(extra, differs_from=op["other"]))
            cards[op["other"]]["links"].append(dict(extra, differs_from=op["id"]))
        elif kind == "done":
            done.add(op["episode"])
    return cards, done


def active(cards):
    return [c for c in cards.values() if c["status"] == "active"]


def log_pair(record):
    with open(paths.PAIRS, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def watermarks():
    try:
        with open(paths.WATERMARKS, encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, ValueError):
        return {}


def set_watermark(sid, **kv):
    marks = watermarks()
    marks.setdefault(sid, {}).update(kv)
    tmp = paths.WATERMARKS + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(marks, f)
    os.replace(tmp, paths.WATERMARKS)
