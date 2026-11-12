"""Stage 0 of the write side: which session files are worth reading at all. No model call.

A session qualifies when it has two or more typed user turns, its opening prompt is not one
that five or more sessions share (a templated or scripted opener), and it was not started in
print mode or through the Agent SDK. Subagent transcripts are never read: their lessons
belong to the parent session.
"""
import collections
import glob
import os
import re
from multiprocessing import Pool

from ..core import paths
from . import sessions

MIN_TURNS, MAX_OPENER = 2, 5


def sid(path):
    return os.path.basename(path)[:-6]


def project(path):
    """Project key: the session directory name, or for archived copies the munged cwd."""
    if os.path.realpath(path).startswith(os.path.realpath(paths.sessions()) + os.sep):
        return os.path.basename(os.path.dirname(path))
    with open(path, "r", errors="replace") as fh:
        for _, line in zip(range(40), fh):
            hit = re.search(r'"cwd":"([^"]+)"', line)
            if hit:
                return re.sub(r"[^A-Za-z0-9]", "-", hit.group(1))
    return os.path.basename(os.path.dirname(path))


def _probe(path):
    try:
        return (path,) + sessions.user_turns(path)
    except OSError:
        return None


def files(roots=()):
    """Every top-level session transcript under the session root and any extra roots, one per id."""
    found, seen = [], set()
    for root in [paths.sessions()] + list(roots or []):
        for path in glob.glob(os.path.join(os.path.expanduser(root), "**", "*.jsonl"), recursive=True):
            if "/subagents/" in path or sid(path) in seen:
                continue
            seen.add(sid(path))
            found.append(path)
    return found


def survey(roots=()):
    """{"scanned", "headless", "too_short", "templated", "qualify", "files"}: the stage-0 funnel."""
    every = files(roots)
    rows = []
    if every:
        with Pool(min(12, len(every))) as pool:
            rows = [r for r in pool.imap_unordered(_probe, every, chunksize=50) if r]
    openers = collections.Counter(first for _, n, first, scripted in rows if not scripted)
    out = collections.Counter()
    keep = []
    for path, n, first, scripted in rows:
        if scripted:
            out["headless"] += 1
        elif n < MIN_TURNS:
            out["too_short"] += 1
        elif openers[first] >= MAX_OPENER:
            out["templated"] += 1
        else:
            keep.append(path)
    return dict(out, scanned=len(every), qualify=len(keep), files=sorted(keep))


def candidates(roots=()):
    """(qualifying session paths, number of files scanned)."""
    found = survey(roots)
    return found["files"], found["scanned"]
