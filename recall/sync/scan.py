"""Credential check on everything that is about to leave the machine.

Cards are redacted when they are written. This is the second look at push time, with the
same patterns, so a pattern added later also covers text stored before it existed.
"""
import collections
import json
import os

from ..core.redact import find, redact


def findings(path, start=0):
    """[(line number, kind)] for credential-shaped text from byte `start` on. Values stay here."""
    hits = []
    try:
        f = open(path, "rb")
    except FileNotFoundError:
        return hits
    with f:
        line_no = f.read(start).count(b"\n")
        for raw in f:
            line_no += 1
            hits += [(line_no, kind) for kind, _ in find(raw.decode("utf-8", "replace"))]
    return hits


def _mask(value, counts):
    if isinstance(value, str):
        return redact(value, counts)
    if isinstance(value, list):
        return [_mask(v, counts) for v in value]
    if isinstance(value, dict):
        return {k: _mask(v, counts) for k, v in value.items()}
    return value


def scrub(path):
    """Mask credentials in a JSONL log in place. Lines without a finding keep their bytes."""
    counts = collections.Counter()
    try:
        with open(path, encoding="utf-8") as f:
            lines = f.readlines()
    except FileNotFoundError:
        return counts
    out = []
    for line in lines:
        if find(line):
            line = json.dumps(_mask(json.loads(line), counts), ensure_ascii=False) + "\n"
        out.append(line)
    if counts:
        with open(path + ".tmp", "w", encoding="utf-8") as f:
            f.writelines(out)
        os.replace(path + ".tmp", path)
    return counts
