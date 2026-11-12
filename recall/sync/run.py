"""Card sync: each device pushes its own event log to a private repository and reads everyone else's.

The repository's branch (default `main`) holds one append-only log per device, so two machines never write the same file
and a push can only lose a race, never conflict. A lost race is retried from the new tip.
The local log stays the record of this device; the clone is rebuilt from it every time.

  events/<device>.jsonl   card operations, append-only
  pairs/<device>.jsonl    every judged card pair
  cards/                  the current cards as markdown (index, general, one file per project)
  manifest.json           per device: home key, line counts, sha256
"""
import fcntl
import glob
import hashlib
import json
import os
import shutil
import subprocess
import time

from ..core import keys, paths, store
from . import github, repo, scan, view

STATE = os.path.join(paths.SYNC, "state.json")


class SyncError(RuntimeError):
    pass


def _state():
    try:
        with open(STATE, encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, ValueError):
        return {}


def _sha(path, limit=None):
    digest, left = hashlib.sha256(), limit
    with open(path, "rb") as f:
        while left is None or left > 0:
            block = f.read(1 << 20 if left is None else min(1 << 20, left))
            if not block:
                break
            digest.update(block)
            left = None if left is None else left - len(block)
    return digest.hexdigest()


def _size(path):
    return os.path.getsize(path) if os.path.exists(path) else 0


def _put(src, dst):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copyfile(src, dst + ".tmp")
    os.replace(dst + ".tmp", dst)


def ready():
    return bool(paths.config()["sync_remote"]) and os.path.isdir(os.path.join(repo.DIR, ".git"))


def _import(me):
    """Copy the other devices' logs out of the clone. Returns how many changed."""
    changed = 0
    for src in sorted(glob.glob(os.path.join(repo.DIR, "events", "*.jsonl"))):
        name = os.path.basename(src)
        dst = os.path.join(paths.REMOTE, name)
        if name[:-6] != me and not (os.path.exists(dst) and _sha(dst) == _sha(src)):
            _put(src, dst)
            changed += 1
    try:
        with open(os.path.join(repo.DIR, "manifest.json"), encoding="utf-8") as f:
            homes = {d: row.get("home", "") for d, row in json.load(f)["devices"].items() if d != me}
    except (FileNotFoundError, ValueError, KeyError):
        homes = {}
    if homes:
        os.makedirs(paths.REMOTE, exist_ok=True)
        with open(keys.DEVICES + ".tmp", "w", encoding="utf-8") as f:
            json.dump(homes, f, indent=2, sort_keys=True)
        os.replace(keys.DEVICES + ".tmp", keys.DEVICES)
    return changed


def _export(me, adopt=False):
    """Copy this device's logs into the clone. The pushed copy must be a prefix of the local log."""
    with store.locked():
        for kind, src in (("events", paths.EVENTS), ("pairs", paths.PAIRS)):
            dst = os.path.join(repo.DIR, kind, me + ".jsonl")
            pushed = _size(dst)
            if pushed and not _size(src) and adopt:   # a reinstalled device takes its log back
                _put(dst, src)
            if pushed and (_size(src) < pushed or _sha(src, pushed) != _sha(dst)):
                raise SyncError(
                    f"the branch already holds a different {kind} log for device '{me}'. A second machine "
                    f"with this name: cli.py sync init --device NAME. This machine reinstalled: "
                    f"cli.py sync init --adopt")
            bad = scan.findings(src, pushed)
            if bad:
                kinds = sorted({k for _, k in bad})
                raise SyncError(f"{kind} log holds credential-shaped text ({', '.join(kinds)}) on line(s) "
                                f"{sorted({n for n, _ in bad})[:8]}; nothing was pushed. Run: cli.py sync scrub")
            if _size(src):
                _put(src, dst)


def _manifest(me):
    path = os.path.join(repo.DIR, "manifest.json")
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except (FileNotFoundError, ValueError):
        data = {"devices": {}}
    row = {"home": keys.home()}
    for kind in ("events", "pairs"):
        log = os.path.join(repo.DIR, kind, me + ".jsonl")
        if os.path.exists(log):
            with open(log, "rb") as f:
                row[kind] = {"lines": sum(1 for _ in f), "sha256": _sha(log)}
    data["devices"][me] = row
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)
        f.write("\n")
    return row


def _sync(cfg, say, adopt):
    me, branch = keys.device(), cfg["sync_branch"]
    imported = 0
    for _ in range(3):
        repo.refresh(branch)
        imported += _import(me)
        _export(me, adopt)
        cards, _ = store.load()
        view.write(repo.DIR, cards, keys.homes())
        row = _manifest(me)
        if github.is_private(cfg["sync_remote"]) is False:
            raise SyncError("the card repository is public. Cards quote session text; nothing was pushed")
        state = repo.publish(branch, f"sync({me}): {len(store.active(cards))} cards, "
                                     f"{row.get('events', {}).get('lines', 0)} events from this device")
        if state != "rejected":
            break
        say("push lost a race to another device, retrying from the new tip")
    else:
        raise SyncError("push rejected three times in a row")
    record = {"state": state, "imported": imported, "cards": len(cards),
              "devices": sorted(d for d, _ in store.logs())}
    if cfg["sync_mirrors"] and (state == "pushed" or imported):   # a mirror also gets what others pushed
        record["mirrors"] = repo.mirror(cfg["sync_mirrors"], branch)
    return record


def sync(say=lambda m: None, adopt=False):
    """One full sync. Returns the record that is also logged; state is error on any failure."""
    cfg = paths.config()
    if not ready():
        return {"state": "error", "error": "not initialised: run cli.py sync init"}
    paths.ensure()
    with open(os.path.join(paths.RUN, "sync.lock"), "w") as fh:
        try:
            fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            return {"state": "busy"}
        try:
            record = _sync(cfg, say, adopt)
        except (SyncError, repo.GitError, subprocess.TimeoutExpired, OSError) as exc:
            record = {"state": "error", "error": str(exc)[:400]}
        record.update(at=int(time.time()), device=keys.device())
        paths.log("sync", record)
        os.makedirs(paths.SYNC, exist_ok=True)
        with open(STATE + ".tmp", "w", encoding="utf-8") as f:
            json.dump(record, f)
        os.replace(STATE + ".tmp", STATE)
        return record


def maybe():
    """Sync from a background pass when it is due. Returns True when other devices' cards changed.

    Due: this device has unpushed lines and the last attempt is older than the push gap, or
    the last attempt is older than the interval (a quiet device still fetches).
    """
    if not ready():
        return False
    cfg, me = paths.config(), keys.device()
    dirty = _size(paths.EVENTS) != _size(os.path.join(repo.DIR, "events", me + ".jsonl"))
    gap = cfg["sync_push_gap_s"] if dirty else cfg["sync_interval_s"]
    if time.time() - _state().get("at", 0) < gap:
        return False
    return bool(sync().get("imported"))


def status():
    cfg, me = paths.config(), keys.device()
    cards, _ = store.load()
    per = {}
    for c in cards.values():
        per[c.get("device", "")] = per.get(c.get("device", ""), 0) + 1
    return {"ready": ready(), "device": me, "remote": repo._clean(cfg["sync_remote"]), "branch": cfg["sync_branch"],
            "mirrors": [repo._clean(m) for m in cfg["sync_mirrors"]], "last": _state(), "cards_by_device": per,
            "unpushed_bytes": _size(paths.EVENTS) - _size(os.path.join(repo.DIR, "events", me + ".jsonl"))}
