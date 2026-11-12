"""The two hook entry points: `hook` (read side, every prompt) and `pass` (write side, in the background)."""
import fcntl
import json
import os
import sys
import time

from ..core import llm, paths, store


def cmd_hook(_):
    from ..read import hook
    try:
        text = hook.respond(json.load(sys.stdin))
    except Exception as exc:   # a recall failure must never cost the user a prompt
        paths.log("read", {"at": int(time.time()), "error": repr(exc)[:300]})
        return
    if text:
        print(json.dumps({"hookSpecificOutput": {"hookEventName": "UserPromptSubmit",
                                                 "additionalContext": text}}))


def _detach():
    """Leave the hook: the parent returns at once, the child carries on without its pipes."""
    if os.fork():
        os._exit(0)
    os.setsid()
    null = os.open(os.devnull, os.O_RDWR)
    for fd in (0, 1, 2):
        os.dup2(null, fd)


def cmd_pass(args):
    from ..write import run, sessions
    path, final, force = args.transcript, args.final, args.force
    if not path:   # called as a hook: the payload is on stdin
        payload = json.load(sys.stdin)
        path, event = payload.get("transcript_path"), payload.get("hook_event_name")
        final, force = event == "SessionEnd", event in ("SessionEnd", "PreCompact")
        if llm.automated():
            return
    if not path or not os.path.isfile(path) or "/subagents/" in path or sessions.headless(path):
        return
    cfg, sid = paths.config(), os.path.basename(path)[:-6]
    if cfg["mode"] == "off":
        return
    last = store.watermarks().get(sid, {}).get("at", 0)
    if not (final or force) and time.time() - last < cfg["pass_interval_s"]:
        return
    if not args.transcript:
        _detach()
    paths.ensure()
    with open(os.path.join(paths.RUN, sid + ".pass.lock"), "w") as fh:
        try:
            fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            return   # another pass already holds this session
        record = run.run_pass(path, final=final)
    if args.transcript:
        print(json.dumps(record))
        return
    from .. import sync
    arrived = sync.maybe()   # push this device's new cards, fetch the other devices'
    if record.get("cards") or arrived:
        _refit_gate()


def _refit_gate():
    """Refit the gate when the pool has outgrown the one it was fitted on. One fitter at a time."""
    cfg = paths.config()
    cards, _ = store.load()
    if not cfg["gate_cards"] or len(store.active(cards)) < cfg["gate_growth"] * cfg["gate_cards"]:
        return
    from ..read import replay
    with open(os.path.join(paths.RUN, "calibrate.lock"), "w") as fh:
        try:
            fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            return
        replay.calibrate()
