"""`cli.py sync ...`: join the shared card store, sync now, inspect, or mask a blocked credential."""
import json
import sys

from ..core import paths, store
from . import join, run, scan


def command(args):
    if args.action == "init":
        try:
            record = join.setup(args.remote, args.device, args.adopt, args.assume_private)
        except run.SyncError as exc:
            sys.exit(f"sync init: {exc}")
    elif args.action == "now":
        record = run.sync(print, args.adopt)
    elif args.action == "scrub":
        with store.locked():
            record = {kind: dict(scan.scrub(path)) for kind, path in (("events", paths.EVENTS), ("pairs", paths.PAIRS))}
    else:
        record = run.status()
    print(json.dumps(record, indent=2))
    if record.get("state") == "error":
        sys.exit(1)


def register(sub):
    p = sub.add_parser("sync", help="back the cards up to a private repository and share them across devices")
    p.add_argument("action", choices=["init", "now", "status", "scrub"])
    p.add_argument("--remote", help="URL of a private repository; default: <gh user>/claude-recall-cards, "
                                    "created private through the GitHub CLI when missing")
    p.add_argument("--device", help="name of this machine in the shared store; default: the hostname")
    p.add_argument("--adopt", action="store_true", help="a reinstalled machine takes its own log back from the branch")
    p.add_argument("--assume-private", action="store_true", help="with --remote: skip the visibility check for a repository "
                                                               "gh cannot see (you checked it is private)")
    p.set_defaults(fn=command)
