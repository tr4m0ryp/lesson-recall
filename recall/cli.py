#!/usr/bin/env python3
"""Lesson recall command line.

  cli.py backfill [--dry-run] [--limit N]  turn past sessions into cards; --dry-run only counts, no model call
  cli.py pass [--transcript P] [--final]   one incremental pass (hook payload on stdin if no P)
  cli.py hook                              read side; reads the UserPromptSubmit payload on stdin
  cli.py query TEXT [--cwd DIR]            show what a prompt would recall
  cli.py mode on|shadow|off                switch injection (shadow: cards are written, nothing injected)
  cli.py stats | replay                    store counts | firing counts over past prompts
  cli.py calibrate [--rate R]              fit the gate thresholds to a target firing rate
  cli.py dense install|status|off          second gate stage: a local encoder reranks the BM25 shortlist
  cli.py sync init|now|status|scrub        optional: back the cards up to a private repository
  cli.py setup [--sync]                    encoder install, gate fit, mode on; --sync adds the backup first
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from recall.commands import install, live, show  # noqa: E402
from recall.core import paths  # noqa: E402


def parser():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("hook").set_defaults(fn=live.cmd_hook)
    p = sub.add_parser("pass")
    p.add_argument("--transcript")
    p.add_argument("--final", action="store_true")
    p.add_argument("--force", action="store_true")
    p.set_defaults(fn=live.cmd_pass)
    p = sub.add_parser("backfill")
    p.add_argument("--dry-run", action="store_true", help="print the plan as JSON; no model call")
    p.add_argument("--limit", type=int, help="only the N most recently modified qualifying sessions")
    p.add_argument("--workers", type=int, default=4)
    p.set_defaults(fn=install.cmd_backfill)
    p = sub.add_parser("query")
    p.add_argument("text")
    p.add_argument("--cwd", default=os.getcwd())
    p.set_defaults(fn=show.cmd_query)
    p = sub.add_parser("mode")
    p.add_argument("mode", choices=["on", "shadow", "off"])
    p.set_defaults(fn=show.cmd_mode)
    sub.add_parser("stats").set_defaults(fn=show.cmd_stats)
    sub.add_parser("replay").set_defaults(fn=show.cmd_replay)
    p = sub.add_parser("dense")
    p.add_argument("action", choices=["install", "status", "off"])
    p.set_defaults(fn=install.cmd_dense)
    p = sub.add_parser("setup")
    p.add_argument("--sync", action="store_true", help="also create or join the private card repository")
    p.add_argument("--remote", help="with --sync: URL of a private repository instead of the default")
    p.add_argument("--assume-private", action="store_true", help="with --sync --remote: see `sync init --help`")
    p.set_defaults(fn=install.cmd_setup)
    from recall import sync
    sync.register(sub)
    p = sub.add_parser("calibrate")
    p.add_argument("--rate", type=float)
    p.set_defaults(fn=install.cmd_calibrate)
    return ap


def main():
    args = parser().parse_args()
    venv = os.path.join(paths.HOME, "venv")   # the dense stage's packages live in a private venv
    python = os.path.join(venv, "bin", "python")
    if os.path.exists(python) and os.path.realpath(sys.prefix) != os.path.realpath(venv):
        os.execv(python, [python] + sys.argv)
    args.fn(args)


if __name__ == "__main__":
    main()
