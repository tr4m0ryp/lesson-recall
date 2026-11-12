"""Building the store and its gate: backfill (or its plan), tune, calibrate, dense, setup."""
import json
import os
import subprocess
import sys
import time

from ..core import paths

CLI = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "cli.py")


def cmd_backfill(args):
    from ..write import backfill
    if args.dry_run:
        print(json.dumps(backfill.plan(limit=args.limit), indent=2))
        return
    backfill.backfill(limit=args.limit, workers=args.workers,
                      say=lambda m: print(time.strftime("%H:%M:%S"), m, flush=True))


def cmd_calibrate(args):
    from ..read import replay
    print(json.dumps(replay.calibrate(args.rate)))


def cmd_tune(args):
    from ..read import tune
    print(json.dumps(tune.tune(), ensure_ascii=False, indent=2))


def cmd_dense(args):
    """install: private venv, pinned packages, checked model files, then fit the gates. off: BM25 gate only."""
    from ..read.dense import encoder, setup
    if args.action == "install":
        python = setup.install()
        paths.set_config(dense=True)
        subprocess.run([python, CLI, "calibrate"], check=True)
    elif args.action == "off":
        paths.set_config(dense=False)
    cfg = paths.config()
    print(json.dumps({"dense": cfg["dense"], "model_ready": encoder.ready(), "venv": os.path.exists(setup.PYTHON),
                      "dense_pre": cfg["dense_pre"], "dense_gate": cfg["dense_gate"]}))


def cmd_setup(args):
    """Local by default: adapt to the user, install the encoder, fit the gates, switch injection on.

    Tune runs first because its stop words change the BM25 scores the gates are fitted on. When the
    user's prompts are mostly not English, the English-only encoder is skipped and the BM25 gate fitted.
    --sync adds the backup first.
    """
    from ..read import tune
    steps = []
    if args.sync:
        init = ["sync", "init"] + (["--remote", args.remote] if args.remote else [])
        steps.append(init + (["--assume-private"] if args.assume_private else []))
    print("== tune", flush=True)
    tuned = tune.tune()
    print(json.dumps(tuned, ensure_ascii=False), flush=True)
    english = tuned["english_prompt_share"] is None or tuned["english_prompt_share"] >= tune.MOSTLY_ENGLISH
    steps += [["dense", "install"]] if english else [["dense", "off"], ["calibrate"]]
    steps += [["mode", "on"]]
    for step in steps:
        print("==", " ".join(step), flush=True)
        subprocess.run([sys.executable, CLI] + step, check=True)
