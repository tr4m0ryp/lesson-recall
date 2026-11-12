"""Where the card store lives, where the sessions are, and how the store is configured.

The store sits outside every repository's working tree: cards quote session text. The only copy
that leaves the machine is the one `sync` pushes to a private repository, and only once the user
has run `cli.py sync init`.
"""
import json
import os

HOME = os.environ.get("RECALL_HOME") or os.path.expanduser("~/.local/share/claude-recall")


def claude_dir():
    """Claude Code's config directory: $CLAUDE_CONFIG_DIR when set, else ~/.claude."""
    return os.path.expanduser(os.environ.get("CLAUDE_CONFIG_DIR") or "~/.claude")


def sessions():
    """Root of the session transcripts, read at call time so CLAUDE_CONFIG_DIR is honoured."""
    return os.path.join(claude_dir(), "projects")


EVENTS = os.path.join(HOME, "events.jsonl")        # append-only card operations
PAIRS = os.path.join(HOME, "pairs.jsonl")          # every judged card pair
WATERMARKS = os.path.join(HOME, "watermarks.json")
CONFIG = os.path.join(HOME, "config.json")
CACHE = os.path.join(HOME, "cache")                # one extraction result per episode
LOGS = os.path.join(HOME, "log")
RUN = os.path.join(HOME, "run")                    # working directory of headless calls
LOCK = os.path.join(HOME, ".lock")
REMOTE = os.path.join(HOME, "remote")              # other devices' event logs, replaced whole by sync
SYNC = os.path.join(HOME, "sync")                  # clone of the card repository plus sync state

DEFAULTS = {
    "mode": "off",             # off | shadow | on; off until setup has the user's go-ahead
    "gate_score": 10.5,        # minimum BM25 score: for the prompt to fire, and for each card
    "gate_rate": 0.33,         # share of past prompts that should fire; calibrate fits gate_score to it
    "gate_cards": 0,           # active cards when gate_score was last fitted (0: never)
    "gate_growth": 1.15,       # refit once the pool has grown by this factor
    "gate_terms": 2,           # minimum distinct query terms the best card must match
    "dense": False,            # second stage: a local encoder reranks the BM25 shortlist (cli.py dense install)
    "dense_pass": 0.70,        # share of past prompts the BM25 pre-gate lets through to the encoder
    "dense_pre": 8.0,          # BM25 score of the best card for the encoder to run; fitted to dense_pass
    "dense_gate": 0.68,        # minimum prompt-to-card cosine: for the prompt to fire, and for each card
    "inject_k": 3,
    "link_score": 25,          # a differs_from link is shown only for pairs at least this close
    "min_prompt_chars": 24,
    "window_chars": 14000,
    "pass_interval_s": 120,    # per-session spacing of background passes
    "extra_roots": [],         # further session roots for the backfill
    "device": "",              # this machine's name in the shared store; set once by sync init
    "sync_repo": "claude-recall-cards",   # name of the private GitHub repository sync init creates or reuses
    "sync_remote": "",         # its URL as this device reaches it; set by sync init
    "sync_branch": "main",
    "sync_mirrors": [],        # further private repositories that get a push-only copy
    "sync_interval_s": 1800,   # how often a quiet device still fetches the other devices' cards
    "sync_push_gap_s": 300,    # minimum spacing of syncs while this device has unpushed cards
}


def ensure():
    for d in (HOME, CACHE, LOGS, RUN):
        os.makedirs(d, exist_ok=True)


def config():
    cfg = dict(DEFAULTS)
    try:
        with open(CONFIG, encoding="utf-8") as f:
            cfg.update(json.load(f))
    except FileNotFoundError:
        pass
    return cfg


def set_config(**kv):
    ensure()
    cfg = {}
    try:
        with open(CONFIG, encoding="utf-8") as f:
            cfg = json.load(f)
    except FileNotFoundError:
        pass
    cfg.update(kv)
    tmp = CONFIG + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)
    os.replace(tmp, CONFIG)
    return cfg


def log(name, record):
    """Append one JSON record to log/<name>.jsonl. Logging never raises into a hook."""
    try:
        ensure()
        with open(os.path.join(LOGS, name + ".jsonl"), "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except OSError:
        pass
