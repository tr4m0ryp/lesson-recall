"""Synthetic Claude Code session transcripts for the tests and the smoke run."""
import json
import os
import uuid

CORRECTED = [
    ("user", "please add a retry to the fetch helper in the http client module, it fails on flaky networks " * 3),
    ("assistant", "I added a simple loop that retries the request five times without any delay between tries."),
    ("user", "no, that is wrong, use the existing backoff helper instead of a bare loop, it already handles jitter"),
    ("assistant", "Switched the fetch helper to the shared backoff helper with jitter, and removed the loop."),
    ("user", "great, now document the helper in the module docstring and add a short usage example"),
    ("assistant", "Documented the helper with an example."),
]


def entry(role, text, entrypoint="cli", cwd="/home/alice/projects/shop-api", at=0):
    return {"type": role, "entrypoint": entrypoint, "cwd": cwd, "sessionId": "x", "isSidechain": False,
            "timestamp": f"2026-01-01T00:{at // 60:02d}:{at % 60:02d}Z",
            "message": {"role": role, "content": text if role == "user" else [{"type": "text", "text": text}]}}


def write(root, turns=CORRECTED, entrypoint="cli", project="-home-alice-projects-shop-api", name=None):
    """Write one session file under <root>/projects/<project>/ and return its path."""
    folder = os.path.join(root, "projects", project)
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, (name or str(uuid.uuid4())) + ".jsonl")
    with open(path, "w", encoding="utf-8") as f:
        for i, (role, text) in enumerate(turns):
            f.write(json.dumps(entry(role, text, entrypoint, at=i), separators=(",", ":")) + "\n")
    return path
