"""Install the dense stage: a private venv and the encoder files, each checked against a pinned sha256."""
import hashlib
import os
import subprocess
import sys
import urllib.request

from ...core import paths
from . import encoder

VENV = os.path.join(paths.HOME, "venv")
PYTHON = os.path.join(VENV, "bin", "python")
PINS = ("onnxruntime==1.31.1", "tokenizers==0.23.3", "numpy==2.5.4")
FLOORS = ("onnxruntime>=1.17", "tokenizers>=0.15", "numpy>=1.26")   # for a Python the pins have no wheel for
SOURCE = "https://huggingface.co/Qdrant/bge-small-en-v1.5-onnx-Q/resolve/aa8f8b060edb00e03bfdd08813a2949946c8ba55/"
SHA256 = {
    "model_optimized.onnx": "51f1bd0addd6e859e42c2c8021a5e5461385bb676a649f4b269aa445449f2431",
    "tokenizer.json": "d241a60d5e8f04cc1b2b3e9ef7a4921b27bf526d9f6050ab90f9267a1f9e5c66",
}


def _sha(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def install(say=print):
    """Create the venv, install the pinned packages and fetch the model. Safe to run again."""
    if not os.path.exists(PYTHON):
        say(f"creating {VENV}")
        subprocess.run([sys.executable, "-m", "venv", VENV], check=True)
    pip = [PYTHON, "-m", "pip", "install", "-q", "--disable-pip-version-check"]
    say("installing " + " ".join(PINS))
    if subprocess.run(pip + list(PINS)).returncode:
        say("PINNED VERSIONS UNAVAILABLE for this Python; installing " + " ".join(FLOORS))
        subprocess.run(pip + list(FLOORS), check=True)
        got = subprocess.run([PYTHON, "-m", "pip", "freeze"], capture_output=True, text=True).stdout
        say("installed instead: " + " ".join(v for v in got.split() if v.split("==")[0] in ("onnxruntime", "tokenizers", "numpy")))
    os.makedirs(encoder.DIR, exist_ok=True)
    for name, want in SHA256.items():
        path = os.path.join(encoder.DIR, name)
        if not (os.path.exists(path) and _sha(path) == want):
            say(f"fetching {name}")
            urllib.request.urlretrieve(SOURCE + name, path + ".part")
            got = _sha(path + ".part")
            if got != want:
                os.remove(path + ".part")
                raise RuntimeError(f"{name}: sha256 {got} does not match the pinned {want}")
            os.replace(path + ".part", path)
    return PYTHON
