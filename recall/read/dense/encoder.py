"""bge-small-en-v1.5 through onnxruntime, with a content-addressed vector cache for cards.

The heavy imports happen on first use, so a prompt that fails the BM25 pre-gate pays nothing.
Every failure surfaces as `Unavailable`; the caller logs it and gates on BM25 instead.
"""
import hashlib
import os

from ...core import paths

MODEL = "bge-small-en-v1.5"
DIR = os.path.join(paths.HOME, "model", MODEL)
VECS = os.path.join(paths.HOME, "vectors", MODEL)
FILES = ("model_optimized.onnx", "tokenizer.json")
PROMPT_CHARS, BATCH = 1200, 16
_state = {}


class Unavailable(RuntimeError):
    """The encoder cannot run here: missing packages, missing model files, or a failed run."""


def ready():
    return all(os.path.exists(os.path.join(DIR, f)) for f in FILES)


def card_text(card):
    return f"{card['referent']}. {card['trigger']} {card['lesson']}"


def _load():
    if _state:
        return _state
    if not ready():
        raise Unavailable(f"model files missing in {DIR}")
    try:
        import numpy as np
        import onnxruntime as ort
        from tokenizers import Tokenizer
        tok = Tokenizer.from_file(os.path.join(DIR, "tokenizer.json"))
        tok.enable_truncation(512)
        tok.enable_padding()
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 2
        opts.log_severity_level = 3
        sess = ort.InferenceSession(os.path.join(DIR, FILES[0]), opts, providers=["CPUExecutionProvider"])
    except Exception as e:
        raise Unavailable(f"{type(e).__name__}: {e}") from e
    _state.update(np=np, tok=tok, sess=sess)
    return _state


def embed(texts):
    """Unit vectors, one row per text (the CLS vector, L2-normalised)."""
    np, tok, sess = (_load()[k] for k in ("np", "tok", "sess"))
    order = sorted(range(len(texts)), key=lambda i: len(texts[i]))   # similar lengths share a batch
    out = np.zeros((len(texts), 384), dtype=np.float32)
    try:
        for at in range(0, len(order), BATCH):
            idx = order[at:at + BATCH]
            enc = tok.encode_batch([texts[i] for i in idx])
            ids = np.array([e.ids for e in enc], dtype=np.int64)
            mask = np.array([e.attention_mask for e in enc], dtype=np.int64)
            vec = sess.run(None, {"input_ids": ids, "attention_mask": mask,
                                  "token_type_ids": np.zeros_like(ids)})[0][:, 0]
            out[idx] = vec / np.linalg.norm(vec, axis=1, keepdims=True)
    except Exception as e:
        raise Unavailable(f"{type(e).__name__}: {e}") from e
    return out


def vectors(cards):
    """Card vectors, read from the cache where the card text is unchanged."""
    np = _load()["np"]
    files = [os.path.join(VECS, hashlib.sha1(card_text(c).encode()).hexdigest()[:20] + ".npy") for c in cards]
    out, missing = [None] * len(cards), []
    for i, path in enumerate(files):
        try:
            out[i] = np.load(path)
        except (OSError, ValueError):
            missing.append(i)
    if missing:
        os.makedirs(VECS, exist_ok=True)
        for i, vec in zip(missing, embed([card_text(cards[i]) for i in missing])):
            out[i] = vec
            tmp = f"{files[i]}.{os.getpid()}.tmp.npy"
            np.save(tmp, vec)
            os.replace(tmp, files[i])
    return np.vstack(out) if out else np.zeros((0, 384), dtype=np.float32)


def cosines(prompt, cards):
    """Cosine between a prompt and each card."""
    if not cards:
        return []
    return [float(x) for x in vectors(cards) @ embed([prompt[:PROMPT_CHARS]])[0]]
