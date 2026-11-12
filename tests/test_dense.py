"""The dense stage of the gate, with the encoder replaced by fixed cosines."""
import json

import pytest

from recall.core import paths, store
from recall.read import dense, hook
from test_read import CARDS

PROMPT = "the pre-push hook blocks pushing commits across several repositories, and tmux launch was rejected"


@pytest.fixture
def cfg(monkeypatch):
    open(paths.EVENTS, "w").close()
    for c in CARDS[:3]:
        store.append({"op": "add", "card": c})
    monkeypatch.setattr(hook.llm, "automated", lambda: False)
    return dict(paths.DEFAULTS, mode="on", dense=True, dense_pre=2.0, dense_gate=0.6, gate_score=2.0)


def fake(monkeypatch, by_id):
    monkeypatch.setattr(dense, "cosines", lambda prompt, cards: [by_id[c["id"]] for c in cards])


def test_cosine_reorders_the_shortlist_and_gates_each_card(cfg, monkeypatch):
    fake(monkeypatch, {"push": 0.61, "tmux": 0.74, "pace": 0.2})
    fires, hits, info = hook.select(PROMPT, store.load()[0], "", cfg)
    assert fires and [c["id"] for _, _, c in hits] == ["tmux", "push"]
    assert info == {"stage": "dense", "cos": [0.74, 0.61]}


def test_no_card_above_the_cosine_gate_means_no_fire(cfg, monkeypatch):
    fake(monkeypatch, {"push": 0.59, "tmux": 0.3, "pace": 0.1})
    fires, hits, info = hook.select(PROMPT, store.load()[0], "", cfg)
    assert not fires and info["cos"][0] == 0.59


def test_weak_keyword_match_never_reaches_the_encoder(cfg, monkeypatch):
    monkeypatch.setattr(dense, "cosines", lambda *a: pytest.fail("encoder ran below the pre-gate"))
    fires, _, info = hook.select(PROMPT, store.load()[0], "", dict(cfg, dense_pre=1e6))
    assert not fires and info == {"stage": "dense", "cos": []}


def test_a_broken_encoder_is_logged_and_bm25_decides(cfg, monkeypatch):
    def broken(prompt, cards):
        raise dense.Unavailable("model files missing")
    monkeypatch.setattr(dense, "cosines", broken)
    paths.set_config(**{k: cfg[k] for k in ("mode", "dense", "dense_pre", "dense_gate", "gate_score")})
    text = hook.respond({"prompt": PROMPT, "cwd": "/home/u/projects/alpha", "session_id": "dense-1"})
    assert "push guard" in text
    last = json.loads(open(paths.LOGS + "/read.jsonl").read().splitlines()[-1])
    assert last["stage"] == "bm25" and "model files missing" in last["dense_error"]


@pytest.mark.skipif(not dense.ready(), reason="encoder not installed in the test store")
def test_real_encoder_prefers_the_matching_card():
    cos = dense.cosines("git push is blocked by the pre-push hook", CARDS[:2])
    assert cos[0] > cos[1]
