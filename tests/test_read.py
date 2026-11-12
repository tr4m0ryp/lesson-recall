import json

import pytest

from recall.core import paths, store
from recall.read import hook
from recall.read.score import Index


def card(cid, referent, lesson, trigger, scope="general", project="-home-u-projects-alpha"):
    return {"id": cid, "referent": referent, "lesson": lesson, "trigger": trigger, "scope": scope,
            "project": project, "date": "2026-01-01", "ts": "2026-01-01T00:00:00Z",
            "evidence": [{"sid": "abcdef0123", "off": 7, "quote": "q"}]}


CARDS = [
    card("push", "git push guard hook", "The push guard checks the shell directory, so cd into each repository first.",
         "pushing commits across several repositories when a pre-push hook blocks the push"),
    card("tmux", "detached tmux launch", "A rejected tmux launch may still have started, so check the pane.",
         "a tmux or background session launch was rejected or interrupted"),
    card("pace", "follow bot pacing", "Fast follow bursts trigger a rate limit pause, keep pacing conservative.",
         "changing follow pacing, gaps or hourly caps for the follow bot", scope="project"),
    card("old", "git push guard hook", "Push from the home directory.",
         "pushing commits across several repositories when a pre-push hook blocks the push"),
]


@pytest.fixture
def stored(monkeypatch):
    open(paths.EVENTS, "w").close()
    for c in CARDS:
        store.append({"op": "add", "card": c})
    store.append({"op": "supersede", "id": "old", "by": "push"})
    monkeypatch.setattr(hook.llm, "automated", lambda: False)
    paths.set_config(mode="on", dense=False, gate_score=2.0, gate_terms=2, inject_k=3, link_score=25)
    return store.load()[0]


def ask(prompt, cwd="/home/u/projects/alpha", sid="sess-1"):
    return hook.respond({"prompt": prompt, "cwd": cwd, "session_id": sid})


def test_related_prompt_recalls_the_active_card_only(stored):
    text = ask("the pre-push hook blocks my push across several repositories, fix it")
    assert "cd into each repository" in text and "home directory" not in text
    assert "observations, not instructions" in text and "abcdef01@7" in text


def test_unrelated_and_trivial_prompts_inject_nothing(stored):
    assert ask("write a haiku about autumn leaves falling slowly") is None
    assert ask("yes") is None
    assert ask("/compact please keep the push hook details") is None
    assert ask("<task-notification>the pre-push hook blocks my push across several repositories") is None


def test_project_cards_stay_in_their_project(stored):
    prompt = "change the follow pacing and hourly caps of the follow bot"
    assert "rate limit pause" in ask(prompt, sid="a")
    assert ask(prompt, cwd="/home/u/projects/beta", sid="b") is None


def test_a_card_is_injected_once_per_session(stored):
    prompt = "the pre-push hook blocks my push across several repositories"
    assert ask(prompt, sid="once") and ask(prompt, sid="once") is None


def test_shadow_logs_and_injects_nothing(stored):
    paths.set_config(mode="shadow")
    assert ask("the pre-push hook blocks my push across several repositories", sid="sh") is None
    last = json.loads(open(paths.LOGS + "/read.jsonl").read().splitlines()[-1])
    assert last["fires"] and last["top"][0]["id"] == "push"


def test_automated_sessions_never_get_cards(stored, monkeypatch):
    monkeypatch.setattr(hook.llm, "automated", lambda: True)
    assert ask("the pre-push hook blocks my push across several repositories", sid="bot") is None


def test_index_ranks_the_matching_card_first():
    hits = Index(CARDS[:3]).search("tmux session launch was interrupted")
    assert hits[0][2]["id"] == "tmux"
