import os

from recall.core import keys, llm
from recall.core.redact import find, redact
from recall.read import hook

KNOWN = ["-home-alice", "-Users-bob-smith"]


def card(project, scope="project"):
    return {"scope": scope, "project": project}


def test_portable_strips_each_known_home():
    assert keys.portable("-home-alice-projects-x", KNOWN) == "projects-x"
    assert keys.portable("-Users-bob-smith-projects-x", KNOWN) == "projects-x"
    assert keys.portable("-Users-bob-smith", KNOWN) == ""
    assert keys.portable("-", KNOWN) == "-"


def test_portable_falls_back_to_the_home_pattern_for_an_unseen_device():
    assert keys.portable("-Users-carol-projects-x", KNOWN) == "projects-x"
    assert keys.portable("-srv-data-x", KNOWN) == "-srv-data-x"


def test_project_card_fires_in_the_same_project_on_another_device():
    here = keys.key("/Users/bob-smith/projects/x")
    assert hook.in_scope(card("-home-alice-projects-x"), here, KNOWN)
    assert hook.in_scope(card("-home-alice-projects-x"), keys.key("/Users/bob-smith/projects/x/sub"), KNOWN)
    assert not hook.in_scope(card("-home-alice-projects-y"), here, KNOWN)


def test_parent_directory_sees_child_project_cards_only_when_deep_enough():
    child = card("-home-alice-projects-research-x")
    assert hook.in_scope(child, keys.key("/Users/bob-smith/projects/research"), KNOWN)
    assert not hook.in_scope(child, keys.key("/Users/bob-smith/projects"), KNOWN)
    assert not hook.in_scope(child, keys.key("/Users/bob-smith"), KNOWN)


def test_cards_written_from_home_or_root_fire_everywhere():
    here = keys.key("/home/alice/projects/x")
    assert hook.in_scope(card("-Users-bob-smith"), here, KNOWN)
    assert hook.in_scope(card("-"), here, KNOWN)
    assert hook.in_scope(card("-home-alice-projects-y", scope="general"), here, KNOWN)


def test_process_lookup_through_ps_matches_proc(monkeypatch):
    argv, parent = llm._process(os.getpid())
    monkeypatch.setattr(llm.os.path, "isdir", lambda p: False)
    ps_argv, ps_parent = llm._process(os.getpid())
    assert ps_parent == parent == os.getppid()
    assert os.path.basename(ps_argv[0]) == os.path.basename(argv[0])


def test_added_credential_shapes_are_masked():
    stripe = "sk_" + "live_" + "a1B2" * 6
    url = "postgres://app:" + "s3cretPw99" + "@db.internal:5432/x"
    text = f"key {stripe} and {url} and https://user:$PASSWORD@host/"
    assert {k for k, _ in find(text)} == {"stripe-key", "url-password"}
    out = redact(text)
    assert stripe not in out and "s3cretPw99" not in out
    assert "postgres://app:[REDACTED:url-password]@db.internal" in out and "$PASSWORD" in out
