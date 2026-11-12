"""Adapting to the user: Unicode terms, learned stop words, configurable correction cues, tune."""
import json
import os
import re
import shutil

import pytest

from recall.commands import config as config_cmd
from recall.core import llm, paths, store, text
from recall.read import replay, tune
from recall.read.score import tokens
from recall.write import episodes, extract

from tests import transcripts

GERMAN = ["bitte die Tests im Ordner {} mit der neuen Flagge starten und das Ergebnis zeigen",
          "warum ist der Build für {} schon wieder kaputt, prüfe die Größe der Datei",
          "nein, das ist falsch, benutze den vorhandenen Helfer für {} und nicht die Schleife",
          "schreibe die Dokumentation für {} und füge ein kurzes Beispiel hinzu"]


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    """Own session root, a saved config, no stop word file, and no model call allowed."""
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path / "cfg"))
    saved = open(paths.CONFIG).read() if os.path.exists(paths.CONFIG) else None

    def forbidden(*_a, **_k):
        raise AssertionError("tune called the model")
    monkeypatch.setattr(llm, "ask", forbidden)
    yield tmp_path / "cfg"
    if saved is None:
        os.path.exists(paths.CONFIG) and os.remove(paths.CONFIG)
    else:
        with open(paths.CONFIG, "w") as f:
            f.write(saved)
    if os.path.exists(paths.STOPWORDS):
        os.remove(paths.STOPWORDS)


def old_tokens(s):
    """The ASCII tokenizer before this change, kept here to pin English and Dutch behaviour."""
    return [text.stem(t) for t in re.findall(r"[a-z0-9_]+", s.lower())
            if len(t) > 1 and t not in text.BUILTIN_STOP]


def test_ascii_english_and_dutch_terms_are_unchanged_without_tuning():
    for s in ["Pushing commits across several repositories when a pre-push hook blocks the push",
              "de tests draaien niet meer, gebruik de backoff helper in plaats van een lus",
              "run_all.py --flag=3 failed again on the staging deploy"]:
        assert tokens(s) == old_tokens(s)


def test_non_ascii_words_survive_and_cjk_becomes_bigrams():
    assert "größe" in tokens("Prüfe die Größe der Datei")
    assert "ändern" in tokens("Bitte ÄNDERN")
    assert text.words("テストを実行") == ["テス", "スト", "トを", "を実", "実行"]
    assert text.words("pytest実行") == ["pytest", "実行"]


def test_learn_takes_frequent_terms_of_any_language_as_stop_words():
    prompts = [p.format(f"modul{i}") for i in range(30) for p in GERMAN]
    cfg = dict(paths.DEFAULTS, stop_min_prompts=50)
    got = tune.learn(prompts, cfg)
    assert got["learned"] and got["prompts"] == 120
    assert {"die", "der", "und", "für"} <= set(got["words"])
    assert "modul7" not in got["words"]                     # in 4 of 120 prompts, under 10%
    assert got["english_prompt_share"] == 0.0
    assert got["non_ascii_token_share"] > 0


def test_too_few_prompts_keep_the_builtin_list():
    got = tune.learn(["the tests fail again on the staging server"] * 10, dict(paths.DEFAULTS))
    assert not got["learned"]


def test_tune_is_idempotent_makes_no_model_call_and_changes_tokens(isolated):
    for i in range(30):
        transcripts.write(str(isolated), [("user", p.format(f"modul{i}")) if j % 2 == 0 else ("assistant", "ok")
                                          for j, p in enumerate(x for g in GERMAN for x in (g, g))])
    paths.set_config(stop_min_prompts=50)
    assert "und" in tokens("Tests und Build")               # built-in list knows no German
    first = tune.tune()
    assert first["stopwords"] == "learned" and first["changed"]["stopwords"]["from"] == "built-in"
    assert first["language_estimate"] == "mostly not English" and first["gate_suggestion"].startswith("bm25")
    assert "und" not in tokens("Tests und Build")
    saved = open(paths.STOPWORDS).read()
    second = tune.tune()
    assert second["changed"] == {} and open(paths.STOPWORDS).read() == saved
    assert second["top_stopwords"] == first["top_stopwords"]


def test_correction_cues_come_from_the_config(isolated):
    turns = [{"kind": "user", "text": t, "off": i, "ts": "2026-01-01T00:00:00Z"} for i, t in
             enumerate(["füge einen Retry hinzu", "das ist falsch, nimm den Helfer", "違うよ、ヘルパーを使って"])]
    assert episodes.events(turns) == []
    config_cmd.change("add", "correction_cues", "falsch")
    config_cmd.change("add", "correction_cues", "違う")
    assert [k for k in episodes.events(turns)] == [(1, "correction"), (2, "correction")]
    assert episodes.events(turns, cues=["nope"]) == []


def test_default_cues_match_the_old_pattern():
    old = re.compile(r"\b(no[,. ]|not what|wrong|stop|don'?t|instead|again|still (broken|not|fail)|revert|undo|"
                     r"why did you|you (forgot|missed|broke)|nee[,. ]|niet|fout|verkeerd|opnieuw|terug)\b", re.I)
    new = episodes.cue_pattern(tuple(paths.DEFAULTS["correction_cues"]))
    for s in ["no, use the helper", "No. Stop.", "that is not what I asked", "it is still broken", "dat is fout",
              "nothing", "notwithstanding", "you broke it", "don't", "nee dank je", "stopwatch", "again?"]:
        assert bool(old.search(s)) == bool(new.search(s)), s


def test_config_command_checks_keys_types_and_regexes(isolated):
    with pytest.raises(config_cmd.ConfigError):
        config_cmd.change("set", "no_such_key", "1")
    with pytest.raises(config_cmd.ConfigError):
        config_cmd.change("set", "stop_df", "lots")
    with pytest.raises(config_cmd.ConfigError):
        config_cmd.change("add", "correction_cues", "(unclosed")
    assert config_cmd.change("set", "stop_df", "0.2")["stop_df"] == 0.2
    assert config_cmd.change("set", "card_language", "German")["card_language"] == "German"
    assert "terug" not in config_cmd.change("remove", "correction_cues", "terug")["correction_cues"]


def test_cards_are_written_in_the_configured_language(isolated, monkeypatch):
    seen = {}

    def ask(system, prompt):
        seen["system"] = system
        return {"cards": []}, 0.0
    monkeypatch.setattr(llm, "ask", ask)
    cache = os.path.join(paths.CACHE)
    turns = [{"kind": "user", "text": "x" * 400, "off": 0, "ts": "2026-01-01T00:00:00Z"}]
    ep = {"id": "lang-test", "lo": 0, "hi": 0, "events": [], "ts": "2026-01-01T00:00:00Z"}
    extract.extract(turns, ep, "s", "-home-u-projects-x", dict(paths.config(), card_language="German"))
    assert "trigger in German." in seen["system"]
    os.remove(os.path.join(cache, "lang-test.json"))


def test_link_score_is_a_quantile_of_the_stored_links():
    cards = {f"c{i}": {"id": f"c{i}", "status": "active",
                       "links": [{"differs_from": "x", "condition": "c", "score": float(i)}]} for i in range(10)}
    assert replay.fit_links(cards, 0.5) == 5.0
    assert replay.fit_links({"a": {"id": "a", "status": "active", "links": []}}, 0.5) is None
    assert store and shutil and json   # imported for fixtures in sibling tests
