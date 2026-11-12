"""Stage-0 selection and the backfill plan: CLAUDE_CONFIG_DIR is honoured and no model is called."""
import json
import os

import pytest

import transcripts
from recall.core import llm, paths, store
from recall.write import backfill, select


@pytest.fixture
def config_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path / "cfg"))
    for path in (paths.EVENTS, paths.PAIRS):
        open(path, "w").close()

    def forbidden(*_a, **_k):
        raise AssertionError("the plan called the model")
    monkeypatch.setattr(llm, "ask", forbidden)
    return tmp_path / "cfg"


def test_sessions_are_found_under_claude_config_dir(config_dir, monkeypatch):
    path = transcripts.write(str(config_dir))
    assert paths.sessions() == str(config_dir / "projects")
    assert select.files() == [path]
    monkeypatch.delenv("CLAUDE_CONFIG_DIR")
    assert paths.sessions().endswith("/.claude/projects")


def test_session_project_key_comes_from_its_directory(config_dir):
    path = transcripts.write(str(config_dir))
    assert select.project(path) == "-home-alice-projects-shop-api"


def test_stage0_excludes_headless_short_and_templated_sessions(config_dir):
    keep = transcripts.write(str(config_dir))
    transcripts.write(str(config_dir), entrypoint="sdk-cli")
    transcripts.write(str(config_dir), turns=transcripts.CORRECTED[:2])
    for _ in range(5):
        transcripts.write(str(config_dir), turns=[("user", "run the nightly report now"), ("assistant", "ok"),
                                                   ("user", "and mail it"), ("assistant", "done")])
    found = select.survey()
    assert found["scanned"] == 8 and found["files"] == [keep]
    assert (found["headless"], found["too_short"], found["templated"]) == (1, 1, 5)


def test_plan_counts_units_and_calls_without_any_model_call(config_dir):
    transcripts.write(str(config_dir))
    transcripts.write(str(config_dir), entrypoint="sdk-ts")
    plan = backfill.plan()
    json.dumps(plan)
    assert plan["sessions_scanned"] == 2 and plan["sessions_qualify"] == 1
    assert plan["sessions_excluded"]["headless"] == 1
    assert plan["units_to_process"] >= 1 and plan["haiku_calls_extract"] >= 1
    assert plan["haiku_calls_judge_at_most"] == 2 * plan["haiku_calls_extract"]
    assert store.load()[0] == {}


def test_plan_limit_takes_the_newest_sessions(config_dir):
    old = transcripts.write(str(config_dir))
    os.utime(old, (1, 1))
    transcripts.write(str(config_dir))
    assert backfill.plan(limit=1)["sessions_selected"] == 1
    assert backfill._chosen(select.survey(), 1) != [old]


def test_cached_units_are_not_counted_as_calls(config_dir):
    transcripts.write(str(config_dir))
    first = backfill.plan()
    turns_jobs = backfill._units(select.survey()["files"], set())[1]
    paths.ensure()
    for _, ep, _, _ in turns_jobs:
        with open(os.path.join(paths.CACHE, ep["id"] + ".json"), "w") as f:
            f.write("{}")
    again = backfill.plan()
    assert again["haiku_calls_extract"] == 0 and again["units_cached"] == first["units_to_process"]

