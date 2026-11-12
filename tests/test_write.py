import json

import pytest

from recall.core import paths, store
from recall.write import decide, episodes, extract, sessions


def turn(kind, text, off, err=False):
    return {"kind": kind, "text": text, "off": off, "ts": f"2026-01-01T00:00:{off:02d}Z", "err": err}


BASE = [turn("user", "add a retry to the fetch helper", 0), turn("assistant", "Added a loop.", 1),
        turn("user", "no, that is wrong, use the backoff helper instead", 2),
        turn("assistant", "Switched to backoff.", 3)]
CALM = turn("user", "great, now document the helper", 4)


def test_episode_stays_open_until_a_calm_user_turn():
    (ep,) = episodes.cut(BASE, "s")
    assert not ep["closed"] and ep["events"] == [(2, "correction")]
    (ep,) = episodes.cut(BASE + [CALM], "s")
    assert ep["closed"] and (ep["lo"], ep["hi"]) == (0, 4)
    assert episodes.cut(BASE, "s", final=True)[0]["closed"]


def test_first_user_turn_is_never_an_event():
    assert episodes.cut([turn("user", "stop using print, don't log secrets", 0)], "s") == []


def test_interrupt_reply_does_not_close_its_own_episode():
    turns = [turn("user", "deploy the service", 0), turn("interrupt", "[user interrupted]", 1),
             turn("user", "use the staging cluster", 2), turn("assistant", "Deploying to staging.", 3)]
    assert not episodes.cut(turns, "s")[0]["closed"]
    assert episodes.cut(turns + [CALM], "s")[0]["hi"] == 4


def test_error_burst_is_one_event():
    turns = [turn("user", "run the tests", 0)] + [turn("result", "boom", i, err=True) for i in range(1, 6)]
    assert [k for _, k in episodes.cut(turns, "s", final=True)[0]["events"]] == ["errors"]


def test_cut_is_stable_when_the_parse_starts_at_a_calm_turn():
    more = [CALM, turn("assistant", "Documented.", 5), turn("user", "wrong file, revert that", 6),
            turn("user", "thanks, next task please", 7)]
    whole = episodes.cut(BASE + more, "s")
    tail = episodes.cut(more, "s")
    assert len(whole) == 2 and [e["id"] for e in tail] == [whole[1]["id"]]


def _calm_session(n):
    turns = []
    for k in range(n):
        turns += [turn("user", f"please build part {k}", 2 * k), turn("assistant", f"Built part {k}.", 2 * k + 1)]
    return turns


def test_calm_chunks_tile_the_stretch_between_episodes():
    turns = _calm_session(7)
    chunks = episodes.spans(turns, "s")
    assert [(c["lo"], c["hi"], c["closed"]) for c in chunks] == [(0, 6, True), (6, 12, True), (12, 13, False)]
    assert episodes.spans(turns, "s", final=True)[-1]["closed"]
    mixed = episodes.spans(_calm_session(2) + [turn("user", "no, that is wrong, redo part 1", 4), CALM], "s")
    assert [(c["lo"], c["hi"], bool(c["events"])) for c in mixed] == [(0, 2, False), (2, 5, True)]


def test_calm_chunks_are_stable_when_the_parse_starts_at_an_open_chunk():
    turns = _calm_session(9)
    early = episodes.spans(turns[:9], "s")           # the pass that ran mid-session
    assert not early[-1]["closed"]
    resumed = episodes.spans(turns[early[-1]["lo"]:], "s")
    whole = episodes.spans(turns, "s")
    assert [c["id"] for c in early[:-1]] + [c["id"] for c in resumed] == [c["id"] for c in whole]


def test_render_handles_a_chunk_without_events():
    turns = _calm_session(3)
    text, _ = episodes.render(turns, episodes.spans(turns, "s", final=True)[0], budget=3000)
    assert "build part 0" in text and "Built part 2" in text


def test_render_keeps_anchors_and_marks_omissions():
    filler = [turn("assistant", "x" * 600, i) for i in range(1, 40)]
    turns = [BASE[0]] + filler + [turn("user", "no, wrong approach, stop", 40), CALM]
    (ep,) = episodes.cut(turns, "s")
    text, labels = episodes.render(turns, ep, budget=3000)
    assert "add a retry" in text and "wrong approach" in text and "document the helper" in text
    assert "turns omitted" in text and len(text) < 4500
    assert labels[1] is turns[0]


def test_quote_must_be_verbatim():
    labels = {1: BASE[0], 2: BASE[2]}
    ok = extract._verified({"turn": 2, "quote": "use the backoff   helper instead"}, labels, "")
    assert ok and ok[0] is BASE[2]
    assert extract._verified({"turn": 1, "quote": "use the backoff helper instead"}, labels, "")[0] is BASE[2]
    assert extract._verified({"turn": 2, "quote": "always use exponential backoff"}, labels, "") is None


def test_parse_skips_meta_redacts_and_ignores_a_half_line(tmp_path):
    secret = "ghp_" + "a1B2" * 9
    rows = [
        {"type": "user", "message": {"role": "user", "content": f"push with {secret} please"}},
        {"type": "user", "isMeta": True, "message": {"role": "user", "content": "meta noise"}},
        {"type": "assistant", "message": {"role": "assistant", "content": [
            {"type": "text", "text": "Pushing."}, {"type": "tool_use", "name": "Bash", "input": {"command": "git push"}}]}},
        {"type": "user", "message": {"role": "user", "content": [
            {"type": "tool_result", "is_error": True, "content": "rejected"}]}},
        {"type": "user", "message": {"role": "user", "content": "<system-reminder>x</system-reminder>"}},
    ]
    path = tmp_path / "s.jsonl"
    path.write_text("".join(json.dumps(r, separators=(",", ":")) + "\n" for r in rows) + '{"type":"user","mess')
    turns = sessions.parse(str(path))
    assert [t["kind"] for t in turns] == ["user", "assistant", "tool", "result"]
    assert secret not in turns[0]["text"] and "REDACTED" in turns[0]["text"]
    assert turns[3]["err"]
    assert sessions.user_turns(str(path))[0] == 1


def card(cid, lesson, ts="2026-01-01T00:00:00Z"):
    return {"id": cid, "referent": "git push guard", "lesson": lesson, "ts": ts, "date": ts[:10],
            "trigger": "pushing commits across several repositories with a push hook", "project": "-p",
            "scope": "general", "evidence": [{"sid": "s", "off": 1, "quote": "q"}]}


@pytest.fixture
def fresh(monkeypatch):
    for path in (paths.EVENTS, paths.PAIRS):
        open(path, "w").close()
    old = card("old", "cd into each repository before pushing it")
    store.append({"op": "add", "card": old})

    def answer(rho, nu, condition="", obsolete="yes"):
        monkeypatch.setattr(decide.llm, "ask", lambda *_a, **_k: ({"obsolete": obsolete, "pairs": [
            {"old": "old", "same_referent": rho, "same_value": nu, "condition": condition}]}, 0.0))
        new = card("new", "push from inside each repository, the hook checks the shell directory",
                   "2026-02-01T00:00:00Z")
        cards, _ = store.load()
        pairs, _ = decide.write(new, cards)
        return pairs[0]["class"], store.load()[0]
    return answer


def test_duplicate_bumps_and_writes_no_new_card(fresh):
    cls, cards = fresh("yes", "yes")
    assert cls == "DUPLICATE" and set(cards) == {"old"}
    assert cards["old"]["count"] == 2 and len(cards["old"]["evidence"]) == 2


def test_supersession_keeps_both_and_retires_the_older(fresh):
    cls, cards = fresh("yes", "no")
    assert cls == "SUPERSESSION" and set(cards) == {"old", "new"}
    assert cards["old"]["status"] == "superseded" and cards["new"]["status"] == "active"
    assert {"supersedes": "old"} in cards["new"]["links"]


def test_unconfirmed_supersession_retires_nothing(fresh):
    cls, cards = fresh("yes", "no", obsolete="no")
    assert cls == "DEFER" and all(c["status"] == "active" for c in cards.values())


def test_restore_undoes_a_supersession(fresh):
    fresh("yes", "no")
    store.append({"op": "restore", "id": "old"})
    cards = store.load()[0]
    assert cards["old"]["status"] == "active" and not cards["old"]["links"] and not cards["new"]["links"]


def test_duplicate_keeps_the_absorbed_text(fresh):
    _, cards = fresh("yes", "yes")
    assert "hook checks the shell directory" in cards["old"]["absorbed"][0]


def test_coincidence_and_defer_keep_both_unlinked(fresh):
    for rho, nu, want in (("no", "yes", "COINCIDENCE"), ("unsure", "no", "DEFER")):
        cls, cards = fresh(rho, nu)
        assert cls == want and all(c["status"] == "active" and not c["links"] for c in cards.values())


def test_interference_links_only_with_a_named_condition(fresh):
    cls, cards = fresh("no", "no", "push versus status check")
    assert cls == "INTERFERENCE" and cards["new"]["links"][0]["differs_from"] == "old"
    cls, cards = fresh("no", "no")
    assert cls == "INTERFERENCE"
    assert len(cards["old"]["links"]) == 1   # the second, unconditioned pair added no link


def test_a_refused_episode_is_recorded_as_read_without_a_card(monkeypatch):
    def refuse(*_a, **_k):
        raise extract.llm.Refused("declined")
    monkeypatch.setattr(extract.llm, "ask", refuse)
    turns = [turn("user", "explain the long failure in the build " * 12, 0), turn("assistant", "Done.", 1), CALM]
    ep = dict(episodes.spans(turns, "s", final=True)[0], id="refused-episode")
    res = extract.extract(turns, ep, "s", "-home-u-projects-x", paths.config())
    assert res["refused"] and res["cards"] == []
