"""Two devices share cards through a bare repository standing in for the private remote."""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

CLI = str(Path(__file__).resolve().parents[1] / "recall" / "cli.py")


def card(i, project):
    return {"op": "add", "at": f"2026-01-01T00:00:{i:02d}Z", "card": {
        "id": f"card{i:08d}", "referent": f"thing {i}", "lesson": f"lesson number {i}", "trigger": f"when {i}",
        "date": "2026-01-01", "ts": f"2026-01-01T00:00:{i:02d}Z", "project": project, "episode": f"e{i}",
        "scope": "project", "kind": "correction", "evidence": []}}


class Device:
    def __init__(self, root, name, remote):
        self.home, self.name, self.remote = root / name, name, remote
        self.home.mkdir()
        self.events = self.home / "events.jsonl"

    def add(self, *ops):
        with open(self.events, "a", encoding="utf-8") as f:
            f.writelines(json.dumps(op) + "\n" for op in ops)

    def run(self, *args):
        out = subprocess.run([sys.executable, CLI, "sync", *args], capture_output=True, text=True,
                             env=dict(os.environ, RECALL_HOME=str(self.home)))
        try:
            return json.loads(out.stdout[out.stdout.index("{"):])
        except ValueError:
            return {"state": "error", "error": out.stderr + out.stdout}

    def init(self, *extra):
        return self.run("init", "--remote", str(self.remote), "--device", self.name, "--assume-private", *extra)

    def clone(self, *parts):
        return self.home.joinpath("sync", "repo", *parts)


@pytest.fixture
def pair(tmp_path):
    remote = tmp_path / "remote.git"
    subprocess.run(["git", "init", "-q", "--bare", str(remote)], check=True)
    return Device(tmp_path, "alpha", remote), Device(tmp_path, "beta", remote)


def test_cards_travel_both_ways_and_the_view_is_written(pair):
    a, b = pair
    a.add(card(1, "-home-alice-projects-x"), card(2, "-home-alice"))
    assert a.init()["state"] == "pushed"
    first = b.init()
    assert first["imported"] == 1 and first["cards"] == 2 and first["devices"] == ["alpha", "beta"]
    b.add(card(3, "-Users-bob-projects-x"))
    assert b.run("now")["state"] == "pushed"
    back = a.run("now")
    assert back["imported"] == 1 and back["cards"] == 3
    assert a.run("status")["cards_by_device"] == {"alpha": 2, "beta": 1}
    assert a.clone("cards", "index.md").exists() and a.clone("cards", "general.md").exists()
    assert "lesson number 3" in a.clone("cards", "projects", "x.md").read_text()
    manifest = json.loads(a.clone("manifest.json").read_text())["devices"]
    assert manifest["alpha"]["events"]["lines"] == 2 and manifest["beta"]["events"]["lines"] == 1


def test_a_sync_with_nothing_new_makes_no_commit(pair):
    a, _ = pair
    a.add(card(1, "-home-alice-projects-x"))
    a.init()
    assert a.run("now")["state"] == "clean"


def test_credential_shaped_text_blocks_the_push_until_scrubbed(pair):
    a, _ = pair
    a.add(card(1, "-home-alice-projects-x"))
    a.init()
    leak = card(2, "-home-alice-projects-x")
    token = "ghp_" + "a1B2c3D4" * 5                     # built here so no literal token sits in the repository
    leak["card"]["lesson"] = "the deploy token is " + token
    a.add(leak)
    blocked = a.run("now")
    assert blocked["state"] == "error" and "credential" in blocked["error"] and token not in json.dumps(blocked)
    assert token not in a.clone("events", "alpha.jsonl").read_text()
    assert a.run("scrub")["events"]
    assert a.run("now")["state"] == "pushed"
    assert token not in a.clone("events", "alpha.jsonl").read_text()


def test_a_second_machine_with_the_same_name_is_refused(pair):
    a, b = pair
    a.add(card(1, "-home-alice-projects-x"))
    a.init()
    b.name = "alpha"
    b.add(card(9, "-Users-bob-projects-y"))
    refused = b.init()
    assert refused["state"] == "error" and "--device" in refused["error"]


def test_a_reinstalled_machine_adopts_its_own_log(pair):
    a, b = pair
    a.add(card(1, "-home-alice-projects-x"))
    a.init()
    b.name = "alpha"
    assert b.init("--adopt")["cards"] == 1 and b.events.read_text() == a.events.read_text()
