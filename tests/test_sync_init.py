"""`sync init`: a local bare repository with --remote, and the GitHub default with a stubbed CLI."""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from recall.sync import github, join, run

CLI = str(Path(__file__).resolve().parents[1] / "recall" / "cli.py")


def cli(home, *args, path=None):
    env = dict(os.environ, RECALL_HOME=str(home))
    if path is not None:
        env["PATH"] = path
    return subprocess.run([sys.executable, CLI, *args], capture_output=True, text=True, env=env, timeout=60)


def test_init_against_a_local_bare_repository_pushes_branch_main(tmp_path):
    remote = tmp_path / "cards.git"
    subprocess.run(["git", "init", "-q", "--bare", str(remote)], check=True)
    home = tmp_path / "store"
    home.mkdir()
    (home / "events.jsonl").write_text(json.dumps({"op": "add", "at": "2026-01-01T00:00:00Z", "card": {
        "id": "c1", "referent": "r", "lesson": "l", "trigger": "t", "project": "-home-alice-projects-x",
        "scope": "project", "date": "2026-01-01", "evidence": []}}) + "\n")
    out = cli(home, "sync", "init", "--remote", str(remote), "--device", "laptop", "--assume-private")
    record = json.loads(out.stdout[out.stdout.index("{"):])
    assert record["state"] == "pushed", out.stderr
    heads = subprocess.run(["git", "--git-dir", str(remote), "branch", "--list"], capture_output=True, text=True).stdout
    assert heads.split() == ["main"]
    files = subprocess.run(["git", "--git-dir", str(remote), "ls-tree", "-r", "--name-only", "main"],
                           capture_output=True, text=True).stdout.split()
    assert "events/laptop.jsonl" in files and "manifest.json" in files and "cards/index.md" in files


def test_init_without_gh_or_remote_says_sync_is_optional(tmp_path):
    bare = tmp_path / "bin"
    bare.mkdir()
    os.symlink(subprocess.check_output(["bash", "-c", "command -v git"], text=True).strip(), bare / "git")
    out = cli(tmp_path / "store", "sync", "init", path=str(bare))
    assert out.returncode == 1 and "not installed" in out.stderr and "works fully without it" in out.stderr
    assert not (tmp_path / "store" / "sync").exists()


def test_a_remote_that_gh_reports_public_is_refused(monkeypatch):
    monkeypatch.setattr(github, "is_private", lambda url: False)
    with pytest.raises(run.SyncError, match="public"):
        join.setup("https://github.com/someone/cards.git", say=lambda m: None)


def test_unknown_visibility_needs_assume_private(monkeypatch):
    monkeypatch.setattr(github, "is_private", lambda url: None)
    with pytest.raises(run.SyncError, match="--assume-private"):
        join.setup("git@example.org:cards.git", say=lambda m: None)


class FakeGh:
    """Stands in for the GitHub CLI: one account, a set of repositories, records every call."""

    def __init__(self, repos):
        self.repos, self.calls = repos, []

    def __call__(self, *args, check=True):
        self.calls.append(args)
        out = subprocess.CompletedProcess(args, 0, "", "")
        if args[:2] == ("api", "user"):
            out.stdout = "alice\n"
        elif args[:2] == ("repo", "view"):
            info = self.repos.get(args[2])
            out.returncode, out.stdout = (0, json.dumps(info)) if info else (1, "")
        elif args[:2] == ("repo", "create"):
            assert "--private" in args
            self.repos[args[2]] = {"isPrivate": True, "url": f"https://github.com/{args[2]}",
                                   "sshUrl": f"git@github.com:{args[2]}.git"}
        elif args[:2] == ("config", "get"):
            out.stdout = "https\n"
        return out


def test_default_repository_is_created_private_when_missing(monkeypatch):
    gh = FakeGh({})
    monkeypatch.setattr(github, "_gh", gh)
    monkeypatch.setattr(github.shutil, "which", lambda name: "/usr/bin/gh")
    url, slug = github.ensure("claude-recall-cards", say=lambda m: None)
    assert slug == "alice/claude-recall-cards" and url == "https://github.com/alice/claude-recall-cards.git"
    assert any(c[:2] == ("repo", "create") for c in gh.calls)


def test_an_existing_public_default_repository_is_refused(monkeypatch):
    gh = FakeGh({"alice/claude-recall-cards": {"isPrivate": False, "url": "u", "sshUrl": "s"}})
    monkeypatch.setattr(github, "_gh", gh)
    monkeypatch.setattr(github.shutil, "which", lambda name: "/usr/bin/gh")
    with pytest.raises(RuntimeError, match="not private"):
        github.ensure("claude-recall-cards", say=lambda m: None)
    assert not any(c[:2] == ("repo", "create") for c in gh.calls)
