"""The plugin surface: manifests parse, hooks.json points at executable scripts, hooks stay silent."""
import json
import os
import re
import subprocess
from pathlib import Path

import transcripts

ROOT = Path(__file__).resolve().parents[1]
PROMPT = {"session_id": "smoke-1", "hook_event_name": "UserPromptSubmit", "cwd": "/home/alice/projects/shop-api",
          "prompt": "add a retry with backoff to the fetch helper in the http client"}


def scripts():
    hooks = json.loads((ROOT / "hooks" / "hooks.json").read_text())["hooks"]
    commands = [h["command"] for groups in hooks.values() for g in groups for h in g["hooks"]]
    return hooks, commands


def test_hooks_json_registers_the_four_events_with_timeouts():
    hooks, commands = scripts()
    assert set(hooks) == {"UserPromptSubmit", "Stop", "PreCompact", "SessionEnd"}
    timeouts = {event: groups[0]["hooks"][0]["timeout"] for event, groups in hooks.items()}
    assert timeouts == {"UserPromptSubmit": 5, "Stop": 10, "PreCompact": 10, "SessionEnd": 10}
    assert all("${CLAUDE_PLUGIN_ROOT}" in c and "~/.claude" not in c for c in commands)


def test_every_referenced_script_exists_and_is_executable():
    _, commands = scripts()
    for command in commands:
        rel = re.search(r"\$\{CLAUDE_PLUGIN_ROOT\}/([^\"\s]+)", command).group(1)
        path = ROOT / rel
        assert path.is_file() and os.access(path, os.X_OK), rel


def test_manifests_name_the_plugin_and_its_root_source():
    plugin = json.loads((ROOT / ".claude-plugin" / "plugin.json").read_text())
    market = json.loads((ROOT / ".claude-plugin" / "marketplace.json").read_text())
    assert plugin["name"] == "lesson-recall" == market["plugins"][0]["name"]
    assert market["plugins"][0]["source"] == "./" and market["owner"]["name"]
    assert (ROOT / "skills" / "setup" / "SKILL.md").is_file()


def run_hook(name, payload, env):
    return subprocess.run(["bash", str(ROOT / "hooks" / name)], input=json.dumps(payload),
                          capture_output=True, text=True, timeout=30, env=env)


def test_prompt_hook_is_silent_in_shadow_mode_and_exits_0(tmp_path):
    env = dict(os.environ, CLAUDE_PLUGIN_ROOT=str(ROOT), RECALL_HOME=str(tmp_path / "store"),
               CLAUDE_CONFIG_DIR=str(tmp_path / "cfg"))
    out = run_hook("prompt.sh", dict(PROMPT, transcript_path=transcripts.write(str(tmp_path / "cfg"))), env)
    assert out.returncode == 0 and out.stdout == "" and out.stderr == ""
    assert (tmp_path / "store" / "log" / "read.jsonl").exists()


def test_hooks_are_inert_without_python(tmp_path):
    bare = tmp_path / "bin"
    bare.mkdir()
    for tool in ("bash", "dirname"):
        os.symlink(subprocess.check_output(["bash", "-c", f"command -v {tool}"], text=True).strip(), bare / tool)
    env = {"PATH": str(bare), "HOME": str(tmp_path), "CLAUDE_PLUGIN_ROOT": str(ROOT),
           "RECALL_HOME": str(tmp_path / "store")}
    for name in ("prompt.sh", "pass.sh"):
        out = subprocess.run([str(bare / "bash"), str(ROOT / "hooks" / name)], input=json.dumps(PROMPT),
                             capture_output=True, text=True, timeout=30, env=env)
        assert out.returncode == 0 and out.stdout == "" and out.stderr == "", name
    assert not (tmp_path / "store").exists()


def test_hooks_exit_at_once_inside_the_plugins_own_model_call(tmp_path):
    env = dict(os.environ, CLAUDE_PLUGIN_ROOT=str(ROOT), RECALL_HOME=str(tmp_path / "store"), RECALL_INNER="1")
    for name in ("prompt.sh", "pass.sh"):
        out = run_hook(name, PROMPT, env)
        assert out.returncode == 0 and out.stdout == ""
    assert not (tmp_path / "store").exists()
