"""Read and change the store's config.json from the command line, with type and regex checks.

  config show                  every value, defaults included
  config set KEY VALUE         VALUE is parsed as JSON when it parses, else taken as a string
  config add KEY VALUE         append to a list value (correction_cues, extra_roots, sync_mirrors)
  config remove KEY VALUE      drop one entry from a list value
"""
import json
import re

from ..core import paths


class ConfigError(SystemExit):
    pass


def _parse(key, raw):
    if key not in paths.DEFAULTS:
        raise ConfigError(f"unknown config key {key!r}; see `cli.py config show`")
    default = paths.DEFAULTS[key]
    try:
        value = json.loads(raw)
    except ValueError:
        value = raw
    if isinstance(default, float) and isinstance(value, int) and not isinstance(value, bool):
        value = float(value)
    if type(value) is not type(default):
        raise ConfigError(f"{key} takes a {type(default).__name__}, got {type(value).__name__}: {raw!r}")
    return value


def _check_cues(cues):
    for cue in cues:
        try:
            re.compile(cue)
        except re.error as e:
            raise ConfigError(f"correction cue {cue!r} is not a valid regular expression: {e}") from e


def change(action, key, raw=None):
    """Apply one change and return the new config. Raises ConfigError on a bad key, type or regex."""
    cfg = paths.config()
    if action == "set":
        value = _parse(key, raw)
    else:
        if not isinstance(paths.DEFAULTS.get(key), list):
            raise ConfigError(f"{key} is not a list; use `config set`")
        current = list(cfg[key])
        if action == "add":
            value = current + ([raw] if raw not in current else [])
        else:
            if raw not in current:
                raise ConfigError(f"{raw!r} is not in {key}")
            value = [v for v in current if v != raw]
    if key == "correction_cues":
        _check_cues(value)
    return paths.set_config(**{key: value})


def cmd_config(args):
    if args.action == "show":
        print(json.dumps(paths.config(), ensure_ascii=False, indent=2))
        return
    if args.key is None or args.value is None:
        raise ConfigError(f"config {args.action} needs KEY and VALUE")
    new = change(args.action, args.key, args.value)
    print(json.dumps({args.key: new[args.key]}, ensure_ascii=False))
