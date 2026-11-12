"""Names that must mean the same thing on every device: the device itself, and a project.

A project key is the working directory with every non-alphanumeric character turned into a
dash, so it starts with the home directory: `-home-alice-projects-x` on one machine is
`-Users-alice-projects-x` on another. `portable` drops that prefix, which lets a card written
on one device match the same project on the next.
"""
import json
import os
import re
import socket

from . import paths

DEVICES = os.path.join(paths.REMOTE, "devices.json")   # {device: home key}, written by sync
_HOME_RX = re.compile(r"^-(?:home|Users)-[^-]+")       # fallback for a device not seen yet


def key(cwd):
    return re.sub(r"[^A-Za-z0-9]", "-", cwd or "")


def home():
    return key(os.path.expanduser("~"))


def device():
    """This machine's name in the shared store: the configured one, else the short hostname."""
    name = paths.config()["device"] or socket.gethostname().split(".")[0]
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-") or "device"


def homes():
    """Home keys of every known device, longest first."""
    known = {home()}
    try:
        with open(DEVICES, encoding="utf-8") as f:
            known.update(v for v in json.load(f).values() if v)
    except (FileNotFoundError, ValueError):
        pass
    return sorted(known, key=len, reverse=True)


def portable(project, known=None):
    """Project key without the device's home prefix. The home directory itself becomes ''."""
    for h in known if known is not None else homes():
        if project == h or project.startswith(h + "-"):
            return project[len(h):].lstrip("-")
    hit = _HOME_RX.match(project)
    return project[hit.end():].lstrip("-") if hit else project
