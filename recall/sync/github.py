"""The GitHub side of `sync init`: find or create the private card repository with the GitHub CLI.

Without `--remote`, sync uses `<your GitHub user>/claude-recall-cards`, created private when it
does not exist. Cards quote session text, so a repository that is not private is never pushed to.
"""
import json
import re
import shutil
import subprocess

TIMEOUT = 60


class Unavailable(RuntimeError):
    """The GitHub CLI is missing or not logged in."""


def _gh(*args, check=True):
    out = subprocess.run(["gh", *args], capture_output=True, text=True, timeout=TIMEOUT)
    if check and out.returncode:
        raise Unavailable(f"gh {args[0]}: {(out.stderr or out.stdout).strip()[-300:]}")
    return out


def user():
    """Login of the GitHub CLI's active account. Raises Unavailable when there is none."""
    if not shutil.which("gh"):
        raise Unavailable("the GitHub CLI (gh) is not installed")
    login = _gh("api", "user", "-q", ".login", check=False).stdout.strip()
    if not login:
        raise Unavailable("the GitHub CLI is not logged in (run: gh auth login)")
    return login


def slug_of(url):
    """'owner/name' for a github.com URL, else None."""
    hit = re.search(r"github\.com[:/]([^/\s]+/[^/\s]+?)(?:\.git)?/?$", url or "")
    return hit.group(1) if hit else None


def view(slug):
    """{"isPrivate", "url", "sshUrl"} of a repository, or None when it does not exist or cannot be seen."""
    out = _gh("repo", "view", slug, "--json", "isPrivate,url,sshUrl", check=False)
    try:
        return json.loads(out.stdout) if out.returncode == 0 else None
    except ValueError:
        return None


def is_private(url):
    """True or False from the GitHub CLI, None when it cannot tell (not GitHub, no gh, no access)."""
    slug = slug_of(url)
    if not slug or not shutil.which("gh"):
        return None
    try:
        info = view(slug)
    except (OSError, subprocess.SubprocessError):
        return None
    return None if info is None else bool(info.get("isPrivate"))


def ensure(name, say=print):
    """URL of `<user>/<name>`: reused when it exists and is private, created private when it does not.

    Raises Unavailable without a logged-in GitHub CLI and RuntimeError for a repository that is
    not private. The URL follows the CLI's git protocol setting (https unless set to ssh).
    """
    slug = f"{user()}/{name}"
    info = view(slug)
    if info is None:
        say(f"creating private repository {slug}")
        _gh("repo", "create", slug, "--private", "--description",
            "Lesson recall cards (append-only logs). Private: cards quote session text.")
        info = view(slug)
    if info is None:
        raise Unavailable(f"created {slug} but cannot read it back")
    if not info.get("isPrivate"):
        raise RuntimeError(f"{slug} is not private. Cards quote session text and go to private repositories "
                           "only. Make it private, or pass --remote with another private repository")
    protocol = _gh("config", "get", "git_protocol", check=False).stdout.strip()
    return (info["sshUrl"] if protocol == "ssh" else info["url"] + ".git"), slug
