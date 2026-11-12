"""Git plumbing for the card repository clone. Sync owns the clone; nothing else writes to it."""
import os
import re
import subprocess

from ..core import paths

DIR = os.path.join(paths.SYNC, "repo")


class GitError(RuntimeError):
    pass


def _clean(text):
    """Error text without any credential a remote URL may carry."""
    return re.sub(r"(://)[^/@\s]+@", r"\1", (text or "").strip())[-300:]


def git(*args, cwd=None, check=True, timeout=180):
    out = subprocess.run(["git", *args], cwd=cwd or DIR, capture_output=True, text=True, timeout=timeout,
                         env=dict(os.environ, GIT_TERMINAL_PROMPT="0"))
    if check and out.returncode:
        raise GitError(f"git {args[0]}: {_clean(out.stderr or out.stdout)}")
    return out


IDENTITY = (("user.name", "lesson-recall"), ("user.email", "lesson-recall@localhost"))


def clone(url, branch, gh_auth=False):
    """Create the clone, or repoint an existing one.

    Commits use the global git identity, else a neutral one. With `gh_auth` an https GitHub
    remote authenticates through the GitHub CLI, set for this clone only.
    """
    os.makedirs(DIR, exist_ok=True)
    if not os.path.isdir(os.path.join(DIR, ".git")):
        git("init", "-q")
        git("symbolic-ref", "HEAD", f"refs/heads/{branch}")
    if git("remote", "get-url", "origin", check=False).returncode:
        git("remote", "add", "origin", url)
    else:
        git("remote", "set-url", "origin", url)
    for key, fallback in IDENTITY:
        if not git("config", "--global", key, check=False).stdout.strip():
            git("config", key, fallback)
    if gh_auth and url.startswith("https://github.com/"):
        helper = "credential.https://github.com.helper"
        git("config", "--unset-all", helper, check=False)
        git("config", "--add", helper, "")                       # drop helpers inherited from global config
        git("config", "--add", helper, "!gh auth git-credential")


def refresh(branch):
    """Move the clone to the remote tip, dropping local leftovers. False: the branch does not exist yet."""
    out = git("fetch", "-q", "origin", f"+refs/heads/{branch}:refs/remotes/origin/{branch}", check=False)
    if out.returncode:
        if "couldn't find remote ref" in out.stderr:
            return False
        raise GitError("git fetch: " + _clean(out.stderr))
    git("checkout", "-q", "-f", "-B", branch, f"origin/{branch}")
    git("clean", "-fdq")
    return True


def publish(branch, message):
    """Commit what changed and push. Returns 'clean', 'pushed' or 'rejected' (lost a race)."""
    git("add", "-A")
    if git("diff", "--cached", "--quiet", check=False).returncode:
        git("commit", "-q", "-m", message)
    head = git("rev-parse", "-q", "--verify", "HEAD", check=False).stdout.strip()
    tip = git("rev-parse", "-q", "--verify", f"refs/remotes/origin/{branch}", check=False).stdout.strip()
    if not head or head == tip:
        return "clean"
    out = git("push", "-q", "origin", f"HEAD:refs/heads/{branch}", check=False)
    if out.returncode == 0:
        return "pushed"
    if re.search(r"rejected|fetch first|non-fast-forward", out.stderr):
        return "rejected"
    raise GitError("git push: " + _clean(out.stderr))


def mirror(urls, branch):
    """Push-only copies. A mirror that fails is reported, never fatal."""
    result = {}
    for url in urls:
        out = git("push", "-q", url, f"HEAD:refs/heads/{branch}", check=False)
        result[_clean(url)] = "ok" if out.returncode == 0 else _clean(out.stderr)
    return result
