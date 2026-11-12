"""`sync init`: choose the private card repository, check it, clone it, and run the first sync."""
from ..core import keys, paths
from . import github, repo
from .run import SyncError, sync

LOCAL_ONLY = ("Card sync is optional and MemBrain works fully without it. To back the cards up to a "
              "private GitHub repository, install the GitHub CLI and run `gh auth login`, or pass --remote with "
              "the URL of another private repository.")


def _resolve(remote, assume_private, say):
    """(url, use gh for auth): the given remote after a privacy check, else the user's card repository."""
    if remote:
        private = github.is_private(remote)
        if private is False:
            raise SyncError("that repository is public. Cards quote session text and go to private repositories only")
        if private is None and not assume_private:
            raise SyncError("cannot confirm the repository is private (needs the GitHub CLI, logged in, for a "
                            "github.com URL). After checking it yourself, pass --assume-private")
        return remote, private is True
    try:
        url, slug = github.ensure(paths.config()["sync_repo"], say)
    except github.Unavailable as exc:
        raise SyncError(f"{exc}. {LOCAL_ONLY}") from exc
    except RuntimeError as exc:
        raise SyncError(str(exc)) from exc
    say(f"card repository: {slug} (private)")
    return url, True


def setup(remote=None, device=None, adopt=False, assume_private=False, say=print):
    """Join the shared store: resolve and check the private repository, clone it, sync once."""
    cfg = paths.config()
    url, gh_auth = _resolve(remote, assume_private, say)
    paths.set_config(sync_remote=url, device=device or keys.device())
    repo.clone(url, cfg["sync_branch"], gh_auth)
    say(f"device {keys.device()}, branch {cfg['sync_branch']}")
    return sync(say, adopt)
