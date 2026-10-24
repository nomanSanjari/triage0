"""Bare-repo cache and disposable worktrees.

Each project is cloned once as a bare repo (history only, no files) and
kept on disk. For each check we create a throwaway worktree: a plain
folder with the files at one exact tag, removed again afterwards.

All network access (clone, fetch) happens here, on the host. The sandbox
container only ever sees the finished worktree, with networking off.

Warning for callers: a worktree is untrusted code and may contain symlinks
to host paths (e.g. ~/.ssh). Read its files inside the sandbox, not from
host-side Python.
"""

from __future__ import annotations

import logging
import re
import shutil
import subprocess
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit

from triage0.checkers.tools.git_tools import list_tags, run_git

logger = logging.getLogger(__name__)

CLONE_TIMEOUT_SECONDS = 600
WORKTREE_TIMEOUT_SECONDS = 120

_SAFE_SEGMENT = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")


class RepoCacheError(Exception):
    """A repo or tag can't be used (bad URL, unknown tag)."""


@dataclass(frozen=True)
class Worktree:
    """A checked-out copy of one tag.

    `tree_hash` fingerprints the exact file contents; evidence records it
    so anyone can confirm which code a check ran against.
    """

    path: Path
    tag: str
    tree_hash: str


def cache_subpath(url: str) -> Path:
    """Turn an https repo URL into a safe relative cache path.

    "https://github.com/curl/curl.git" -> github.com/curl/curl.git

    Only plain https URLs are accepted. URLs with credentials are refused
    so tokens never end up in folder names or logs, and every path segment
    must be a simple name, so ".." can't escape the cache directory.
    """
    parts = urlsplit(url)
    if parts.scheme != "https":
        raise RepoCacheError("only https:// repo URLs are allowed")
    if parts.username or parts.password:
        raise RepoCacheError("repo URLs must not contain credentials")
    if parts.port or parts.query or parts.fragment:
        raise RepoCacheError("repo URLs must not have a port, query or fragment")
    host = parts.hostname or ""
    path = parts.path.strip("/")
    if path.endswith(".git"):
        path = path[: -len(".git")]
    segments = [host, *path.split("/")]
    if len(segments) < 2 or not all(_SAFE_SEGMENT.fullmatch(s) for s in segments):
        raise RepoCacheError("repo URL has an unexpected host or path")
    if any(s in (".", "..") or s.endswith(".") for s in segments):
        raise RepoCacheError("repo URL has an unexpected host or path")
    return Path(*segments[:-1], segments[-1] + ".git")


class RepoCache:
    """Keeps one bare clone per repo under `cache_dir`.

    `cache_dir` is required on purpose: where the cache lives long-term
    hasn't been decided, so callers (and tests) choose it.
    """

    def __init__(self, cache_dir: Path) -> None:
        self.cache_dir = cache_dir

    def ensure_clone(self, url: str) -> Path:
        """Return the bare repo for `url`, cloning or fetching new tags first."""
        dest = self.cache_dir / cache_subpath(url)
        if dest.exists():
            run_git(
                ["fetch", "--quiet", "origin", "+refs/tags/*:refs/tags/*"],
                repo=dest,
                timeout=CLONE_TIMEOUT_SECONDS,
            )
            return dest

        dest.parent.mkdir(parents=True, exist_ok=True)
        try:
            run_git(
                ["clone", "--bare", "--quiet", "--", url, str(dest)],
                timeout=CLONE_TIMEOUT_SECONDS,
            )
        except BaseException:
            # A half-finished clone would look valid next time; remove it.
            shutil.rmtree(dest, ignore_errors=True)
            raise
        return dest


@contextmanager
def worktree(bare_repo: Path, tag: str) -> Iterator[Worktree]:
    """Check out `tag` into a fresh temp folder; always remove it on exit.

    `tag` must be one of the repo's own tags (from resolve_tag()). We still
    re-check it, and refer to it as refs/tags/<tag> so it can never be read
    as a git option or a branch name.
    """
    if tag not in list_tags(bare_repo):
        raise RepoCacheError(f"tag not found in repo: {tag}")
    ref = f"refs/tags/{tag}"
    tree_hash = run_git(["rev-parse", "--verify", f"{ref}^{{tree}}"], repo=bare_repo).stdout.strip()

    # Forget worktrees whose folders vanished (e.g. after a hard crash).
    run_git(["worktree", "prune"], repo=bare_repo)
    path = Path(tempfile.mkdtemp(prefix="triage0-wt-"))
    try:
        run_git(
            ["worktree", "add", "--detach", "--quiet", str(path), ref],
            repo=bare_repo,
            timeout=WORKTREE_TIMEOUT_SECONDS,
        )
        yield Worktree(path=path, tag=tag, tree_hash=tree_hash)
    finally:
        _remove_worktree(bare_repo, path)


def _remove_worktree(bare_repo: Path, path: Path) -> None:
    """Best-effort cleanup that never hides the error that got us here."""
    try:
        run_git(["worktree", "remove", "--force", str(path)], repo=bare_repo)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        logger.warning("git worktree remove failed for %s: %s", path, exc)
    shutil.rmtree(path, ignore_errors=True)
    try:
        run_git(["worktree", "prune"], repo=bare_repo)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        logger.warning("git worktree prune failed for %s: %s", bare_repo, exc)
    if path.exists():
        logger.warning("worktree folder could not be removed: %s", path)
