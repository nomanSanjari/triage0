"""Tests for the bare-repo cache and disposable worktrees (local git, no network)."""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

import pytest

from triage0.storage.repo_cache import (
    RepoCache,
    RepoCacheError,
    cache_subpath,
    worktree,
)

VERSION_FILE = Path("include/curl/curlver.h")


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), "-c", "user.name=test", "-c", "user.email=test@example.com", *args],
        check=True, capture_output=True, text=True, timeout=30,
    )
    return result.stdout.strip()


@pytest.fixture
def temp_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Send tempfile.mkdtemp() into tmp_path so we can check for leftovers."""
    root = tmp_path / "tmp"
    root.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(root))
    return root


@pytest.fixture
def bare_repo(tmp_path: Path) -> Path:
    """A tiny curl-like bare repo with tags curl-8_13_0 and curl-8_14_0."""
    src = tmp_path / "src"
    (src / VERSION_FILE.parent).mkdir(parents=True)
    _git(src, "init", "-q")
    for version, tag in [("8.13.0", "curl-8_13_0"), ("8.14.0", "curl-8_14_0")]:
        (src / VERSION_FILE).write_text(f'#define LIBCURL_VERSION "{version}"\n')
        _git(src, "add", ".")
        _git(src, "commit", "-q", "-m", version)
        _git(src, "tag", "-a", tag, "-m", tag)
    bare = tmp_path / "bare.git"
    subprocess.run(["git", "clone", "-q", "--bare", str(src), str(bare)], check=True, timeout=30)
    return bare


def _registered_worktrees(bare: Path) -> list[str]:
    porcelain = _git(bare, "worktree", "list", "--porcelain")
    return [line for line in porcelain.splitlines() if line.startswith("worktree ")]


# --- worktree() --------------------------------------------------------------


def test_worktree_checks_out_the_exact_tag(bare_repo: Path, temp_root: Path) -> None:
    with worktree(bare_repo, "curl-8_13_0") as wt:
        assert '"8.13.0"' in (wt.path / VERSION_FILE).read_text()
        assert wt.tag == "curl-8_13_0"
        assert wt.tree_hash == _git(bare_repo, "rev-parse", "curl-8_13_0^{tree}")
    with worktree(bare_repo, "curl-8_14_0") as wt:
        assert '"8.14.0"' in (wt.path / VERSION_FILE).read_text()


def test_worktree_is_removed_after_normal_exit(bare_repo: Path, temp_root: Path) -> None:
    with worktree(bare_repo, "curl-8_13_0") as wt:
        assert wt.path.exists()
        assert len(_registered_worktrees(bare_repo)) == 2  # bare repo + ours
    assert not wt.path.exists()
    assert list(temp_root.iterdir()) == []
    assert len(_registered_worktrees(bare_repo)) == 1


def test_worktree_is_removed_after_an_error(bare_repo: Path, temp_root: Path) -> None:
    with pytest.raises(RuntimeError, match="boom"):
        with worktree(bare_repo, "curl-8_13_0") as wt:
            raise RuntimeError("boom")
    assert not wt.path.exists()
    assert list(temp_root.iterdir()) == []
    assert len(_registered_worktrees(bare_repo)) == 1


def test_worktree_refuses_unknown_tags(bare_repo: Path, temp_root: Path) -> None:
    for tag in ["curl-9_9_9", "--upload-pack=x", "main"]:
        with pytest.raises(RepoCacheError):
            with worktree(bare_repo, tag):
                pass
    assert list(temp_root.iterdir()) == []


def test_worktree_does_not_run_repo_hooks(bare_repo: Path, temp_root: Path, tmp_path: Path) -> None:
    marker = tmp_path / "hook-ran"
    hook = bare_repo / "hooks" / "post-checkout"
    hook.write_text(f"#!/bin/sh\ntouch {marker}\n")
    hook.chmod(0o755)
    with worktree(bare_repo, "curl-8_13_0"):
        pass
    assert not marker.exists()


# --- URLs and ensure_clone() -------------------------------------------------


def test_cache_subpath_for_a_normal_github_url() -> None:
    assert cache_subpath("https://github.com/curl/curl.git") == Path("github.com/curl/curl.git")
    assert cache_subpath("https://github.com/curl/curl") == Path("github.com/curl/curl.git")


@pytest.mark.parametrize(
    "url",
    [
        "http://github.com/curl/curl.git",
        "file:///etc",
        "ext::sh -c touch% /tmp/pwned",
        "git@github.com:curl/curl.git",
        "https://user:token@github.com/curl/curl.git",
        "https://github.com:8443/curl/curl.git",
        "https://github.com/curl/../../etc",
        "https://github.com/",
        "https://github.com/curl/curl.git?x=1",
        "--upload-pack=x",
    ],
)
def test_cache_subpath_rejects_unsafe_urls(url: str) -> None:
    with pytest.raises(RepoCacheError):
        cache_subpath(url)


def test_ensure_clone_rejects_bad_url_before_touching_disk(tmp_path: Path) -> None:
    cache = RepoCache(tmp_path / "cache")
    with pytest.raises(RepoCacheError):
        cache.ensure_clone("file:///etc")
    assert not (tmp_path / "cache").exists()


# --- ensure_clone() branches (git replaced by a recorder, no network) ---------


class _FakeGit:
    """Records run_git calls; optionally fails like an interrupted clone."""

    def __init__(self, fail_clone: bool = False) -> None:
        self.calls: list[list[str]] = []
        self.fail_clone = fail_clone

    def __call__(self, args: list[str], *, repo: Path | None = None, timeout: float = 0) -> None:
        self.calls.append(args)
        if args[0] == "clone":
            Path(args[-1]).mkdir(parents=True)  # git creates the folder early
            (Path(args[-1]) / "HEAD").write_text("partial")
            if self.fail_clone:
                raise subprocess.CalledProcessError(128, ["git", *args])


URL = "https://github.com/curl/curl.git"


def test_ensure_clone_clones_then_fetches(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeGit()
    monkeypatch.setattr("triage0.storage.repo_cache.run_git", fake)
    cache = RepoCache(tmp_path / "cache")

    first = cache.ensure_clone(URL)
    second = cache.ensure_clone(URL)

    assert first == second == tmp_path / "cache/github.com/curl/curl.git"
    assert fake.calls[0][:2] == ["clone", "--bare"]
    assert fake.calls[0][-3:] == ["--", URL, str(first)]
    assert fake.calls[1] == ["fetch", "--quiet", "origin", "+refs/tags/*:refs/tags/*"]
    assert len(fake.calls) == 2


def test_ensure_clone_removes_a_half_finished_clone(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("triage0.storage.repo_cache.run_git", _FakeGit(fail_clone=True))
    cache = RepoCache(tmp_path / "cache")
    with pytest.raises(subprocess.CalledProcessError):
        cache.ensure_clone(URL)
    # Otherwise the next call would see the folder and only "fetch" a broken repo.
    assert not (tmp_path / "cache/github.com/curl/curl.git").exists()
