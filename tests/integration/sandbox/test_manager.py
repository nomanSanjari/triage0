"""End-to-end: version text -> tag -> worktree -> offline container -> output -> cleanup.

Needs Docker and the image:  docker build -t triage0-sandbox:dev - < Dockerfile.sandbox
The real-curl test also needs internet:  TRIAGE0_NETWORK_TESTS=1
"""

from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path

import pytest

from triage0.checkers.tools.git_tools import TagResolutionStatus
from triage0.sandbox.manager import SandboxManager, TagNotResolved, project_from_url
from triage0.sandbox.providers.docker_local import (
    DockerLocal,
    SandboxLimits,
    list_sandbox_containers,
)
from triage0.storage.repo_cache import RepoCache

from .providers.test_docker_local import IMAGE, _image_available

pytestmark = pytest.mark.skipif(not _image_available(), reason=f"docker or {IMAGE} not available")

VERSION_FILE = "include/curl/curlver.h"


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), "-c", "user.name=test", "-c", "user.email=test@example.com", *args],
        check=True, capture_output=True, text=True, timeout=30,
    )
    return result.stdout.strip()


@pytest.fixture
def temp_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Send worktrees into tmp_path so we can check none are left behind."""
    root = tmp_path / "tmp"
    root.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(root))
    return root


@pytest.fixture
def bare_repo(tmp_path: Path) -> Path:
    src = tmp_path / "src"
    (src / "include/curl").mkdir(parents=True)
    _git(src, "init", "-q")
    for version, tag in [("8.13.0", "curl-8_13_0"), ("8.14.0", "curl-8_14_0")]:
        (src / VERSION_FILE).write_text(f'#define LIBCURL_VERSION "{version}"\n')
        _git(src, "add", ".")
        _git(src, "commit", "-q", "-m", version)
        _git(src, "tag", "-a", tag, "-m", tag)
    bare = tmp_path / "bare.git"
    subprocess.run(["git", "clone", "-q", "--bare", str(src), str(bare)], check=True, timeout=30)
    return bare


@pytest.fixture
def manager(tmp_path: Path) -> SandboxManager:
    limits = SandboxLimits(memory="256m", lifetime_seconds=120, exec_timeout_seconds=30)
    return SandboxManager(RepoCache(tmp_path / "cache"), DockerLocal(limits), IMAGE)


def _assert_nothing_left(bare_repo: Path, temp_root: Path, container_id: str | None = None) -> None:
    if container_id is not None:
        assert container_id not in list_sandbox_containers()
    assert list(temp_root.iterdir()) == []
    worktrees = _git(bare_repo, "worktree", "list", "--porcelain").splitlines()
    assert len([line for line in worktrees if line.startswith("worktree ")]) == 1


def test_session_runs_against_the_exact_version(
    manager: SandboxManager, bare_repo: Path, temp_root: Path
) -> None:
    with manager.session_for_repo(bare_repo, "curl", "curl 8.13.0") as session:
        result = session.run(["cat", f"/src/{VERSION_FILE}"])
        net = session.run(["ls", "/sys/class/net"])
        container_id = session.container_id
        assert container_id in list_sandbox_containers()
    assert session.tag == "curl-8_13_0"
    assert session.tree_hash == _git(bare_repo, "rev-parse", "curl-8_13_0^{tree}")
    assert result.exit_code == 0 and '"8.13.0"' in result.stdout
    assert net.stdout.split() == ["lo"]
    _assert_nothing_left(bare_repo, temp_root, container_id)


def test_cleanup_happens_after_an_error(
    manager: SandboxManager, bare_repo: Path, temp_root: Path
) -> None:
    container_id = None
    with pytest.raises(RuntimeError, match="checker crashed"):
        with manager.session_for_repo(bare_repo, "curl", "curl 8.14.0") as session:
            container_id = session.container_id
            raise RuntimeError("checker crashed")
    assert container_id is not None
    _assert_nothing_left(bare_repo, temp_root, container_id)


def test_cleanup_happens_after_a_timeout(
    manager: SandboxManager, bare_repo: Path, temp_root: Path
) -> None:
    with manager.session_for_repo(bare_repo, "curl", "curl 8.13.0") as session:
        result = session.run(["sleep", "60"], timeout=2)
    assert result.timed_out
    _assert_nothing_left(bare_repo, temp_root, session.container_id)


@pytest.mark.parametrize(
    "text, status",
    [
        ("curl 9.9.9", TagResolutionStatus.NO_MATCHING_TAG),
        ("curl 8.13.0-DEV", TagResolutionStatus.NO_VERSION),
        ("curl 8.13.0 or curl 8.14.0", TagResolutionStatus.AMBIGUOUS_VERSION),
    ],
)
def test_unresolvable_versions_start_nothing(
    manager: SandboxManager, bare_repo: Path, temp_root: Path, text: str, status: TagResolutionStatus
) -> None:
    before = set(list_sandbox_containers())
    with pytest.raises(TagNotResolved) as excinfo:
        with manager.session_for_repo(bare_repo, "curl", text):
            pass
    assert excinfo.value.resolution.status is status
    assert set(list_sandbox_containers()) == before
    _assert_nothing_left(bare_repo, temp_root)


def test_project_from_url() -> None:
    assert project_from_url("https://github.com/curl/curl.git") == "curl"
    assert project_from_url("https://github.com/nodejs/node") == "node"


@pytest.mark.skipif(
    os.environ.get("TRIAGE0_NETWORK_TESTS") != "1", reason="set TRIAGE0_NETWORK_TESTS=1 (clones curl)"
)
def test_definition_of_done_real_curl(tmp_path: Path, temp_root: Path) -> None:
    """Issue #5 acceptance: "curl 8.13.0" -> offline container with that code -> output -> no leftovers."""
    url = "https://github.com/curl/curl.git"
    cache = RepoCache(tmp_path / "cache")
    manager = SandboxManager(cache, DockerLocal(), IMAGE)

    # Exercise the fetch path too: clone, delete the tag locally, and let
    # session() (which calls ensure_clone again) fetch it back from GitHub.
    bare = cache.ensure_clone(url)
    _git(bare, "tag", "--delete", "curl-8_13_0")
    assert "curl-8_13_0" not in _git(bare, "tag", "--list").split()

    with manager.session(url, "curl 8.13.0") as session:
        # In git, LIBCURL_VERSION always reads "x.y.z-DEV" (curl's release
        # script rewrites it only in tarballs), so check the numeric version.
        version = session.run(["grep", "-m1", "^#define LIBCURL_VERSION_NUM ", f"/src/{VERSION_FILE}"])
        net = session.run(["ls", "/sys/class/net"])
    assert session.tag == "curl-8_13_0"
    assert "0x080d00" in version.stdout
    assert net.stdout.split() == ["lo"]
    assert session.container_id not in list_sandbox_containers()
    assert list(temp_root.iterdir()) == []
