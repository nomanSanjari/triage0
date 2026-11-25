"""Shared curl fixtures for the checker integration tests.

The clone is session-scoped: every checker test module in this run uses the
same bare repository, so a full integration run clones curl once.
"""

from __future__ import annotations

import tempfile
from collections.abc import Iterator
from pathlib import Path

import pytest

from triage0.sandbox.manager import SandboxManager, Session
from triage0.sandbox.providers.docker_local import DockerLocal
from triage0.storage.repo_cache import RepoCache

from ..sandbox.providers.test_docker_local import IMAGE

URL = "https://github.com/curl/curl.git"


@pytest.fixture(scope="session")
def cache(tmp_path_factory: pytest.TempPathFactory) -> RepoCache:
    return RepoCache(tmp_path_factory.mktemp("cache"))


@pytest.fixture
def temp_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Send worktrees into tmp_path so we can check none are left behind."""
    root = tmp_path / "tmp"
    root.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(root))
    return root


@pytest.fixture
def manager(cache: RepoCache, temp_root: Path) -> SandboxManager:
    return SandboxManager(cache, DockerLocal(), IMAGE)


@pytest.fixture
def session_curl_8_13_0(manager: SandboxManager) -> Iterator[Session]:
    with manager.open(URL, "curl-8_13_0") as session:
        yield session
