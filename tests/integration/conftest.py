"""Fixtures shared by the integration tests."""

from __future__ import annotations

import shutil
from collections.abc import Iterator

import pytest

from triage0.storage.repo_cache import RepoCache


@pytest.fixture(scope="session")
def repos(tmp_path_factory: pytest.TempPathFactory) -> Iterator[RepoCache]:
    # One clone for every test that needs a real repo, deleted afterwards:
    # pytest would otherwise keep the last few runs' 140 MB clones around.
    cache_dir = tmp_path_factory.mktemp("repo-cache")
    try:
        yield RepoCache(cache_dir)
    finally:
        shutil.rmtree(cache_dir)
