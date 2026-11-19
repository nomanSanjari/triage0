"""A tiny git repo, built inside the test, shared by the checker tests."""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import pytest

from triage0.checkers.tools.git_tools import run_git


class Repo(NamedTuple):
    path: Path
    added: str  # commit that adds old_name
    renamed: str  # commit that replaces old_name with new_name


def _commit(repo: Path, message: str) -> str:
    run_git(["add", "-A"], repo=repo)
    run_git(
        ["-c", "user.name=Test", "-c", "user.email=test@example.com",
         "commit", "--quiet", "-m", message],
        repo=repo,
    )
    return run_git(["rev-parse", "HEAD"], repo=repo).stdout.strip()


@pytest.fixture
def repo(tmp_path: Path) -> Repo:
    run_git(["init", "--quiet"], repo=tmp_path)
    source = tmp_path / "lib" / "a.c"
    source.parent.mkdir()
    source.write_text("int old_name(void);\nint keep(void);\nint keep2(void);\n")
    added = _commit(tmp_path, "add old_name")
    source.write_text("int new_name(void);\nint keep(void);\nint keep2(void);\n")
    return Repo(tmp_path, added, _commit(tmp_path, "rename old_name"))
