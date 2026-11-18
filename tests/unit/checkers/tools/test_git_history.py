"""Tests for the git history helpers, on a tiny repo built inside the test."""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.unit.checkers.conftest import Repo
from triage0.checkers.tools import git_history
from triage0.checkers.tools.git_history import (
    blame_lines,
    last_change,
    line_history,
    parents,
    path_in_tree,
    symbol_in_tree,
    tree_hash,
)
from triage0.checkers.tools.git_tools import run_git

SHA = "0" * 40  # well-formed; only used where git must never be reached


def test_symbol_in_tree_matches_whole_words_at_the_given_commit(repo: Repo) -> None:
    assert symbol_in_tree(repo.path, repo.added, "old_name") is True
    assert symbol_in_tree(repo.path, repo.renamed, "old_name") is False
    assert symbol_in_tree(repo.path, repo.renamed, "new") is False  # "new_name" isn't "new"


def test_last_change_finds_the_commit_that_removed_a_symbol(repo: Repo) -> None:
    assert last_change(repo.path, "old_name", repo.renamed) == repo.renamed
    assert last_change(repo.path, "old_name", repo.added) == repo.added
    assert last_change(repo.path, "old_name") == repo.renamed  # every ref


def test_last_change_stops_at_the_given_commit_and_knows_never_existed(repo: Repo) -> None:
    assert last_change(repo.path, "new_name", repo.added) is None  # not written yet
    assert last_change(repo.path, "new_name", repo.renamed) == repo.renamed
    assert last_change(repo.path, "never_existed") is None
    assert last_change(repo.path, "never_existed", repo.renamed) is None


def test_path_in_tree_counts_files_only(repo: Repo) -> None:
    assert path_in_tree(repo.path, repo.renamed, "lib/a.c") is True
    assert path_in_tree(repo.path, repo.renamed, "lib") is False  # a directory
    assert path_in_tree(repo.path, repo.renamed, "lib/missing.c") is False


def test_tree_hash_is_the_commits_tree(repo: Repo) -> None:
    first_line = run_git(["cat-file", "-p", repo.added], repo=repo.path).stdout.splitlines()[0]
    assert first_line == f"tree {tree_hash(repo.path, repo.added)}"
    assert tree_hash(repo.path, repo.added) != tree_hash(repo.path, repo.renamed)


def test_parents_names_the_commit_before(repo: Repo) -> None:
    assert parents(repo.path, repo.renamed) == (repo.added,)
    assert parents(repo.path, repo.added) == ()


def test_line_history_lists_commits_that_changed_the_lines(repo: Repo) -> None:
    # Only hashes come back (no patch text), newest first.
    assert line_history(repo.path, repo.renamed, "lib/a.c", 1, 1) == [repo.renamed, repo.added]
    assert line_history(repo.path, repo.renamed, "lib/a.c", 2, 2) == [repo.added]


def test_blame_lines_names_the_commit_that_wrote_each_line(repo: Repo) -> None:
    assert blame_lines(repo.path, repo.renamed, "lib/a.c", 1, 3) == {
        1: repo.renamed,
        2: repo.added,
        3: repo.added,
    }


@pytest.fixture
def no_git(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(*args: object, **kwargs: object) -> None:
        raise AssertionError("git must not be called for a rejected value")

    monkeypatch.setattr(git_history, "run_git", fail)


@pytest.mark.parametrize(
    "symbol",
    ["", "-Sx", "--all", "a b", "x;rm -rf /", "a'b", "a\n", "1abc"],
)
def test_bad_symbol_never_reaches_git(no_git: None, symbol: str) -> None:
    with pytest.raises(ValueError, match="C identifier"):
        symbol_in_tree(Path("."), SHA, symbol)
    with pytest.raises(ValueError, match="C identifier"):
        last_change(Path("."), symbol)


@pytest.mark.parametrize("path", ["", "../x", "a/../b", "/etc/passwd", "-lib", "a\nb"])
def test_bad_path_never_reaches_git(no_git: None, path: str) -> None:
    with pytest.raises(ValueError, match="safe repository path"):
        path_in_tree(Path("."), SHA, path)
    with pytest.raises(ValueError, match="safe repository path"):
        line_history(Path("."), SHA, path, 1, 2)
    with pytest.raises(ValueError, match="safe repository path"):
        blame_lines(Path("."), SHA, path, 1, 2)


@pytest.mark.parametrize("commit", ["HEAD", "--all", "main", "abc123", SHA.replace("0", "A")])
def test_anything_but_a_full_commit_id_never_reaches_git(no_git: None, commit: str) -> None:
    with pytest.raises(ValueError, match="full commit id"):
        symbol_in_tree(Path("."), commit, "x")
    with pytest.raises(ValueError, match="full commit id"):
        last_change(Path("."), "x", commit)
    with pytest.raises(ValueError, match="full commit id"):
        path_in_tree(Path("."), commit, "a.c")
    with pytest.raises(ValueError, match="full commit id"):
        tree_hash(Path("."), commit)


@pytest.mark.parametrize("start, end", [(0, 1), (5, 4), (1, 201)])
def test_bad_line_range_never_reaches_git(no_git: None, start: int, end: int) -> None:
    with pytest.raises(ValueError, match="line range"):
        line_history(Path("."), SHA, "a.c", start, end)
    with pytest.raises(ValueError, match="line range"):
        blame_lines(Path("."), SHA, "a.c", start, end)
