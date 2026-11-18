"""Tests for the code investigator, on the tiny shared repo.

In that repo ``old_name`` is added in one commit and renamed to ``new_name`` in
the next; ``lib/a.c`` is the only file.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from tests.unit.checkers.conftest import Repo
from triage0.checkers.agents.code_agent import investigate
from triage0.checkers.tools import git_history
from triage0.checkers.tools.git_history import tree_hash
from triage0.checkers.tools.git_tools import run_git
from triage0.core.evidence import EvidenceResult, EvidenceStatus, MissingEvidence
from triage0.core.models import AnyClaim, FileLineClaim, SymbolClaim

SHA = "0" * 40  # well-formed; only used where git must never be reached


def _not_found(
    symbol: str,
    *,
    file_hint: str | None = "lib/a.c",
    expected_present: bool = True,
    missing: MissingEvidence | None = MissingEvidence.SYMBOL_ABSENT,
) -> tuple[SymbolClaim, EvidenceResult]:
    """A symbol claim and the result a checker gives when it finds nothing."""

    claim = SymbolClaim(
        claim_id="c1",
        statement=f"{symbol} exists",
        symbol=symbol,
        file_hint=file_hint,
        expected_present=expected_present,
    )
    result = EvidenceResult(
        status=EvidenceStatus.CANT_CHECK, missing=missing, summary="checker: not found"
    )
    return claim, result


def test_removed_symbol_is_rescued_with_the_removing_commit(repo: Repo) -> None:
    claim, result = _not_found("old_name")

    final = investigate(claim, result, repo=repo.path, commit=repo.renamed)

    assert final.status is EvidenceStatus.CANT_CHECK  # never CONTRADICTED
    assert final.missing is MissingEvidence.SYMBOL_REMOVED
    assert repo.renamed in (final.summary or "")
    assert final.proof is not None and final.proof.stdout.strip() == repo.renamed


def test_symbol_in_no_history_is_contradicted_with_a_proof_git_reproduces(repo: Repo) -> None:
    claim, result = _not_found("made_up_function")

    final = investigate(claim, result, repo=repo.path, commit=repo.renamed)

    assert final.status is EvidenceStatus.CONTRADICTED
    proof = final.proof
    assert proof is not None
    assert proof.git_tree_hash == tree_hash(repo.path, repo.renamed)
    assert proof.commit == repo.renamed
    assert str(repo.path) not in " ".join(proof.argv)
    # The proof is the command that was run: running it again gives the same.
    again = run_git(list(proof.argv[1:]), repo=repo.path)
    assert proof.argv[0] == "git"
    assert (again.returncode, again.stdout) == (proof.exit_code, proof.stdout)


@pytest.mark.parametrize("file_hint", [None, "lib/missing.c", "lib", "../etc/passwd"])
def test_symbol_not_placed_in_this_project_is_not_contradicted(
    repo: Repo, file_hint: str | None
) -> None:
    # It could come from an external library or system header.
    claim, result = _not_found("made_up_function", file_hint=file_hint)

    final = investigate(claim, result, repo=repo.path, commit=repo.renamed)

    assert final.status is EvidenceStatus.CANT_CHECK
    assert final.missing is None and final.proof is None
    assert "from outside it" in (final.summary or "")


def test_symbol_git_still_finds_is_not_contradicted(repo: Repo) -> None:
    claim, result = _not_found("keep")  # the checker was wrong or looked for a definition

    final = investigate(claim, result, repo=repo.path, commit=repo.renamed)

    assert final.status is EvidenceStatus.CANT_CHECK and final.missing is None


def test_symbol_that_only_appears_in_later_history_is_not_contradicted(repo: Repo) -> None:
    claim, result = _not_found("new_name")  # written in the second commit

    final = investigate(claim, result, repo=repo.path, commit=repo.added)

    assert final.status is EvidenceStatus.CANT_CHECK and final.missing is None
    assert "elsewhere" in (final.summary or "")


def test_text_hit_that_was_never_a_whole_word_names_no_commit(repo: Repo) -> None:
    # git's text search hits "old_name" for "old_na", but it was never a word.
    claim, result = _not_found("old_na")

    final = investigate(claim, result, repo=repo.path, commit=repo.renamed)

    assert final.status is EvidenceStatus.CANT_CHECK
    assert final.missing is None and final.proof is None
    assert "does not show where" in (final.summary or "")


@pytest.fixture
def no_git(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(*args: object, **kwargs: object) -> None:
        raise AssertionError("git must not be called")

    monkeypatch.setattr(git_history, "run_git", fail)


def test_claims_it_does_not_handle_come_back_unchanged(no_git: None) -> None:
    _, found = _not_found("x")
    unhandled: list[tuple[AnyClaim, EvidenceResult]] = [
        (FileLineClaim(claim_id="c1", statement="s", path="lib/a.c", line_start=1), found),
        _not_found("x", expected_present=False),
        _not_found("x", missing=None),  # a checker that gave no not-found signal
        _not_found("x", missing=MissingEvidence.SYMBOL_REMOVED),  # already investigated
    ]

    for claim, result in unhandled:
        assert investigate(claim, result, repo=Path("."), commit=SHA) is result


@pytest.mark.parametrize(
    "error",
    [subprocess.CalledProcessError(128, "git"), subprocess.TimeoutExpired("git", 120)],
    ids=["git-fails", "git-times-out"],
)
def test_git_failure_leaves_the_checkers_cant_check(
    monkeypatch: pytest.MonkeyPatch, error: Exception
) -> None:
    def fail(*args: object, **kwargs: object) -> None:
        raise error

    monkeypatch.setattr(git_history, "run_git", fail)
    claim, result = _not_found("x")

    assert investigate(claim, result, repo=Path("."), commit=SHA) is result


def test_symbol_git_may_not_be_given_leaves_the_checkers_cant_check(no_git: None) -> None:
    claim, result = _not_found("foo(); rm -rf /")

    assert investigate(claim, result, repo=Path("."), commit=SHA) is result
