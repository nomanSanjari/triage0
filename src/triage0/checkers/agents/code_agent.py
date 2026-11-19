"""Decide what a "not found" symbol really means, using git history (task 2.4c).

A checker that can't find a symbol only says ``CANT_CHECK`` with
``missing=SYMBOL_ABSENT``: symbols get renamed and code moves, so "not at this
version" isn't a contradiction. This investigator searches the history to tell
those cases apart. It is plain deterministic code with no LLM, so nothing it
concludes depends on a model's judgment (rule 1).

Only one outcome is a ``CONTRADICTED``: the name appears nowhere in any ref's
history *and* the claim places it in a file of this project. Everything else,
and any git failure or timeout, is ``CANT_CHECK``.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from triage0.checkers.tools.git_history import (
    last_change,
    last_change_argv,
    parents,
    path_in_tree,
    symbol_in_tree,
    tree_hash,
)
from triage0.core.evidence import (
    DeterministicProof,
    EvidenceResult,
    EvidenceStatus,
    MissingEvidence,
)
from triage0.core.models import AnyClaim, SymbolClaim

__all__ = ["investigate"]


def investigate(
    claim: AnyClaim, result: EvidenceResult, *, repo: Path, commit: str
) -> EvidenceResult:
    """Return the final result for a claim a checker reported as "not found".

    ``repo`` is the cached bare repo and ``commit`` the full id of the version
    the checker looked at. Anything this doesn't handle comes back unchanged:
    other claim kinds, claims that say a symbol is *absent*, and results
    without the not-found signal. So do git errors and timeouts, which leave
    the checker's own ``CANT_CHECK``.
    """

    if not (
        isinstance(claim, SymbolClaim)
        and claim.expected_present
        and result.missing is MissingEvidence.SYMBOL_ABSENT
    ):
        return result
    try:
        return _investigate_symbol(claim, repo, commit)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, ValueError):
        # ValueError: the symbol or path in the claim isn't one git may be
        # given (see git_history); a proof that fails validation also lands here.
        return result


def _investigate_symbol(claim: SymbolClaim, repo: Path, commit: str) -> EvidenceResult:
    symbol = claim.symbol
    if symbol_in_tree(repo, commit, symbol):
        return _cant_check(
            f"{symbol} appears in the code at this version, "
            "though the checker found no definition"
        )
    if last_change(repo, symbol) is None:
        return _never_existed(claim, repo, commit)

    # It exists somewhere in history. Where did it go?
    removed_in = last_change(repo, symbol, commit)
    if removed_in is None:
        return _cant_check(
            f"{symbol} is not in this version's history, but appears elsewhere "
            "in the repository's history"
        )
    if not _word_removed_in(repo, symbol, removed_in):
        # git's text search also hits names that merely contain the symbol,
        # so a hit that fails this check says nothing about the symbol.
        return _cant_check(
            f"{symbol} is not at this version and history does not show where it went"
        )
    return EvidenceResult(
        status=EvidenceStatus.CANT_CHECK,
        missing=MissingEvidence.SYMBOL_REMOVED,
        proof=_proof(repo, commit, last_change_argv(symbol, commit), f"{removed_in}\n"),
        summary=(
            f"{symbol} is not at this version but existed earlier: it was removed "
            f"in commit {removed_in} (renamed or deleted; git can't say which)"
        ),
    )


def _never_existed(claim: SymbolClaim, repo: Path, commit: str) -> EvidenceResult:
    """The name appears nowhere in history; contradict only if it's ours."""

    if not _placed_in_project(claim.file_hint, repo, commit):
        # Without a file of this project, the symbol may come from a library
        # or system header, which no search of this repository can rule out.
        return _cant_check(
            f"{claim.symbol} appears nowhere in this repository's history, but the "
            "claim doesn't place it in a file of this project, so it may come "
            "from outside it"
        )
    return EvidenceResult(
        status=EvidenceStatus.CONTRADICTED,
        # `git log -S` printed nothing and exited 0: no commit in any ref ever
        # added or removed this text (a failure would have raised instead).
        proof=_proof(repo, commit, last_change_argv(claim.symbol), ""),
        summary=(
            f"{claim.symbol} appears nowhere in this repository's git history, and "
            "the claim places it in a file of this project"
        ),
    )


def _placed_in_project(file_hint: str | None, repo: Path, commit: str) -> bool:
    if not file_hint:
        return False
    try:
        return path_in_tree(repo, commit, file_hint)
    except ValueError:  # not a safe path, so it places nothing
        return False


def _word_removed_in(repo: Path, symbol: str, commit: str) -> bool:
    """Whether the whole word is in the commit's parent and gone in the commit."""

    before = parents(repo, commit)
    return (
        len(before) == 1
        and symbol_in_tree(repo, before[0], symbol)
        and not symbol_in_tree(repo, commit, symbol)
    )


def _proof(repo: Path, commit: str, args: tuple[str, ...], stdout: str) -> DeterministicProof:
    # No host paths: argv records the git arguments, not where the repo lives.
    return DeterministicProof(
        tool="git",
        argv=("git", *args),
        exit_code=0,
        stdout=stdout,
        git_tree_hash=tree_hash(repo, commit),
        commit=commit,
    )


def _cant_check(summary: str) -> EvidenceResult:
    return EvidenceResult(status=EvidenceStatus.CANT_CHECK, summary=summary)
