"""Check every golden fixture's evidence against the real source.

The unit tests prove the fixtures are well-formed; only this test proves the
labels weren't written from memory. Needs internet: TRIAGE0_NETWORK_TESTS=1
(clones curl, about 140 MB).
"""

from __future__ import annotations

import os
import shutil
from collections.abc import Iterator
from pathlib import Path
from subprocess import CalledProcessError

import pytest

from tests.golden_fixtures import AbsentEvidence, GoldenCase, LineEvidence, load_all
from triage0.checkers.tools.git_tools import run_git
from triage0.storage.repo_cache import RepoCache

pytestmark = pytest.mark.skipif(
    os.environ.get("TRIAGE0_NETWORK_TESTS") != "1",
    reason="set TRIAGE0_NETWORK_TESTS=1 (clones curl)",
)

# `git log -S` reads every commit; about 10 s for curl on a laptop, so the
# default git timeout is too tight on a slower machine.
HISTORY_SEARCH_TIMEOUT_SECONDS = 300

CASES_WITH_A_VERSION = [case for case in load_all() if case.tag is not None]


@pytest.fixture(scope="module")
def repos(tmp_path_factory: pytest.TempPathFactory) -> Iterator[RepoCache]:
    # One clone for all cases, deleted afterwards: pytest would otherwise keep
    # the last few runs' 140 MB clones around.
    cache_dir = tmp_path_factory.mktemp("repo-cache")
    try:
        yield RepoCache(cache_dir)
    finally:
        shutil.rmtree(cache_dir)


def _line_problem(repo: Path, case_commit: str, ev: LineEvidence) -> str | None:
    commit = ev.commit or case_commit
    lines = run_git(["show", f"{commit}:{ev.path}"], repo=repo).stdout.splitlines()
    actual = lines[ev.line - 1] if ev.line <= len(lines) else "<past end of file>"
    if actual == ev.text:
        return None
    return f"{ev.path}:{ev.line} at {commit[:10]}: expected {ev.text!r}, found {actual!r}"


def _absent_problem(repo: Path, commit: str, ev: AbsentEvidence) -> str | None:
    if ev.scope == "tree":
        try:
            found = run_git(
                ["grep", "-l", "-w", "-F", "-e", ev.pattern, commit], repo=repo
            ).stdout
        except CalledProcessError as error:
            if error.returncode != 1:  # 1 means "no match"; anything else is a real error
                raise
            return None
    else:
        found = run_git(
            ["log", "--format=%h", f"-S{ev.pattern}", commit],
            repo=repo,
            timeout=HISTORY_SEARCH_TIMEOUT_SECONDS,
        ).stdout
    if not found:
        return None
    return f"{ev.pattern!r} is not absent from {ev.scope}: {found.split()[:5]}"


@pytest.mark.parametrize("case", CASES_WITH_A_VERSION, ids=lambda case: case.case_id)
def test_evidence_matches_source(case: GoldenCase, repos: RepoCache) -> None:
    repo = repos.ensure_clone(case.repo_url)
    tag_commit = run_git(["rev-parse", f"{case.tag}^{{commit}}"], repo=repo).stdout.strip()
    assert case.commit is not None and tag_commit == case.commit, (
        f"{case.tag} points to {tag_commit}, fixture says {case.commit}"
    )

    problems = []
    for expected in case.expected_claims:
        for ev in expected.evidence:
            if isinstance(ev, LineEvidence):
                problem = _line_problem(repo, case.commit, ev)
            else:
                problem = _absent_problem(repo, case.commit, ev)
            if problem:
                problems.append(f"{expected.claim.claim_id}: {problem}")
    assert not problems, "\n".join(problems)
