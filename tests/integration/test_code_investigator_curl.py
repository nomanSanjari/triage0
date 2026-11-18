"""#11's "Done when", on real curl: the router and code investigator together
give the golden `renamed_symbol` claims their labels.

A stub checker finds nothing for every symbol, as a real one would for these
two; everything after that is the real code on real history. Needs internet:
TRIAGE0_NETWORK_TESTS=1 (clones curl, about 140 MB).
"""

from __future__ import annotations

import os
from functools import partial

import pytest

from tests.checker_stubs import StubChecker, StubSession
from tests.golden_fixtures import load_all
from triage0.checkers.agents.code_agent import investigate
from triage0.checkers.router import route
from triage0.checkers.tools.git_tools import run_git
from triage0.core.evidence import EvidenceResult, EvidenceStatus, MissingEvidence
from triage0.core.models import ClaimKind, VerificationPlan
from triage0.storage.repo_cache import RepoCache

pytestmark = pytest.mark.skipif(
    os.environ.get("TRIAGE0_NETWORK_TESTS") != "1",
    reason="set TRIAGE0_NETWORK_TESTS=1 (clones curl)",
)

CASE = next(case for case in load_all() if case.case_id == "renamed_symbol")


def test_renamed_symbol_is_rescued_and_made_up_symbol_contradicted_with_proof(
    repos: RepoCache,
) -> None:
    repo = repos.ensure_clone(CASE.repo_url)
    assert CASE.commit is not None
    plan = VerificationPlan(
        plan_id="golden",
        report_id=CASE.report.report_id,
        repo_url=CASE.repo_url,
        claims=tuple(expected.claim for expected in CASE.expected_claims),
    )
    finds_nothing = StubChecker(
        ClaimKind.SYMBOL,
        result=EvidenceResult(
            status=EvidenceStatus.CANT_CHECK, missing=MissingEvidence.SYMBOL_ABSENT
        ),
    )

    routed = route(
        plan,
        [finds_nothing],
        StubSession(),
        partial(investigate, repo=repo, commit=CASE.commit),
    )

    results = {claim.claim_id: claim.result for claim in routed.claims}
    labels = {e.claim.claim_id: e.expected_status for e in CASE.expected_claims}
    for claim_id, result in results.items():
        assert result is not None and result.status is labels[claim_id]

    made_up, renamed = results["c1"], results["c2"]  # Curl_cookie_parse_fast, Curl_dyn_add
    assert made_up is not None and made_up.proof is not None
    tree = run_git(["rev-parse", f"{CASE.commit}^{{tree}}"], repo=repo).stdout.strip()
    assert (made_up.proof.commit, made_up.proof.git_tree_hash) == (CASE.commit, tree)
    assert renamed is not None and renamed.missing is MissingEvidence.SYMBOL_REMOVED
    assert "255aac56f9" in (renamed.summary or "")  # the rename commit, from the fixture note
