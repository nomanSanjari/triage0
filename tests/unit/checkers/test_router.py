"""Tests for the claim router, with stub checkers that follow BaseClaimChecker.

akib's real checkers (#10) don't exist yet, so these stubs stand in for them.
"""

from __future__ import annotations

from tests.checker_stubs import StubChecker, StubSession
from triage0.checkers.router import route
from triage0.core.evidence import EvidenceResult, EvidenceStatus, MissingEvidence
from triage0.core.models import (
    AnyClaim,
    ClaimKind,
    FileLineClaim,
    PoCClaim,
    SymbolClaim,
    VerificationPlan,
)


def _not_found(summary: str = "checker: not found") -> EvidenceResult:
    return EvidenceResult(
        status=EvidenceStatus.CANT_CHECK,
        missing=MissingEvidence.SYMBOL_ABSENT,
        summary=summary,
    )


def _plan(*claims: AnyClaim) -> VerificationPlan:
    return VerificationPlan(
        plan_id="p1", report_id="r1", repo_url="https://example.org/x", claims=claims
    )


SYMBOL = SymbolClaim(claim_id="c1", statement="s", symbol="f")
FILE_LINE = FileLineClaim(claim_id="c2", statement="s", path="a.c", line_start=1)
POC = PoCClaim(claim_id="c3", statement="s", steps=("run",), expected_outcome="crash")


class Investigator:
    """Records what it was handed and answers with a scripted result."""

    def __init__(self, result: EvidenceResult | None = None, error: Exception | None = None) -> None:
        self.result, self.error = result, error
        self.calls: list[tuple[str, EvidenceResult]] = []

    def __call__(self, claim: AnyClaim, result: EvidenceResult) -> EvidenceResult:
        self.calls.append((claim.claim_id, result))
        if self.error:
            raise self.error
        return self.result or result


def test_each_claim_goes_to_the_checker_that_supports_it() -> None:
    symbol_checker = StubChecker(ClaimKind.SYMBOL)
    file_checker = StubChecker(ClaimKind.FILE_LINE)
    session = StubSession()
    plan = _plan(FILE_LINE, SYMBOL)

    routed = route(plan, [symbol_checker, file_checker], session, Investigator())

    assert [c.claim_id for c in routed.claims] == ["c2", "c1"]  # order kept
    assert [c.result.summary for c in routed.claims if c.result] == [
        "file_line checker ran",
        "symbol checker ran",
    ]
    assert symbol_checker.seen == [("c1", session)]  # the same session object
    assert file_checker.seen == [("c2", session)]
    assert (routed.plan_id, routed.repo_url) == (plan.plan_id, plan.repo_url)


def test_claim_the_skeptic_already_marked_is_not_checked_again() -> None:
    marked = SYMBOL.with_result(
        EvidenceResult(status=EvidenceStatus.CANT_CHECK, summary="too vague to check: x")
    )
    checker = StubChecker(ClaimKind.SYMBOL)

    routed = route(_plan(marked), [checker], StubSession(), Investigator())

    assert routed.claims == (marked,)
    assert checker.seen == []


def test_unsupported_and_failing_claims_get_cant_check_and_the_rest_still_run() -> None:
    leaky = RuntimeError("SECRET text copied from the report")
    checkers = [
        StubChecker(ClaimKind.SYMBOL),
        StubChecker(ClaimKind.FILE_LINE, error=leaky),
    ]

    routed = route(_plan(SYMBOL, FILE_LINE, POC), checkers, StubSession(), Investigator())

    symbol, file_line, poc = (c.result for c in routed.claims)
    assert symbol is not None and symbol.summary == "symbol checker ran"
    assert file_line is not None and file_line.status is EvidenceStatus.CANT_CHECK
    assert file_line.summary == "checker failed: RuntimeError"  # type only, not the message
    assert poc is not None and poc.summary == "no checker for poc claims"
    assert poc.status is EvidenceStatus.CANT_CHECK
    assert all(claim.result is not None for claim in routed.claims)  # every claim has one


def test_only_not_found_results_go_to_the_investigator() -> None:
    final = EvidenceResult(
        status=EvidenceStatus.CANT_CHECK,
        missing=MissingEvidence.SYMBOL_REMOVED,
        summary="investigator: removed in abc",
    )
    investigator = Investigator(result=final)
    checkers = [StubChecker(ClaimKind.SYMBOL, result=_not_found()), StubChecker(ClaimKind.FILE_LINE)]

    routed = route(_plan(SYMBOL, FILE_LINE), checkers, StubSession(), investigator)

    assert [call[0] for call in investigator.calls] == ["c1"]  # not the plain CANT_CHECK
    assert investigator.calls[0][1].summary == "checker: not found"
    results = [c.result for c in routed.claims]
    assert results[0] == final  # what the investigator returned is attached
    assert results[1] is not None and results[1].summary == "file_line checker ran"
    assert results[1].status is EvidenceStatus.CANT_CHECK  # never upgraded by the router


def test_investigator_failure_keeps_the_checkers_result_with_a_note() -> None:
    investigator = Investigator(error=RuntimeError("SECRET"))

    routed = route(
        _plan(SYMBOL), [StubChecker(ClaimKind.SYMBOL, result=_not_found())], StubSession(), investigator
    )

    [claim] = routed.claims
    assert claim.result is not None
    assert claim.result.status is EvidenceStatus.CANT_CHECK
    assert claim.result.missing is MissingEvidence.SYMBOL_ABSENT  # the checker's result stands
    assert claim.result.summary == "checker: not found; investigation failed: RuntimeError"
