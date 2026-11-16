"""Tests for the skeptic and its bounded loop, driven through FakeLLMClient."""

from __future__ import annotations

import json
from typing import Any

import pytest

from tests.golden_fixtures import GoldenCase, load_all
from triage0.core.evidence import EvidenceStatus
from triage0.core.models import (
    NormalizedReport,
    ReportSource,
    SymbolClaim,
    VerificationPlan,
)
from triage0.extraction.decomposer import decompose
from triage0.extraction.skeptic import review
from triage0.llm.client import LLMResponseError
from triage0.llm.fake import FakeLLMClient

REPORT = NormalizedReport(
    report_id="r1",
    source=ReportSource.MARKDOWN,
    body=(
        "`curl_easy_setopt` in `lib/setopt.c` copies the option. "
        "The curl code has memory bugs. "
        "`curlx_dyn_add` in lib/curlx/dynbuf.c grows the buffer.\n"
    ),
)
SPECIFIC = SymbolClaim(
    claim_id="c1",
    statement="curl_easy_setopt is in lib/setopt.c",
    source_excerpt="`curl_easy_setopt` in `lib/setopt.c`",
    symbol="curl_easy_setopt",
    file_hint="lib/setopt.c",
)
VAGUE = SymbolClaim(
    claim_id="c2",
    statement="curl has memory bugs",
    source_excerpt="The curl code has memory bugs",
    symbol="curl",
)
UNCLEAR = SymbolClaim(
    claim_id="c3",
    statement="curlx_dyn_add exists",
    source_excerpt="`curlx_dyn_add`",
    symbol="curlx_dyn_add",
)
REFINED = UNCLEAR.model_copy(
    update={
        "source_excerpt": "`curlx_dyn_add` in lib/curlx/dynbuf.c",
        "file_hint": "lib/curlx/dynbuf.c",
    }
)


def _plan(*claims: SymbolClaim) -> VerificationPlan:
    return VerificationPlan(
        plan_id="p1", report_id="r1", repo_url="https://example.org/x", claims=claims
    )


def _verdicts(**verdicts: str) -> str:
    return json.dumps(
        {
            "assessments": [
                {"claim_id": claim_id, "verdict": verdict, "reason": "a reason"}
                for claim_id, verdict in verdicts.items()
            ]
        }
    )


def _refined(claim: SymbolClaim) -> str:
    return json.dumps(
        {"claim": claim.model_dump(mode="json", exclude={"claim_id", "result"})}
    )


def test_vague_claim_gets_cant_check_and_specific_one_is_untouched() -> None:
    client = FakeLLMClient([_verdicts(c1="testable", c2="vague")])

    reviewed = review(_plan(SPECIFIC, VAGUE), REPORT, client)

    specific, vague = reviewed.claims
    assert specific == SPECIFIC
    assert vague.result is not None
    assert vague.result.status is EvidenceStatus.CANT_CHECK
    assert vague.result.summary == "too vague to check: a reason"
    assert vague.model_copy(update={"result": None}) == VAGUE


def test_skeptic_cannot_give_a_verdict() -> None:
    client = FakeLLMClient([_verdicts(c1="CONTRADICTED")])

    with pytest.raises(LLMResponseError):
        review(_plan(SPECIFIC), REPORT, client)


def test_skeptic_cannot_add_claims() -> None:
    client = FakeLLMClient([_verdicts(c9="vague")])

    reviewed = review(_plan(SPECIFIC), REPORT, client)

    assert reviewed.claims == (SPECIFIC,)


def test_unclear_claim_is_sent_back_once_and_reviewed_again() -> None:
    client = FakeLLMClient(
        [_verdicts(c3="needs_detail"), _refined(REFINED), _verdicts(c3="testable")]
    )

    reviewed = review(_plan(UNCLEAR), REPORT, client)

    assert reviewed.claims == (REFINED,)


def test_loop_stops_after_max_rounds() -> None:
    # A fourth call would make the fake raise: it has no answer scripted.
    client = FakeLLMClient(
        [_verdicts(c3="needs_detail"), _refined(REFINED), _verdicts(c3="needs_detail")]
    )

    reviewed = review(_plan(UNCLEAR), REPORT, client, max_rounds=2)

    [claim] = reviewed.claims
    assert claim.model_copy(update={"result": None}) == REFINED
    assert claim.result is not None
    assert claim.result.summary == "not specific enough to check: a reason"


def test_loop_stops_early_when_send_back_changes_nothing() -> None:
    client = FakeLLMClient([_verdicts(c3="needs_detail"), _refined(UNCLEAR)])

    reviewed = review(_plan(UNCLEAR), REPORT, client, max_rounds=5)

    [claim] = reviewed.claims
    assert claim.result is not None
    assert claim.result.status is EvidenceStatus.CANT_CHECK
    assert len(client.calls) == 2


CASES = {case.case_id: case for case in load_all()}


@pytest.mark.parametrize("case", CASES.values(), ids=CASES.keys())
def test_decompose_then_review_runs_end_to_end_on_the_fixtures(
    case: GoldenCase,
) -> None:
    drafts: list[dict[str, Any]] = [
        e.claim.model_dump(mode="json", exclude={"claim_id", "result"})
        for e in case.expected_claims
    ]
    extraction = json.dumps(
        {"requested_version": case.requested_version, "claims": drafts}
    )
    all_testable = _verdicts(**{e.claim.claim_id: "testable" for e in case.expected_claims})
    client = FakeLLMClient([extraction, all_testable])

    plan = decompose(case.report, case.repo_url, client)
    reviewed = review(plan, case.report, client)

    assert reviewed.claims == tuple(e.claim for e in case.expected_claims)
    # No claims (vague_no_version) means no skeptic call at all.
    assert len(client.calls) == (2 if case.expected_claims else 1)
