"""Mark claims too vague to check before any checker runs (roadmap task 2.1c).

The skeptic is an LLM, so it can only annotate: the verdicts it may give are
fixed by the answer form below, and code, not the LLM, writes the
``CANT_CHECK`` result. A wrong "vague" costs a missed check, never a false
flag (rule 1).
"""

from __future__ import annotations

import json
from importlib.resources import files
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from triage0.core.evidence import EvidenceResult, EvidenceStatus
from triage0.core.models import AnyClaim, NormalizedReport, VerificationPlan
from triage0.extraction.decomposer import refine
from triage0.llm.client import LLMClient

__all__ = ["review"]

# Read at import, so a missing prompt file fails at startup, not mid-run.
_PROMPT = (files("triage0.extraction.prompts") / "skeptic.txt").read_text(
    encoding="utf-8"
)


class _Form(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class _Assessment(_Form):
    claim_id: str
    # No CONFIRMED or CONTRADICTED here: an answer using them fails validation,
    # so the limit is enforced by the type, not only asked for in the prompt.
    verdict: Literal["testable", "vague", "needs_detail"]
    reason: str = Field(min_length=1)


class _Review(_Form):
    assessments: tuple[_Assessment, ...]


def _assess(
    claims: list[AnyClaim], client: LLMClient
) -> dict[str, _Assessment]:
    """One skeptic call over ``claims``, keyed by claim_id.

    Ids the plan doesn't have are ignored, so the skeptic can't add claims. A
    claim it skips stays testable: the worst case is a checker running on it.
    """

    request = json.dumps(
        {"claims": [c.model_dump(mode="json", exclude={"result"}) for c in claims]},
        ensure_ascii=False,
    )
    answer = client.complete_structured(_PROMPT, request, _Review)
    known = {claim.claim_id for claim in claims}
    return {a.claim_id: a for a in answer.assessments if a.claim_id in known}


def _cant_check(claim: AnyClaim, summary: str) -> AnyClaim:
    return claim.with_result(
        EvidenceResult(status=EvidenceStatus.CANT_CHECK, summary=summary)
    )


def review(
    plan: VerificationPlan,
    report: NormalizedReport,
    client: LLMClient,
    max_rounds: int = 2,
) -> VerificationPlan:
    """Mark vague claims ``CANT_CHECK``, sending unclear ones back once first.

    Each round is one skeptic call over the claims still under review. A
    ``needs_detail`` claim goes back to the decomposer with the skeptic's
    reason, and the new version is reviewed next round. The loop ends when no
    claim is under review or after ``max_rounds``; a claim still unclear then,
    or that the decomposer couldn't improve, becomes ``CANT_CHECK``. With the
    default of 2, each claim is sent back at most once.

    ``report`` is needed because a send-back re-reads the report.
    """

    claims = {claim.claim_id: claim for claim in plan.claims}
    pending = list(plan.claims)
    for round_number in range(1, max_rounds + 1):
        if not pending:
            break
        is_last = round_number == max_rounds
        assessments = _assess(pending, client)
        still_pending: list[AnyClaim] = []
        for claim in pending:
            assessment = assessments.get(claim.claim_id)
            if assessment is None or assessment.verdict == "testable":
                continue
            unclear = f"not specific enough to check: {assessment.reason}"
            if assessment.verdict == "vague":
                claims[claim.claim_id] = _cant_check(
                    claim, f"too vague to check: {assessment.reason}"
                )
            elif is_last:
                claims[claim.claim_id] = _cant_check(claim, unclear)
            else:
                revised = refine(report, claim, assessment.reason, client)
                if revised is None or revised == claim:
                    claims[claim.claim_id] = _cant_check(claim, unclear)
                else:
                    claims[claim.claim_id] = revised
                    still_pending.append(revised)
        pending = still_pending

    return plan.model_copy(update={"claims": tuple(claims.values())})
