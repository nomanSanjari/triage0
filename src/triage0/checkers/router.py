"""Send each claim in a plan to the checker that handles it (roadmap task 2.4a).

The router only dispatches and attaches results. It never creates a verdict:
``CONFIRMED`` and ``CONTRADICTED`` can only come from a checker or the code
investigator, each with its own proof, so the router can't turn a "can't
check" into one (rule 1). One bad claim never stops the others, and every
claim leaves with a result.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

from triage0.core.evidence import EvidenceResult, EvidenceStatus
from triage0.core.interfaces import BaseClaimChecker, SandboxSession
from triage0.core.models import AnyClaim, VerificationPlan

__all__ = ["route"]

# Called with a claim and the checker's "not found" result; returns the final one.
Investigator = Callable[[AnyClaim, EvidenceResult], EvidenceResult]


def route(
    plan: VerificationPlan,
    checkers: Sequence[BaseClaimChecker],
    session: SandboxSession,
    investigate: Investigator,
) -> VerificationPlan:
    """Return ``plan`` with a result on every claim.

    A claim that already has a result (the skeptic marked it too vague) is
    left alone. Otherwise the first checker whose ``supports`` accepts it runs
    it, and a result carrying the not-found signal (``missing``) goes to
    ``investigate`` before it is attached.
    """

    claims = tuple(_route_claim(claim, checkers, session, investigate) for claim in plan.claims)
    return plan.model_copy(update={"claims": claims})


def _route_claim(
    claim: AnyClaim,
    checkers: Sequence[BaseClaimChecker],
    session: SandboxSession,
    investigate: Investigator,
) -> AnyClaim:
    if claim.result is not None:
        return claim
    try:
        checker = next((c for c in checkers if c.supports(claim)), None)
        if checker is None:
            return claim.with_result(_cant_check(f"no checker for {claim.kind} claims"))
        result = checker.check(claim, session)
    except Exception as error:  # isolate one claim's failure from the rest
        # Only the type: the message can echo report text (untrusted).
        return claim.with_result(_cant_check(f"checker failed: {type(error).__name__}"))

    if result.missing is not None:
        try:
            result = investigate(claim, result)
        except Exception as error:
            # The checker's own CANT_CHECK stands, with a note so the failed
            # search isn't invisible (type only, as above).
            note = f"investigation failed: {type(error).__name__}"
            summary = f"{result.summary}; {note}" if result.summary else note
            result = result.model_copy(update={"summary": summary})
    return claim.with_result(result)


def _cant_check(summary: str) -> EvidenceResult:
    return EvidenceResult(status=EvidenceStatus.CANT_CHECK, summary=summary)
