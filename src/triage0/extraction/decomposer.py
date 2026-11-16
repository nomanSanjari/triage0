"""Turn a report's prose into core claims (roadmap task 2.1b).

The LLM does the reading; this module checks everything it returns, because
its output is as untrusted as the report itself.
"""

from __future__ import annotations

import json
from collections import Counter
from importlib.resources import files
from typing import Annotated, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from triage0.core.models import (
    AnyClaim,
    BuildFlagClaim,
    BuildSystem,
    CallPathClaim,
    FileLineClaim,
    NormalizedReport,
    PoCClaim,
    SymbolClaim,
    SymbolKind,
    VerificationPlan,
)
from triage0.llm.client import LLMClient, LLMResponseError

__all__ = ["decompose", "refine"]


def _read_prompt(name: str) -> str:
    return (files("triage0.extraction.prompts") / name).read_text(encoding="utf-8")


# Read at import, so a missing prompt file fails at startup, not mid-run. Both
# tasks share one set of claim rules, so the two can't drift apart.
_CLAIM_RULES = _read_prompt("claim_rules.txt")
_DECOMPOSE_PROMPT = f"{_CLAIM_RULES}\n{_read_prompt('decompose.txt')}"
_REFINE_PROMPT = f"{_CLAIM_RULES}\n{_read_prompt('refine.txt')}"


# --- Draft forms: what the LLM is allowed to fill in --------------------------
#
# Each draft mirrors a core claim minus two fields:
# - ``result``: a core claim can carry a verdict with a proof, and a proof is
#   only checked for looking like a git hash, which an LLM can make up. With no
#   such field, a report can't get the LLM to pre-stamp its own claim.
# - ``claim_id``: code numbers the claims, so ids are always unique.
# ``extra="forbid"`` rejects any other field, so an answer in a different shape
# (e.g. one a report's injected instructions asked for) fails validation.
#
# Value checks (min lengths, line numbers >= 1) are left to the core classes,
# which run them in ``_to_claim``, so they are written in one place only.


class _Form(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class _ClaimDraft(_Form):
    statement: str
    # Optional here so one claim without a quote is dropped and counted,
    # instead of failing the whole answer.
    source_excerpt: str | None = None


class _SymbolDraft(_ClaimDraft):
    kind: Literal["symbol"]
    symbol: str
    symbol_kind: SymbolKind = SymbolKind.UNKNOWN
    file_hint: str | None = None
    expected_present: bool = True


class _FileLineDraft(_ClaimDraft):
    kind: Literal["file_line"]
    path: str
    line_start: int
    line_end: int | None = None
    expected_snippet: str | None = None


class _CallPathDraft(_ClaimDraft):
    kind: Literal["call_path"]
    entrypoint: str
    target: str
    expected_reachable: bool = True
    file_hint: str | None = None


class _BuildFlagDraft(_ClaimDraft):
    kind: Literal["build_flag"]
    build_system: BuildSystem
    flag: str
    expected_enabled: bool = True
    config: str | None = None
    target: str | None = None


class _PoCDraft(_ClaimDraft):
    kind: Literal["poc"]
    steps: tuple[str, ...]
    expected_outcome: str
    crash_type: str | None = None
    sanitizer: str | None = None


_AnyDraft = Annotated[
    _SymbolDraft | _FileLineDraft | _CallPathDraft | _BuildFlagDraft | _PoCDraft,
    Field(discriminator="kind"),
]


class _Extraction(_Form):
    """The whole answer. No defaults: the LLM must state both, even if empty."""

    requested_version: str | None
    claims: tuple[_AnyDraft, ...]


class _Refinement(_Form):
    """The answer to a send-back: the claim again, or null if the report lacks it."""

    claim: _AnyDraft | None


_CLAIM_CLASSES: dict[
    str,
    type[SymbolClaim]
    | type[FileLineClaim]
    | type[CallPathClaim]
    | type[BuildFlagClaim]
    | type[PoCClaim],
] = {
    "symbol": SymbolClaim,
    "file_line": FileLineClaim,
    "call_path": CallPathClaim,
    "build_flag": BuildFlagClaim,
    "poc": PoCClaim,
}


def _to_claim(draft: _AnyDraft, claim_id: str) -> AnyClaim:
    """Copy a checked draft onto the real core claim.

    Core runs its own value checks here. A draft that fails them is malformed
    LLM output, reported like any other (``LLMResponseError``), never a crash.
    """

    claim_class = _CLAIM_CLASSES[draft.kind]
    try:
        return claim_class(claim_id=claim_id, **draft.model_dump())
    except ValidationError as exc:
        raise LLMResponseError(
            f"model output had an invalid {draft.kind} claim: "
            f"{exc.error_count()} error(s)"
        ) from None


# --- Checks: is each claim really in the report? -----------------------------

_EXCERPT_MISSING = "excerpt not in report"
_NAME_MISSING = "name not in report"


def _key_names(claim: AnyClaim) -> tuple[str, ...]:
    """The names a checker will search for, so they must come from the report.

    PoC steps are free text with no single name to look up; the excerpt check
    is their only guard.
    """

    match claim:
        case SymbolClaim():
            return (claim.symbol,)
        case FileLineClaim():
            return (claim.path,)
        case CallPathClaim():
            return (claim.entrypoint, claim.target)
        case BuildFlagClaim():
            return (claim.flag,)
        case PoCClaim():
            return ()


def _drop_reason(claim: AnyClaim, body: str) -> str | None:
    """Why ``claim`` can't be shown to come from ``body``, or None if it can."""

    # Checked first on its own: "" is a substring of every string.
    if not claim.source_excerpt or claim.source_excerpt not in body:
        return _EXCERPT_MISSING
    # The excerpt alone isn't enough: a real quote can sit next to an invented
    # name, and the checkers would then look for something never reported.
    if any(name not in body for name in _key_names(claim)):
        return _NAME_MISSING
    return None


def _drop_note(dropped: Counter[str], total: int, version_dropped: bool) -> str | None:
    parts = []
    if dropped:
        reasons = ", ".join(f"{n} {reason}" for reason, n in sorted(dropped.items()))
        parts.append(f"dropped {sum(dropped.values())} of {total} claim(s): {reasons}")
    if version_dropped:
        parts.append("dropped requested_version: not in report")
    return "; ".join(parts) or None


# --- Public entry point -------------------------------------------------------


def decompose(
    report: NormalizedReport, repo_url: str, client: LLMClient
) -> VerificationPlan:
    """Extract the report's checkable claims into a plan for the checkers.

    Only the body is sent, as the whole user message: with no markers around
    it, there is no end-of-report marker a report could fake. Claims and the
    version text are kept only if they can be found in the body. Dropped ones
    are counted in ``notes`` (only counts, never LLM text). Malformed LLM output
    raises ``LLMResponseError``: an empty plan always means "nothing to check".
    """

    answer = client.complete_structured(_DECOMPOSE_PROMPT, report.body, _Extraction)

    claims: list[AnyClaim] = []
    dropped: Counter[str] = Counter()
    for draft in answer.claims:
        claim = _to_claim(draft, claim_id=f"c{len(claims) + 1}")
        reason = _drop_reason(claim, report.body)
        if reason is None:
            claims.append(claim)
        else:
            dropped[reason] += 1

    raw_version = (answer.requested_version or "").strip()
    version = raw_version if raw_version and raw_version in report.body else None
    version_dropped = bool(raw_version) and version is None

    return VerificationPlan(
        plan_id=uuid4().hex,
        report_id=report.report_id,
        repo_url=repo_url,
        project=report.project,
        requested_version=version,
        claims=tuple(claims),
        notes=_drop_note(dropped, len(answer.claims), version_dropped),
    )


def refine(
    report: NormalizedReport, claim: AnyClaim, reason: str, client: LLMClient
) -> AnyClaim | None:
    """Ask again for one claim the skeptic sent back, with its reason.

    Returns the new claim under the same ``claim_id``, or None if the report
    doesn't state the missing detail or the new claim fails the same checks
    as in ``decompose`` (so a send-back can't sneak a claim past them).

    The claim and reason were written by an LLM, so they are as untrusted as
    the report: all three travel as one JSON user message, never in the
    instructions. JSON escapes quotes, so none of them can fake the end of
    its field.
    """

    request = json.dumps(
        {
            "report": report.body,
            "claim": claim.model_dump(mode="json", exclude={"claim_id", "result"}),
            "reviewer_note": reason,
        },
        ensure_ascii=False,
    )
    answer = client.complete_structured(_REFINE_PROMPT, request, _Refinement)
    if answer.claim is None:
        return None
    revised = _to_claim(answer.claim, claim_id=claim.claim_id)
    return revised if _drop_reason(revised, report.body) is None else None
