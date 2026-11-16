"""Turn a report's prose into core claims (roadmap task 2.1b).

The LLM does the reading; this module checks everything it returns, because
its output is as untrusted as the report itself.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from triage0.core.models import (
    AnyClaim,
    BuildFlagClaim,
    BuildSystem,
    CallPathClaim,
    FileLineClaim,
    PoCClaim,
    SymbolClaim,
    SymbolKind,
)
from triage0.llm.client import LLMResponseError

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
