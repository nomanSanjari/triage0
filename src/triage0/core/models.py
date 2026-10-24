"""Shared domain models for triage0.

These models form the contract between report ingestion, verification planning,
sandbox checking, and reporting.  They depend only on :mod:`triage0.core.evidence`
and Pydantic, keeping the core package free of tools and LLM clients.

Claims are modelled as a tagged union discriminated by ``kind``.  The base
class :class:`ClaimBase` deliberately does not declare ``kind`` so each concrete
claim can pin its discriminator with a ``Literal`` without fighting static type
checkers.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import StrEnum
from hashlib import sha256
from typing import Annotated, Any, Literal, Self

from pydantic import (
    Field,
    computed_field,
    field_validator,
    model_validator,
)

from triage0.core._base import DomainModel
from triage0.core.evidence import EvidenceResult

__all__ = [
    "AnyClaim",
    "BuildFlagClaim",
    "BuildSystem",
    "CallPathClaim",
    "ClaimBase",
    "ClaimKind",
    "FileLineClaim",
    "NormalizedReport",
    "PoCClaim",
    "ReportSource",
    "SymbolClaim",
    "SymbolKind",
    "VerificationPlan",
]


class ReportSource(StrEnum):
    """Where a normalized report came from."""

    MARKDOWN = "markdown"
    EMAIL = "email"
    HACKERONE = "hackerone"
    GITHUB_ADVISORY = "github_advisory"


class ClaimKind(StrEnum):
    """The atomic claim categories understood by the checker router."""

    SYMBOL = "symbol"
    FILE_LINE = "file_line"
    CALL_PATH = "call_path"
    BUILD_FLAG = "build_flag"
    POC = "poc"


class SymbolKind(StrEnum):
    """What kind of code symbol a :class:`SymbolClaim` refers to."""

    FUNCTION = "function"
    STRUCT = "struct"
    MACRO = "macro"
    VARIABLE = "variable"
    TYPE = "type"
    UNKNOWN = "unknown"


class BuildSystem(StrEnum):
    """Build system owning the flags referenced by a :class:`BuildFlagClaim`."""

    CMAKE = "cmake"
    AUTOCONF = "autoconf"
    MAKE = "make"
    MESON = "meson"
    OTHER = "other"


def _require_aware(value: datetime | None) -> datetime | None:
    if value is not None and (value.tzinfo is None or value.utcoffset() is None):
        raise ValueError("datetime values must be timezone-aware")
    return value


class NormalizedReport(DomainModel):
    """A single incoming report, normalized to plain text.

    Adapters are responsible for filling ``body`` with the canonical report
    text (Markdown or plain text).  ``project`` is optional because pasted
    reports and emails often omit it; the repository on the resulting
    :class:`VerificationPlan` is the authoritative execution target.
    """

    report_id: str = Field(min_length=1)
    source: ReportSource
    project: str | None = None
    title: str | None = None
    body: str
    reported_version: str | None = None
    submitted_at: datetime | None = None
    source_url: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("body")
    @classmethod
    def _body_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("body must contain report text")
        return value

    @field_validator("submitted_at")
    @classmethod
    def _submitted_at_aware(cls, value: datetime | None) -> datetime | None:
        return _require_aware(value)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def content_hash(self) -> str:
        """Stable SHA-256 of the normalized body, used for deduplication."""

        return sha256(self.body.encode("utf-8")).hexdigest()


class ClaimBase(DomainModel):
    """Common shape for every atomic claim.

    This class is intentionally not part of :data:`AnyClaim`; instantiate one
    of the concrete subclasses instead.
    """

    claim_id: str = Field(min_length=1)
    statement: str = Field(min_length=1)
    source_excerpt: str | None = None
    result: EvidenceResult | None = None

    def with_result(self, result: EvidenceResult) -> Self:
        """Return a copy of this claim annotated with a checker verdict."""

        return self.model_copy(update={"result": result})


class SymbolClaim(ClaimBase):
    """A claim that a named code symbol exists (or does not exist)."""

    kind: Literal["symbol"] = "symbol"
    symbol: str = Field(min_length=1)
    symbol_kind: SymbolKind = SymbolKind.UNKNOWN
    file_hint: str | None = None
    expected_present: bool = True


class FileLineClaim(ClaimBase):
    """A claim about the contents of one or more lines in a file."""

    kind: Literal["file_line"] = "file_line"
    path: str = Field(min_length=1)
    line_start: int = Field(ge=1)
    line_end: int | None = Field(default=None, ge=1)
    expected_snippet: str | None = None

    @model_validator(mode="after")
    def _validate_line_range(self) -> "FileLineClaim":
        if self.line_end is not None and self.line_end < self.line_start:
            raise ValueError(
                "line_end must be greater than or equal to line_start"
            )
        return self

    @property
    def effective_line_end(self) -> int:
        """Convenience accessor treating a missing ``line_end`` as one line."""

        return self.line_end if self.line_end is not None else self.line_start


class CallPathClaim(ClaimBase):
    """A claim that one function can (or cannot) reach another."""

    kind: Literal["call_path"] = "call_path"
    entrypoint: str = Field(min_length=1)
    target: str = Field(min_length=1)
    expected_reachable: bool = True
    file_hint: str | None = None


class BuildFlagClaim(ClaimBase):
    """A claim about a compile-time flag or option in the build system."""

    kind: Literal["build_flag"] = "build_flag"
    build_system: BuildSystem
    flag: str = Field(min_length=1)
    expected_enabled: bool = True
    config: str | None = None
    target: str | None = None


class PoCClaim(ClaimBase):
    """A claim that a sequence of steps reproduces a specific outcome."""

    kind: Literal["poc"] = "poc"
    steps: tuple[str, ...] = Field(min_length=1)
    expected_outcome: str = Field(min_length=1)
    crash_type: str | None = None
    sanitizer: str | None = None

    @field_validator("steps")
    @classmethod
    def _steps_not_blank(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if any(not step.strip() for step in value):
            raise ValueError("PoC steps must not be blank")
        return value


AnyClaim = Annotated[
    SymbolClaim
    | FileLineClaim
    | CallPathClaim
    | BuildFlagClaim
    | PoCClaim,
    Field(discriminator="kind"),
]


class VerificationPlan(DomainModel):
    """The immutable work order produced from a report.

    ``requested_version`` is the raw version text parsed from the report;
    ``resolved_ref`` and ``resolved_commit`` are filled in once fuzzy tag
    resolution has run.  ``repo_url`` is the repository the sandbox checks out.

    ``result`` fields on individual claims are populated *after* the plan is
    created (via :meth:`ClaimBase.with_result`); persisting a completed run is
    a later concern.
    """

    plan_id: str = Field(min_length=1)
    report_id: str = Field(min_length=1)
    repo_url: str = Field(min_length=1)
    project: str | None = None
    requested_version: str | None = None
    resolved_ref: str | None = None
    resolved_commit: str | None = None
    claims: tuple[AnyClaim, ...]
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    notes: str | None = None

    @field_validator("created_at")
    @classmethod
    def _created_at_aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("created_at must be timezone-aware")
        return value

    @field_validator("claims")
    @classmethod
    def _unique_claim_ids(
        cls, claims: tuple[AnyClaim, ...]
    ) -> tuple[AnyClaim, ...]:
        seen: set[str] = set()
        duplicates: set[str] = set()
        for claim in claims:
            if claim.claim_id in seen:
                duplicates.add(claim.claim_id)
            seen.add(claim.claim_id)
        if duplicates:
            formatted = ", ".join(sorted(duplicates))
            raise ValueError(f"duplicate claim_id values: {formatted}")
        return claims
