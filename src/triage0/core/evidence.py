"""Deterministic evidence contracts for triage0.

This module is the constitutional core of the evidence model.  It depends only
on Pydantic -- never on sandboxes, tools, LLMs, or the network -- so that every
other package can import it without pulling in heavyweight dependencies.

The central invariant is the project's zero-false-flag guarantee: a claim may
only be marked ``CONFIRMED`` or ``CONTRADICTED`` when it carries a
:class:`DeterministicProof` sealed from a real tool invocation.  ``CANT_CHECK``
is the only verdict allowed without proof, and is what model judgment alone can
produce.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from enum import StrEnum

from pydantic import Field, field_validator, model_validator

from triage0.core._base import EvidenceModel

__all__ = [
    "CommandResult",
    "DeterministicProof",
    "EvidenceResult",
    "EvidenceStatus",
]

# Git tree object ids are either SHA-1 (40 hex) or SHA-256 (64 hex).
_TREE_HASH_RE = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})")


class EvidenceStatus(StrEnum):
    """The three verdicts a claim checker may return."""

    CONFIRMED = "CONFIRMED"
    CONTRADICTED = "CONTRADICTED"
    CANT_CHECK = "CANT_CHECK"


class CommandResult(EvidenceModel):
    """Raw output of a sandbox command.

    This is *not* proof.  It records an exploratory invocation; only
    :class:`DeterministicProof` may justify a terminal verdict.
    """

    tool: str = Field(min_length=1)
    command: str = Field(min_length=1)
    exit_code: int
    stdout: str = ""
    stderr: str = ""
    duration_ms: int | None = Field(default=None, ge=0)

    @field_validator("tool", "command")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("must not be blank")
        return stripped


class DeterministicProof(EvidenceModel):
    """Sealed, immutable proof produced by re-running a decisive command.

    A proof is only valid when it records the exact command, its exit code, and
    the git tree object id of the checkout it ran against.  ``commit`` and
    ``workdir`` are optional context; ``stdout``/``stderr`` are retained for
    human audit.
    """

    tool: str = Field(min_length=1)
    command: str = Field(min_length=1)
    exit_code: int
    git_tree_hash: str
    commit: str | None = None
    workdir: str | None = None
    stdout: str | None = None
    stderr: str | None = None
    duration_ms: int | None = Field(default=None, ge=0)
    captured_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    @field_validator("tool", "command")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("must not be blank")
        return stripped

    @field_validator("git_tree_hash")
    @classmethod
    def _validate_tree_hash(cls, value: str) -> str:
        normalized = value.strip().lower()
        if not _TREE_HASH_RE.fullmatch(normalized):
            raise ValueError(
                "git_tree_hash must be a full 40- or 64-character hex tree "
                "object id"
            )
        return normalized

    @field_validator("captured_at")
    @classmethod
    def _require_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("captured_at must be timezone-aware")
        return value

    @classmethod
    def from_command_result(
        cls,
        result: CommandResult,
        *,
        git_tree_hash: str,
        commit: str | None = None,
        workdir: str | None = None,
    ) -> "DeterministicProof":
        """Promote a raw command result into a sealed proof.

        Callers must supply the tree hash of the checkout the command ran
        against; the proof gate is responsible for re-running the decisive
        command before calling this.
        """

        return cls(
            tool=result.tool,
            command=result.command,
            exit_code=result.exit_code,
            git_tree_hash=git_tree_hash,
            commit=commit,
            workdir=workdir,
            stdout=result.stdout,
            stderr=result.stderr,
            duration_ms=result.duration_ms,
        )


class EvidenceResult(EvidenceModel):
    """A verdict for a single claim, backed by deterministic proof.

    ``CONFIRMED`` and ``CONTRADICTED`` both require a populated
    :class:`DeterministicProof`.  ``CANT_CHECK`` may carry one for diagnostics
    but is the only status permitted without it.
    """

    status: EvidenceStatus
    proof: DeterministicProof | None = None
    summary: str | None = None

    @model_validator(mode="after")
    def _terminal_verdicts_require_proof(self) -> "EvidenceResult":
        if self.status is not EvidenceStatus.CANT_CHECK and self.proof is None:
            raise ValueError(
                f"{self.status.value} verdicts require a populated "
                "DeterministicProof (tool command, exit code, and git tree "
                "hash); use CANT_CHECK when no deterministic evidence is "
                "available"
            )
        return self

    @property
    def is_terminal(self) -> bool:
        """Whether the verdict is deterministic (i.e. not ``CANT_CHECK``)."""

        return self.status is not EvidenceStatus.CANT_CHECK

    @classmethod
    def downgrade(
        cls,
        *,
        status: EvidenceStatus,
        proof: DeterministicProof | None = None,
        summary: str | None = None,
    ) -> "EvidenceResult":
        """Build a result, safely downgrading an unproven verdict.

        This is the explicit opt-in alternative to validation: callers that
        cannot supply proof get ``CANT_CHECK`` instead of an exception.
        """

        if status is EvidenceStatus.CANT_CHECK or proof is not None:
            return cls(status=status, proof=proof, summary=summary)

        note = "downgraded to CANT_CHECK: no DeterministicProof was supplied"
        merged = f"{summary}; {note}" if summary else note
        return cls(status=EvidenceStatus.CANT_CHECK, proof=None, summary=merged)
