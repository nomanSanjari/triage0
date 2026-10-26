"""Deterministic evidence contracts for triage0.

This module is the constitutional core of the evidence model.  It depends only
on Pydantic -- never on sandboxes, tools, LLMs, or the network -- so that every
other package can import it without pulling in heavyweight dependencies.

The central invariant is the project's zero-false-flag guarantee: a claim may
only be marked ``CONFIRMED`` or ``CONTRADICTED`` when it carries a
:class:`DeterministicProof` sealed from a real tool invocation.  ``CANT_CHECK``
is the only verdict allowed without proof, and is what model judgment alone can
produce.

Commands are recorded as ``argv`` sequences, never as shell strings.  Report
text is attacker-controlled, so interpolating a value into a shell command
would let a fake claim forge a successful tool result.  ``command`` is derived
for display only.
"""

from __future__ import annotations

import re
import shlex
from datetime import datetime, timezone
from enum import StrEnum

from pydantic import Field, computed_field, field_validator, model_validator

from triage0.core._base import EvidenceModel

__all__ = [
    "Argv",
    "CommandResult",
    "DeterministicProof",
    "EvidenceResult",
    "EvidenceStatus",
]

# A command is a sequence of arguments, never a shell string.  ``str`` is
# deliberately excluded even though it satisfies ``Sequence[str]``: accepting
# one here would reintroduce shell parsing.
Argv = tuple[str, ...] | list[str]

# Git tree object ids are either SHA-1 (40 hex) or SHA-256 (64 hex).
_TREE_HASH_RE = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})")


class EvidenceStatus(StrEnum):
    """The three verdicts a claim checker may return."""

    CONFIRMED = "CONFIRMED"
    CONTRADICTED = "CONTRADICTED"
    CANT_CHECK = "CANT_CHECK"


class _CommandRecord(EvidenceModel):
    """Fields shared by a raw command result and a sealed proof.

    ``argv`` is the canonical record of the invocation and is executed without
    a shell.  ``command`` is a derived, human-readable rendering for display and
    audit only; it must never be passed back to a shell.
    """

    tool: str = Field(min_length=1)
    argv: tuple[str, ...]
    exit_code: int
    stdout: str = ""
    stderr: str = ""
    duration_ms: int | None = Field(default=None, ge=0)
    timed_out: bool = False
    truncated: bool = False

    @field_validator("tool")
    @classmethod
    def _tool_not_blank(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("must not be blank")
        return stripped

    @field_validator("argv", mode="before")
    @classmethod
    def _reject_shell_string(cls, value: object) -> object:
        if isinstance(value, (str, bytes)):
            raise ValueError(
                "argv must be a sequence of arguments, not a shell string"
            )
        return value

    @field_validator("argv")
    @classmethod
    def _validate_argv(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if not value:
            raise ValueError("argv must contain at least one element")
        if not value[0].strip():
            raise ValueError("argv[0] must not be blank")
        if any("\x00" in arg for arg in value):
            raise ValueError("argv must not contain NUL bytes")
        return value

    @computed_field  # type: ignore[prop-decorator]
    @property
    def command(self) -> str:
        """POSIX-quoted display form; never execute this string."""

        return shlex.join(self.argv)

    @model_validator(mode="before")
    @classmethod
    def _discard_supplied_command(cls, data: object) -> object:
        """Re-derive ``command`` from ``argv``; ignore serialized copies."""

        if isinstance(data, dict) and "command" in data:
            return {
                key: value for key, value in data.items() if key != "command"
            }
        return data


class CommandResult(_CommandRecord):
    """Raw output of a sandbox command.

    This is *not* proof.  It records an exploratory invocation; only
    :class:`DeterministicProof` may justify a terminal verdict.
    """


class DeterministicProof(_CommandRecord):
    """Sealed, immutable proof produced by re-running a decisive command.

    A proof is only valid when it records the exact argv, its exit code, and
    the git tree object id of the checkout it ran against.  ``commit`` and
    ``workdir`` are optional context; ``stdout``/``stderr`` are retained for
    human audit.
    """

    git_tree_hash: str
    commit: str | None = None
    workdir: str | None = None
    captured_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

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
        argv before calling this.
        """

        return cls(
            tool=result.tool,
            argv=result.argv,
            exit_code=result.exit_code,
            git_tree_hash=git_tree_hash,
            commit=commit,
            workdir=workdir,
            stdout=result.stdout,
            stderr=result.stderr,
            duration_ms=result.duration_ms,
            timed_out=result.timed_out,
            truncated=result.truncated,
        )


def _incomplete_proof_modes(proof: DeterministicProof) -> tuple[str, ...]:
    """Return the reasons a proof cannot support a terminal verdict."""

    modes: list[str] = []
    if proof.timed_out:
        modes.append("timed_out")
    if proof.truncated:
        modes.append("truncated")
    return tuple(modes)


class EvidenceResult(EvidenceModel):
    """A verdict for a single claim, backed by deterministic proof.

    ``CONFIRMED`` and ``CONTRADICTED`` both require a populated
    :class:`DeterministicProof` that is conclusive: a proof with ``timed_out``
    or ``truncated`` set cannot support a terminal verdict.  ``CANT_CHECK`` may
    carry a proof for diagnostics but is the only status permitted without one.
    """

    status: EvidenceStatus
    proof: DeterministicProof | None = None
    summary: str | None = None

    @model_validator(mode="after")
    def _terminal_verdicts_require_proof(self) -> "EvidenceResult":
        if self.status is EvidenceStatus.CANT_CHECK:
            return self

        if self.proof is None:
            raise ValueError(
                f"{self.status.value} verdicts require a populated "
                "DeterministicProof (tool argv, exit code, and git tree "
                "hash); use CANT_CHECK when no deterministic evidence is "
                "available"
            )

        incomplete = _incomplete_proof_modes(self.proof)
        if incomplete:
            flags = ", ".join(f"{mode}=True" for mode in incomplete)
            raise ValueError(
                f"{self.status.value} verdicts cannot be sealed from an "
                f"incomplete proof ({flags}); use CANT_CHECK or re-run with "
                "bounded output"
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
        """Build a result, safely downgrading an inconclusive verdict.

        This is the explicit opt-in alternative to validation: callers without
        proof, or with a timed-out/truncated proof, get ``CANT_CHECK`` instead
        of an exception.
        """

        if status is EvidenceStatus.CANT_CHECK:
            return cls(status=status, proof=proof, summary=summary)

        if proof is None:
            note = (
                "downgraded to CANT_CHECK: no DeterministicProof was supplied"
            )
            merged = f"{summary}; {note}" if summary else note
            return cls(
                status=EvidenceStatus.CANT_CHECK,
                proof=None,
                summary=merged,
            )

        incomplete = _incomplete_proof_modes(proof)
        if incomplete:
            reason = " and ".join(
                mode.replace("_", " ") for mode in incomplete
            )
            note = f"downgraded to CANT_CHECK: proof was {reason}"
            merged = f"{summary}; {note}" if summary else note
            return cls(
                status=EvidenceStatus.CANT_CHECK,
                proof=proof,
                summary=merged,
            )

        return cls(status=status, proof=proof, summary=summary)
