"""Structural interfaces for triage0's modular implementations.

These protocols describe the seams between report ingestion, sandbox execution,
and claim checking.  They intentionally contain no implementations and no
runtime dependencies beyond the core models, so alternate backends (local
Docker, microVMs, test fakes) can be swapped in without touching callers.

``SandboxSession.run`` is for exploratory commands.  ``SandboxSession.seal``
re-runs the decisive command and returns a :class:`DeterministicProof`; that is
the proof gate referenced by the sandbox daemon design.

Both methods take an ``argv`` sequence and must execute it without a shell.
Report text is attacker-controlled, so a shell string would allow command
injection from claim fields.  The resulting ``command`` string is display-only.
"""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import AbstractContextManager
from typing import Any, Protocol, runtime_checkable

from triage0.core.evidence import (
    Argv,
    CommandResult,
    DeterministicProof,
    EvidenceResult,
)
from triage0.core.models import (
    AnyClaim,
    ClaimKind,
    NormalizedReport,
    ReportSource,
)

__all__ = [
    "BaseClaimChecker",
    "IntakeAdapter",
    "SandboxProvider",
    "SandboxSession",
]


@runtime_checkable
class IntakeAdapter(Protocol):
    """Normalizes one source format into a :class:`NormalizedReport`."""

    source: ReportSource

    def normalize(
        self,
        payload: str | bytes | Mapping[str, Any],
        *,
        report_id: str | None = None,
        forced_source: ReportSource | None = None,
    ) -> NormalizedReport:
        """Parse ``payload`` into the shared report shape."""
        ...


@runtime_checkable
class SandboxSession(Protocol):
    """A disposable, isolated checkout at a specific git ref."""

    @property
    def tree_hash(self) -> str:
        """Git tree object id of the mounted worktree."""
        ...

    @property
    def commit(self) -> str | None:
        """Resolved commit hash, when known."""
        ...

    def run(
        self,
        argv: Argv,
        *,
        timeout_s: float | None = None,
    ) -> CommandResult:
        """Run an exploratory command without a shell.

        Implementations must set ``timed_out`` and ``truncated`` when they
        enforce execution or output limits.
        """
        ...

    def seal(
        self,
        argv: Argv,
        *,
        timeout_s: float | None = None,
    ) -> DeterministicProof:
        """Re-run the decisive argv and seal it as proof.

        Implementations must execute ``argv`` directly (``shell=False`` or an
        equivalent exec array) and must never pass it through a shell.  They
        must also set ``timed_out``/``truncated`` so an incomplete run cannot
        be presented as terminal evidence.
        """
        ...


@runtime_checkable
class SandboxProvider(Protocol):
    """Boots and tears down isolated sandbox sessions."""

    def open(
        self,
        repo_url: str,
        ref: str,
        *,
        network_disabled: bool = True,
    ) -> AbstractContextManager[SandboxSession]:
        """Open a session for ``repo_url`` at ``ref``.

        The returned context manager must clean up the container and any
        disposable worktree on exit.
        """
        ...


@runtime_checkable
class BaseClaimChecker(Protocol):
    """Verifies one category of claim inside a sandbox session."""

    claim_kind: ClaimKind

    def supports(self, claim: AnyClaim) -> bool:
        """Whether this checker can handle ``claim``."""
        ...

    def check(
        self,
        claim: AnyClaim,
        sandbox: SandboxSession,
    ) -> EvidenceResult:
        """Return a verdict backed by deterministic proof (or ``CANT_CHECK``)."""
        ...
