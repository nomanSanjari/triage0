"""Stand-ins for the sandbox session and for akib's checkers (#10, not done yet).

They follow the ``BaseClaimChecker`` and ``SandboxSession`` protocols in
``triage0.core.interfaces``, so the router can be tested before the real ones
exist.
"""

from __future__ import annotations

from triage0.core.evidence import (
    CommandResult,
    DeterministicProof,
    EvidenceResult,
    EvidenceStatus,
)
from triage0.core.interfaces import SandboxSession
from triage0.core.models import AnyClaim, ClaimKind

class StubSession:
    """The router never uses the session; checkers get it passed through."""

    tree_hash = "a" * 40
    commit = None

    def run(self, argv: object, *, timeout_s: float | None = None) -> CommandResult:
        raise NotImplementedError

    def seal(self, argv: object, *, timeout_s: float | None = None) -> DeterministicProof:
        raise NotImplementedError


class StubChecker:
    """Handles one claim kind and answers with a scripted result (or error)."""

    def __init__(
        self, kind: ClaimKind, result: EvidenceResult | None = None, error: Exception | None = None
    ) -> None:
        self.claim_kind = kind
        self._result = result or EvidenceResult(
            status=EvidenceStatus.CANT_CHECK, summary=f"{kind.value} checker ran"
        )
        self._error = error
        self.seen: list[tuple[str, object]] = []

    def supports(self, claim: AnyClaim) -> bool:
        return claim.kind == self.claim_kind.value

    def check(self, claim: AnyClaim, sandbox: SandboxSession) -> EvidenceResult:
        self.seen.append((claim.claim_id, sandbox))
        if self._error:
            raise self._error
        return self._result
