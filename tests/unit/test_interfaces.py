"""Unit tests for the structural interfaces and a fake end-to-end wiring."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from typing import Any

from triage0.core.evidence import (
    CommandResult,
    DeterministicProof,
    EvidenceResult,
    EvidenceStatus,
)
from triage0.core.interfaces import (
    BaseClaimChecker,
    IntakeAdapter,
    SandboxProvider,
    SandboxSession,
)
from triage0.core.models import (
    AnyClaim,
    ClaimKind,
    NormalizedReport,
    ReportSource,
)

TREE_HASH = "a" * 40


class FakeAdapter:
    source = ReportSource.MARKDOWN

    def normalize(
        self,
        payload: str | bytes | Mapping[str, Any],
        *,
        report_id: str | None = None,
    ) -> NormalizedReport:
        body = payload.decode() if isinstance(payload, bytes) else str(payload)
        return NormalizedReport(
            report_id=report_id or "report-1",
            source=self.source,
            body=body,
        )


class FakeSession:
    tree_hash = TREE_HASH
    commit = "c" * 40

    def run(
        self, command: str, *, timeout_s: float | None = None
    ) -> CommandResult:
        return CommandResult(tool="git", command=command, exit_code=0)

    def seal(
        self, command: str, *, timeout_s: float | None = None
    ) -> DeterministicProof:
        return DeterministicProof(
            tool="git",
            command=command,
            exit_code=0,
            git_tree_hash=self.tree_hash,
            commit=self.commit,
        )


class FakeProvider:
    @contextmanager
    def open(
        self,
        repo_url: str,
        ref: str,
        *,
        network_disabled: bool = True,
    ) -> Iterator[SandboxSession]:
        yield FakeSession()


class FakeChecker:
    claim_kind = ClaimKind.SYMBOL

    def supports(self, claim: AnyClaim) -> bool:
        return claim.kind == "symbol"

    def check(
        self, claim: AnyClaim, sandbox: SandboxSession
    ) -> EvidenceResult:
        return EvidenceResult(
            status=EvidenceStatus.CANT_CHECK,
            summary=f"not implemented for {claim.kind}",
        )


class NotAnAdapter:
    pass


def test_intake_adapter_protocol_is_satisfied() -> None:
    assert isinstance(FakeAdapter(), IntakeAdapter)


def test_non_adapter_fails_protocol_check() -> None:
    assert not isinstance(NotAnAdapter(), IntakeAdapter)


def test_sandbox_session_protocol_is_satisfied() -> None:
    assert isinstance(FakeSession(), SandboxSession)


def test_sandbox_provider_protocol_is_satisfied() -> None:
    assert isinstance(FakeProvider(), SandboxProvider)


def test_claim_checker_protocol_is_satisfied() -> None:
    assert isinstance(FakeChecker(), BaseClaimChecker)


def test_fake_wiring_runs_end_to_end() -> None:
    adapter = FakeAdapter()
    provider = FakeProvider()
    checker = FakeChecker()

    report = adapter.normalize("curl crashes on malformed URL", report_id="r1")
    assert report.report_id == "r1"

    with provider.open(report.report_id, "curl-8_13_0") as session:
        raw = session.run("git grep curl_easy_init")
        proof = session.seal("git grep curl_easy_init")

        assert isinstance(raw, CommandResult)
        assert isinstance(proof, DeterministicProof)
        assert proof.git_tree_hash == session.tree_hash

        result = checker.check(
            # The fake checker only needs the discriminated ``kind``.
            claim=type(
                "C",
                (),
                {"kind": "symbol", "claim_id": "c1"},
            )(),  # type: ignore[arg-type]
            sandbox=session,
        )
        assert result.status is EvidenceStatus.CANT_CHECK


def test_session_seal_and_run_are_distinct() -> None:
    session = FakeSession()
    raw = session.run("git status")
    sealed = session.seal("git status")

    assert isinstance(raw, CommandResult)
    assert isinstance(sealed, DeterministicProof)
    assert sealed.commit == session.commit
