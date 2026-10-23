"""Unit tests for the deterministic evidence contract."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from triage0.core.evidence import (
    CommandResult,
    DeterministicProof,
    EvidenceResult,
    EvidenceStatus,
)

TREE_HASH = "a" * 40


def make_proof(**overrides: object) -> DeterministicProof:
    data: dict[str, object] = {
        "tool": "git",
        "command": "git grep -n curl_easy_init",
        "exit_code": 0,
        "git_tree_hash": TREE_HASH,
    }
    data.update(overrides)
    return DeterministicProof(**data)


def test_evidence_status_values() -> None:
    assert {status.value for status in EvidenceStatus} == {
        "CONFIRMED",
        "CONTRADICTED",
        "CANT_CHECK",
    }


def test_contradicted_without_proof_raises() -> None:
    with pytest.raises(ValidationError, match="CONTRADICTED verdicts require"):
        EvidenceResult(status=EvidenceStatus.CONTRADICTED)


def test_confirmed_without_proof_raises() -> None:
    with pytest.raises(ValidationError, match="CONFIRMED verdicts require"):
        EvidenceResult(status=EvidenceStatus.CONFIRMED)


def test_cant_check_without_proof_is_allowed() -> None:
    result = EvidenceResult(
        status=EvidenceStatus.CANT_CHECK,
        summary="No deterministic way to test this claim.",
    )
    assert result.status is EvidenceStatus.CANT_CHECK
    assert result.proof is None
    assert result.is_terminal is False


@pytest.mark.parametrize(
    "status", [EvidenceStatus.CONFIRMED, EvidenceStatus.CONTRADICTED]
)
def test_terminal_verdict_with_proof_is_allowed(status: EvidenceStatus) -> None:
    proof = make_proof(exit_code=1 if status is EvidenceStatus.CONTRADICTED else 0)
    result = EvidenceResult(status=status, proof=proof, summary="ok")
    assert result.status is status
    assert result.proof is proof
    assert result.is_terminal is True


def test_cant_check_may_carry_proof_for_diagnostics() -> None:
    result = EvidenceResult(
        status=EvidenceStatus.CANT_CHECK,
        proof=make_proof(exit_code=124),
        summary="command timed out",
    )
    assert result.proof is not None


def test_downgrade_unproven_contradiction() -> None:
    result = EvidenceResult.downgrade(
        status=EvidenceStatus.CONTRADICTED,
        summary="checker could not seal proof",
    )
    assert result.status is EvidenceStatus.CANT_CHECK
    assert result.proof is None
    assert "downgraded to CANT_CHECK" in (result.summary or "")


def test_downgrade_keeps_terminal_verdict_when_proof_present() -> None:
    proof = make_proof()
    result = EvidenceResult.downgrade(
        status=EvidenceStatus.CONFIRMED, proof=proof
    )
    assert result.status is EvidenceStatus.CONFIRMED
    assert result.proof is proof


def test_tree_hash_is_normalized_to_lowercase() -> None:
    proof = make_proof(git_tree_hash="A" * 40)
    assert proof.git_tree_hash == "a" * 40


@pytest.mark.parametrize(
    "bad_hash",
    ["", "abc123", "a" * 39, "a" * 41, "g" * 40, "a" * 63],
)
def test_invalid_tree_hash_rejected(bad_hash: str) -> None:
    with pytest.raises(ValidationError, match="git_tree_hash"):
        make_proof(git_tree_hash=bad_hash)


def test_sha256_tree_hash_accepted() -> None:
    proof = make_proof(git_tree_hash="b" * 64)
    assert proof.git_tree_hash == "b" * 64


@pytest.mark.parametrize("field_name", ["tool", "command"])
def test_blank_identifiers_rejected(field_name: str) -> None:
    with pytest.raises(ValidationError, match="must not be blank"):
        make_proof(**{field_name: "   "})


def test_negative_exit_code_is_allowed_for_signals() -> None:
    proof = make_proof(exit_code=-11)
    assert proof.exit_code == -11


def test_captured_at_must_be_timezone_aware() -> None:
    with pytest.raises(ValidationError, match="timezone-aware"):
        make_proof(captured_at=datetime(2026, 1, 1, 12, 0, 0))


def test_captured_at_defaults_to_utc() -> None:
    proof = make_proof()
    assert proof.captured_at.tzinfo is not None
    assert proof.captured_at.utcoffset() == timezone.utc.utcoffset(None)


def test_from_command_result_preserves_output() -> None:
    result = CommandResult(
        tool="git",
        command="git log -S curl_easy_init",
        exit_code=0,
        stdout="abc123 commit\n",
        stderr="",
        duration_ms=12,
    )
    proof = DeterministicProof.from_command_result(
        result,
        git_tree_hash=TREE_HASH,
        commit="c" * 40,
        workdir="/work/curl",
    )
    assert proof.command == result.command
    assert proof.exit_code == result.exit_code
    assert proof.stdout == result.stdout
    assert proof.commit == "c" * 40
    assert proof.workdir == "/work/curl"


def test_command_result_does_not_strip_stdout() -> None:
    result = CommandResult(
        tool="git",
        command="git status",
        exit_code=0,
        stdout="  leading and trailing  \n",
    )
    assert result.stdout == "  leading and trailing  \n"


def test_models_are_frozen() -> None:
    proof = make_proof()
    with pytest.raises(ValidationError, match="frozen"):
        proof.command = "something else"  # type: ignore[misc]


def test_extra_fields_are_rejected() -> None:
    with pytest.raises(ValidationError):
        make_proof(unexpected="field")
