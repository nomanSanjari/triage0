"""Unit tests for the deterministic evidence contract."""

from __future__ import annotations

import shlex
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
        "argv": ("git", "grep", "-n", "curl_easy_init"),
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
        proof=make_proof(exit_code=124, timed_out=True),
        summary="command timed out",
    )
    assert result.proof is not None


def test_incomplete_flags_default_to_false() -> None:
    result = CommandResult(tool="git", argv=("git", "status"), exit_code=0)
    proof = make_proof()
    assert result.timed_out is False
    assert result.truncated is False
    assert proof.timed_out is False
    assert proof.truncated is False


@pytest.mark.parametrize("flag", ["timed_out", "truncated"])
def test_terminal_verdict_rejected_for_incomplete_proof(flag: str) -> None:
    proof = make_proof(**{flag: True})
    for status in (EvidenceStatus.CONFIRMED, EvidenceStatus.CONTRADICTED):
        with pytest.raises(ValidationError, match="incomplete proof"):
            EvidenceResult(status=status, proof=proof)


@pytest.mark.parametrize("flag", ["timed_out", "truncated"])
def test_cant_check_allows_incomplete_proof(flag: str) -> None:
    proof = make_proof(**{flag: True})
    result = EvidenceResult(status=EvidenceStatus.CANT_CHECK, proof=proof)
    assert result.status is EvidenceStatus.CANT_CHECK
    assert result.proof is proof


@pytest.mark.parametrize(
    ("flag", "fragment"),
    [("timed_out", "timed out"), ("truncated", "truncated")],
)
def test_downgrade_incomplete_proof(flag: str, fragment: str) -> None:
    proof = make_proof(**{flag: True})
    result = EvidenceResult.downgrade(
        status=EvidenceStatus.CONTRADICTED, proof=proof
    )
    assert result.status is EvidenceStatus.CANT_CHECK
    assert result.proof is proof
    assert fragment in (result.summary or "")


def test_truncated_grep_cannot_contradict() -> None:
    proof = make_proof(exit_code=1, truncated=True)
    with pytest.raises(ValidationError, match="truncated=True"):
        EvidenceResult(status=EvidenceStatus.CONTRADICTED, proof=proof)
    result = EvidenceResult.downgrade(
        status=EvidenceStatus.CONTRADICTED, proof=proof
    )
    assert result.status is EvidenceStatus.CANT_CHECK


def test_round_trip_preserves_incomplete_flags() -> None:
    proof = make_proof(timed_out=True, truncated=True)
    restored = DeterministicProof.model_validate_json(proof.model_dump_json())
    assert restored == proof
    assert restored.timed_out is True
    assert restored.truncated is True


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


def test_blank_tool_rejected() -> None:
    with pytest.raises(ValidationError, match="must not be blank"):
        make_proof(tool="   ")


def test_argv_rejects_shell_string() -> None:
    with pytest.raises(ValidationError, match="shell string"):
        make_proof(argv="grep -rn foo")  # type: ignore[arg-type]


def test_argv_rejects_empty_sequence() -> None:
    with pytest.raises(ValidationError, match="at least one element"):
        make_proof(argv=[])


def test_argv_rejects_blank_program() -> None:
    with pytest.raises(ValidationError, match=r"argv\[0\] must not be blank"):
        make_proof(argv=["   ", "foo"])


def test_argv_rejects_nul_bytes() -> None:
    with pytest.raises(ValidationError, match="NUL"):
        make_proof(argv=["grep", "foo\x00bar"])


def test_shell_metacharacters_stay_one_argument() -> None:
    result = CommandResult(
        tool="grep",
        argv=["grep", "-rn", "foo; echo match"],
        exit_code=0,
    )
    assert result.argv == ("grep", "-rn", "foo; echo match")
    assert result.command == "grep -rn 'foo; echo match'"
    assert shlex.split(result.command) == list(result.argv)


def test_command_result_round_trips_through_json() -> None:
    result = CommandResult(tool="git", argv=("git", "status"), exit_code=0)
    restored = CommandResult.model_validate_json(result.model_dump_json())
    assert restored == result
    assert restored.command == result.command


def test_proof_round_trips_through_json() -> None:
    proof = make_proof()
    restored = DeterministicProof.model_validate_json(proof.model_dump_json())
    assert restored == proof


def test_supplied_command_is_recomputed_from_argv() -> None:
    result = CommandResult.model_validate(
        {
            "tool": "grep",
            "argv": ["grep", "-rn", "foo; echo match"],
            "exit_code": 0,
            "command": "grep -rn foo",  # forged display string
        }
    )
    assert result.command == "grep -rn 'foo; echo match'"


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
        argv=("git", "log", "-S", "curl_easy_init"),
        exit_code=0,
        stdout="abc123 commit\n",
        stderr="",
        duration_ms=12,
        timed_out=True,
        truncated=True,
    )
    proof = DeterministicProof.from_command_result(
        result,
        git_tree_hash=TREE_HASH,
        commit="c" * 40,
        workdir="/work/curl",
    )
    assert proof.argv == result.argv
    assert proof.command == result.command
    assert proof.exit_code == result.exit_code
    assert proof.stdout == result.stdout
    assert proof.commit == "c" * 40
    assert proof.workdir == "/work/curl"
    assert proof.timed_out is True
    assert proof.truncated is True


def test_command_result_does_not_strip_stdout() -> None:
    result = CommandResult(
        tool="git",
        argv=("git", "status"),
        exit_code=0,
        stdout="  leading and trailing  \n",
    )
    assert result.stdout == "  leading and trailing  \n"


def test_models_are_frozen() -> None:
    proof = make_proof()
    with pytest.raises(ValidationError, match="frozen"):
        proof.exit_code = 1  # type: ignore[misc]


def test_extra_fields_are_rejected() -> None:
    with pytest.raises(ValidationError):
        make_proof(unexpected="field")
