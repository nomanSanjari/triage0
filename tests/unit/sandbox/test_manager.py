"""Tests for Session: the translation to evidence types, with no Docker.

DockerLocal.exec is replaced by a function that returns a canned result.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from triage0.core.evidence import EvidenceResult, EvidenceStatus
from triage0.sandbox.manager import TIMEOUT_EXIT_CODE, Session
from triage0.sandbox.providers.docker_local import DockerLocal, ExecResult

TREE_HASH = "a" * 40
COMMIT = "b" * 40


def _session(monkeypatch: pytest.MonkeyPatch, result: ExecResult) -> tuple[Session, list[tuple]]:
    """A Session whose container "runs" every command by returning `result`."""
    calls: list[tuple] = []

    def fake_exec(container_id, argv, timeout=None):  # type: ignore[no-untyped-def]
        calls.append((container_id, tuple(argv), timeout))
        return result

    provider = DockerLocal()
    monkeypatch.setattr(provider, "exec", fake_exec)
    session = Session(
        tag="curl-8_13_0", tree_hash=TREE_HASH, commit=COMMIT, container_id="c" * 64, provider=provider
    )
    return session, calls


def _exec_result(**overrides: object) -> ExecResult:
    fields: dict[str, object] = {
        "argv": ("rg", "-n", "curl_easy_init"),
        "exit_code": 0,
        "stdout": "lib/easy.c:1\n",
        "stderr": "",
        "timed_out": False,
        "truncated": False,
    }
    fields.update(overrides)
    return ExecResult(**fields)  # type: ignore[arg-type]


# --- run(): ExecResult -> CommandResult -----------------------------------------


def test_run_translates_a_normal_result(monkeypatch: pytest.MonkeyPatch) -> None:
    session, calls = _session(monkeypatch, _exec_result(exit_code=1, stderr="warn"))
    result = session.run(["rg", "-n", "curl_easy_init"], timeout_s=5)
    assert (result.tool, result.argv, result.exit_code) == ("rg", ("rg", "-n", "curl_easy_init"), 1)
    assert (result.stdout, result.stderr) == ("lib/easy.c:1\n", "warn")
    assert not result.timed_out and not result.truncated
    assert calls == [("c" * 64, ("rg", "-n", "curl_easy_init"), 5)]


def test_run_uses_the_program_name_not_its_path(monkeypatch: pytest.MonkeyPatch) -> None:
    session, _ = _session(monkeypatch, _exec_result(argv=("/usr/bin/rg", "x")))
    assert session.run(["/usr/bin/rg", "x"]).tool == "rg"


def test_run_survives_a_program_path_with_an_empty_basename(monkeypatch: pytest.MonkeyPatch) -> None:
    session, _ = _session(monkeypatch, _exec_result(argv=("/usr/bin/",), exit_code=126))
    assert session.run(["/usr/bin/"]).tool == "/usr/bin/"


def test_run_turns_a_timeout_into_minus_nine_and_a_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    session, _ = _session(monkeypatch, _exec_result(exit_code=None, timed_out=True, stdout=""))
    result = session.run(["sleep", "60"])
    assert result.timed_out is True
    assert result.exit_code == TIMEOUT_EXIT_CODE == -9


def test_run_keeps_the_truncated_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    session, _ = _session(monkeypatch, _exec_result(truncated=True))
    assert session.run(["rg", "x"]).truncated is True


# --- seal(): ExecResult -> DeterministicProof -----------------------------------


def test_seal_records_which_code_the_command_ran_against(monkeypatch: pytest.MonkeyPatch) -> None:
    session, _ = _session(monkeypatch, _exec_result())
    proof = session.seal(["rg", "-n", "curl_easy_init"])
    assert proof.git_tree_hash == TREE_HASH
    assert proof.commit == COMMIT
    assert proof.workdir == "/src"
    assert proof.argv == ("rg", "-n", "curl_easy_init")
    assert proof.command == "rg -n curl_easy_init"
    assert proof.exit_code == 0


def test_a_complete_proof_can_back_a_verdict(monkeypatch: pytest.MonkeyPatch) -> None:
    session, _ = _session(monkeypatch, _exec_result())
    result = EvidenceResult(status=EvidenceStatus.CONFIRMED, proof=session.seal(["rg", "x"]))
    assert result.is_terminal


@pytest.mark.parametrize(
    "flags", [{"truncated": True}, {"exit_code": None, "timed_out": True}], ids=["truncated", "timed_out"]
)
@pytest.mark.parametrize("status", [EvidenceStatus.CONFIRMED, EvidenceStatus.CONTRADICTED])
def test_an_incomplete_proof_cannot_back_a_verdict(
    monkeypatch: pytest.MonkeyPatch, flags: dict[str, object], status: EvidenceStatus
) -> None:
    # The sandbox cut the output or killed the command, so "no match" proves nothing.
    session, _ = _session(monkeypatch, _exec_result(**flags))
    proof = session.seal(["rg", "x"])
    with pytest.raises(ValidationError, match="incomplete proof"):
        EvidenceResult(status=status, proof=proof)
    assert EvidenceResult.downgrade(status=status, proof=proof).status is EvidenceStatus.CANT_CHECK
