"""Call-path verdicts require a bounded witness in successful sealed output."""

from __future__ import annotations

import json

import pytest

from triage0.checkers.call_path_checker import CallPathChecker
from triage0.core.evidence import Argv, CommandResult, DeterministicProof, EvidenceStatus
from triage0.core.interfaces import BaseClaimChecker
from triage0.core.models import CallPathClaim, ClaimKind


class FakeSession:
    tree_hash = "a" * 40
    commit = "b" * 40

    def __init__(self, graph: str, **options: object) -> None:
        self.graph = graph
        self.options = options
        self.calls: list[tuple[str, ...]] = []

    def run(self, argv: Argv, *, timeout_s: float | None = None) -> CommandResult:
        self.calls.append(tuple(argv))
        assert argv[0] == "realpath"
        return CommandResult(tool="realpath", argv=argv, exit_code=0, stdout=argv[-1] + "\n")

    def seal(self, argv: Argv, *, timeout_s: float | None = None) -> DeterministicProof:
        assert timeout_s is not None and 0 < timeout_s <= 30
        self.calls.append(tuple(argv))
        fields: dict[str, object] = {
            "tool": "perl", "argv": argv, "exit_code": 0, "stdout": self.graph,
        }
        fields.update(self.options)
        return DeterministicProof.from_command_result(
            CommandResult(**fields), git_tree_hash=self.tree_hash,
        )


def claim(**overrides: object) -> CallPathClaim:
    fields: dict[str, object] = {
        "claim_id": "c1", "statement": "entry reaches target", "entrypoint": "entry",
        "target": "target", "file_hint": "test.c",
    }
    fields.update(overrides)
    return CallPathClaim(**fields)


GRAPH = '[["entry", "helper", 5], ["helper", "target", 8]]'


def test_each_hop_is_cited_and_sealed() -> None:
    checker = CallPathChecker()
    assert isinstance(checker, BaseClaimChecker)
    assert checker.claim_kind is ClaimKind.CALL_PATH and checker.supports(claim())
    result = checker.check(claim(), FakeSession(GRAPH))

    assert result.status is EvidenceStatus.CONFIRMED
    assert result.proof is not None and result.proof.stdout == GRAPH
    assert "entry -> helper (test.c:5)" in (result.summary or "")
    assert "helper -> target (test.c:8)" in (result.summary or "")


@pytest.mark.parametrize("options", [{"timed_out": True}, {"truncated": True}, {"exit_code": 1}])
def test_incomplete_or_failed_proof_cannot_confirm(options: dict[str, object]) -> None:
    result = CallPathChecker().check(claim(), FakeSession(GRAPH, **options))
    assert result.status is EvidenceStatus.CANT_CHECK
    assert result.proof is not None and result.missing is None


@pytest.mark.parametrize("graph", ["[", '[["entry", "target", true]]', '[["entry", "entry", 1]]'])
def test_invalid_or_non_witnessing_graph_produces_no_missing_signal(graph: str) -> None:
    result = CallPathChecker().check(claim(), FakeSession(graph))
    assert result.status is EvidenceStatus.CANT_CHECK and result.missing is None


def test_depth_limit_prevents_a_verdict_for_longer_paths() -> None:
    graph = json.dumps([[f"f{i}", f"f{i + 1}", 1] for i in range(9)])
    result = CallPathChecker().check(claim(entrypoint="f0", target="f9"), FakeSession(graph))
    assert result.status is EvidenceStatus.CANT_CHECK and result.missing is None


def test_a_path_at_the_depth_limit_is_confirmed() -> None:
    """The boundary is inclusive: eight edges are reported, nine are not."""

    graph = json.dumps([[f"f{i}", f"f{i + 1}", i + 1] for i in range(8)])
    result = CallPathChecker().check(claim(entrypoint="f0", target="f8"), FakeSession(graph))

    assert result.status is EvidenceStatus.CONFIRMED
    assert (result.summary or "").count(" -> ") == 8


def test_the_shortest_path_is_reported() -> None:
    graph = json.dumps(
        [["entry", "a", 2], ["a", "b", 3], ["b", "target", 4], ["entry", "target", 9]]
    )
    result = CallPathChecker().check(claim(), FakeSession(graph))

    assert result.status is EvidenceStatus.CONFIRMED
    assert result.summary is not None and "test.c:9" in result.summary


def test_a_cycle_terminates_and_the_path_is_still_found() -> None:
    graph = json.dumps([["entry", "x", 1], ["x", "entry", 2], ["x", "target", 3]])
    result = CallPathChecker().check(claim(), FakeSession(graph))

    assert result.status is EvidenceStatus.CONFIRMED
    assert (result.summary or "").count(" -> ") == 2


def test_a_recursive_call_witnesses_the_claim() -> None:
    """A function that calls itself is a witnessed one-hop path."""

    result = CallPathChecker().check(
        claim(entrypoint="entry", target="entry"), FakeSession('[["entry", "entry", 4]]')
    )

    assert result.status is EvidenceStatus.CONFIRMED


def test_a_duplicate_edge_resolves_deterministically() -> None:
    graph = '[["entry", "target", 5], ["entry", "target", 6]]'
    result = CallPathChecker().check(claim(), FakeSession(graph))

    assert result.status is EvidenceStatus.CONFIRMED
    assert "test.c:5" in (result.summary or "")


@pytest.mark.parametrize(
    "graph",
    [
        '[["entry", "target"]]',
        '[["entry", "target", 1, 2]]',
        '[{"caller": "entry"}]',
        '[["entry", "target", 0]]',
        '[["entry", "target", -3]]',
        '[["entry", "target", 1.5]]',
        '[["", "target", 1]]',
    ],
)
def test_a_malformed_graph_cannot_confirm(graph: str) -> None:
    result = CallPathChecker().check(claim(), FakeSession(graph))

    assert result.status is EvidenceStatus.CANT_CHECK
    assert result.missing is None


@pytest.mark.parametrize("endpoint", ["--help", "$(id)", "a b", "x\ninjected"])
def test_a_non_identifier_endpoint_never_reaches_the_sandbox(endpoint: str) -> None:
    session = FakeSession(GRAPH)
    result = CallPathChecker().check(claim(entrypoint=endpoint), session)

    assert result.status is EvidenceStatus.CANT_CHECK
    assert result.missing is None
    assert session.calls == []


def test_claim_names_never_reach_a_command() -> None:
    session = FakeSession(GRAPH)
    CallPathChecker().check(claim(), session)

    assert [call[0] for call in session.calls] == ["realpath", "/usr/bin/perl"]
    assert session.calls[-1][-1] == "/src/test.c"


@pytest.mark.parametrize("overrides", [{"file_hint": "../test.c"}, {"file_hint": None}, {"expected_reachable": False}])
def test_out_of_scope_claims_are_not_analyzed(overrides: dict[str, object]) -> None:
    session = FakeSession(GRAPH)
    result = CallPathChecker().check(claim(**overrides), session)
    assert result.status is EvidenceStatus.CANT_CHECK and result.missing is None
    assert session.calls == []
