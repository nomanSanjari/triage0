"""Unit tests for SymbolChecker against a fake sandbox; no Docker needed."""

from __future__ import annotations

import json
from collections.abc import Callable

import pytest

from triage0.checkers.symbol_checker import SymbolChecker
from triage0.core.evidence import (
    Argv,
    CommandResult,
    DeterministicProof,
    EvidenceStatus,
    MissingEvidence,
)
from triage0.core.interfaces import BaseClaimChecker
from triage0.core.models import ClaimKind, FileLineClaim, SymbolClaim, SymbolKind

TREE_HASH = "a" * 40


def _tag(name: str, path: str, line: int, kind: str) -> dict[str, object]:
    return {"_type": "tag", "name": name, "path": path, "line": line, "kind": kind}


class FakeSession:
    """Serves an index and its matching syntax captures unless overridden."""

    tree_hash = TREE_HASH
    commit = "c" * 40

    def __init__(
        self,
        files: list[str],
        tags: list[dict[str, object]],
        *,
        truncated: Callable[[list[str]], bool] | None = None,
        find_exit_code: int = 0,
        find_timed_out: bool = False,
        seal_exit_code: int = 0,
        seal_timed_out: bool = False,
        seal_truncated: bool = False,
        seal_stdout: str | None = None,
        resolved_path: str | None = None,
    ) -> None:
        self.files = files
        self.tags = tags
        self.truncated = truncated or (lambda chunk: False)
        self.find_exit_code = find_exit_code
        self.find_timed_out = find_timed_out
        self.seal_exit_code = seal_exit_code
        self.seal_timed_out = seal_timed_out
        self.seal_truncated = seal_truncated
        self.seal_stdout = seal_stdout
        self.resolved_path = resolved_path
        self.calls: list[tuple[str, ...]] = []
        self.seal_calls: list[tuple[str, ...]] = []

    def run(self, argv: Argv, *, timeout_s: float | None = None) -> CommandResult:
        self.calls.append(tuple(argv))
        if argv[0] == "find":
            return CommandResult(
                tool="find",
                argv=argv,
                exit_code=self.find_exit_code,
                stdout="".join(f + "\x00" for f in self.files),
                timed_out=self.find_timed_out,
            )
        if argv[0] == "realpath":
            return CommandResult(
                tool="realpath", argv=argv, exit_code=0,
                stdout=(self.resolved_path or argv[-1]) + "\n",
            )
        assert argv[0] == "ctags"
        targets = list(argv[list(argv).index("-") + 1 :])
        lines = [json.dumps(tag) for tag in self.tags if tag["path"] in targets]
        return CommandResult(
            tool="ctags",
            argv=argv,
            exit_code=0,
            stdout="\n".join(lines),
            truncated=self.truncated(targets),
        )

    def seal(self, argv: Argv, *, timeout_s: float | None = None) -> DeterministicProof:
        self.seal_calls.append(tuple(argv))
        assert argv[:2] == ("tree-sitter", "query")
        if self.seal_stdout is not None:
            stdout = self.seal_stdout
        else:
            stdout = argv[-1] + "\n" + "".join(
                _syntax_match(tag) for tag in self.tags if tag["path"] == argv[-1]
            )
        result = CommandResult(
            tool="tree-sitter",
            argv=argv,
            exit_code=self.seal_exit_code,
            stdout=stdout,
            timed_out=self.seal_timed_out,
            truncated=self.seal_truncated,
        )
        return DeterministicProof.from_command_result(
            result, git_tree_hash=self.tree_hash, commit=self.commit
        )


def _syntax_match(tag: dict[str, object]) -> str:
    line = tag["line"]
    name = tag["name"]
    assert isinstance(line, int) and isinstance(name, str)
    row = line - 1
    return (
        f"  pattern: 0\n    capture: {tag['kind']}, start: ({row}, 0), end: ({row + 1}, 0)\n"
        f"    capture: 0 - name, start: ({row}, 4), end: ({row}, {4 + len(name)}), text: `{name}`\n"
    )


def claim(**overrides: object) -> SymbolClaim:
    data: dict[str, object] = {
        "claim_id": "c1",
        "statement": "Curl_dyn_add exists",
        "symbol": "Curl_dyn_add",
        "symbol_kind": SymbolKind.FUNCTION,
        "file_hint": "lib/dynbuf.c",
    }
    data.update(overrides)
    return SymbolClaim(**data)


FUNCTION_TAG = _tag("Curl_dyn_add", "/src/lib/dynbuf.c", 180, "function")


def test_supports_only_symbol_claims() -> None:
    checker = SymbolChecker()
    assert isinstance(checker, BaseClaimChecker)
    assert checker.claim_kind is ClaimKind.SYMBOL
    assert checker.supports(claim()) is True
    assert (
        checker.supports(
            FileLineClaim(claim_id="c2", statement="x", path="lib/dynbuf.c", line_start=1)
        )
        is False
    )


def test_found_with_matching_kind_is_confirmed() -> None:
    session = FakeSession(["/src/lib/dynbuf.c"], [FUNCTION_TAG])
    result = SymbolChecker().check(claim(), session)

    assert result.status is EvidenceStatus.CONFIRMED
    assert result.proof is not None
    assert "lib/dynbuf.c:180" in (result.summary or "")
    assert result.proof.tool == "tree-sitter"
    # The decisive command is the one file, not the whole index.
    assert len(session.seal_calls) == 1
    assert session.seal_calls[0][-1] == "/src/lib/dynbuf.c"


def test_unknown_symbol_kind_accepts_a_found_symbol() -> None:
    session = FakeSession(["/src/lib/dynbuf.c"], [FUNCTION_TAG])
    result = SymbolChecker().check(claim(symbol_kind=SymbolKind.UNKNOWN), session)
    assert result.status is EvidenceStatus.CONFIRMED


def test_found_while_claimed_absent_is_contradicted() -> None:
    session = FakeSession(["/src/lib/dynbuf.c"], [FUNCTION_TAG])
    result = SymbolChecker().check(claim(expected_present=False), session)

    assert result.status is EvidenceStatus.CONTRADICTED
    assert result.proof is not None
    assert "cannot exist" in (result.summary or "")


def test_kind_mismatch_is_cant_check_not_contradicted() -> None:
    session = FakeSession(
        ["/src/lib/dynbuf.c"], [_tag("Curl_dyn_add", "/src/lib/dynbuf.c", 9, "macro")]
    )
    result = SymbolChecker().check(claim(), session)

    assert result.status is EvidenceStatus.CANT_CHECK
    assert result.missing is None
    assert "macro" in (result.summary or "")
    assert session.seal_calls == []


def test_claimed_missing_but_only_another_kind_has_the_name_is_cant_check() -> None:
    """A macro with the same name is not the function the claim says is absent."""

    session = FakeSession(
        ["/src/lib/dynbuf.c"], [_tag("Curl_dyn_add", "/src/lib/dynbuf.c", 9, "macro")]
    )
    result = SymbolChecker().check(claim(expected_present=False), session)

    assert result.status is EvidenceStatus.CANT_CHECK
    assert result.missing is None
    assert "macro" in (result.summary or "")
    assert session.seal_calls == []


def test_claimed_missing_cites_the_matching_kind_not_the_hint() -> None:
    tags = [
        _tag("Curl_dyn_add", "/src/lib/a.c", 1, "macro"),
        _tag("Curl_dyn_add", "/src/lib/b.c", 2, "function"),
    ]
    session = FakeSession(["/src/lib/a.c", "/src/lib/b.c"], tags)
    result = SymbolChecker().check(
        claim(expected_present=False, file_hint="lib/a.c"), session
    )

    assert result.status is EvidenceStatus.CONTRADICTED
    assert session.seal_calls[0][-1] == "/src/lib/b.c"


def test_claimed_missing_with_an_unknown_kind_still_contradicts() -> None:
    """Without a kind from the report, a matching definition of a supported kind contradicts it."""

    session = FakeSession(
        ["/src/lib/dynbuf.c"], [_tag("Curl_dyn_add", "/src/lib/dynbuf.c", 9, "macro")]
    )
    result = SymbolChecker().check(
        claim(expected_present=False, symbol_kind=SymbolKind.UNKNOWN), session
    )

    assert result.status is EvidenceStatus.CONTRADICTED


def test_not_found_carries_the_symbol_absent_signal() -> None:
    session = FakeSession(["/src/lib/dynbuf.c"], [])
    result = SymbolChecker().check(claim(), session)

    assert result.status is EvidenceStatus.CANT_CHECK
    assert result.missing is MissingEvidence.SYMBOL_ABSENT
    assert "indexed 1 C files under /src" in (result.summary or "")
    assert session.seal_calls == []


def test_claimed_absent_and_truly_absent_is_cant_check() -> None:
    """You cannot seal a negative, so this must never become CONFIRMED."""

    session = FakeSession(["/src/lib/dynbuf.c"], [])
    result = SymbolChecker().check(claim(expected_present=False), session)

    assert result.status is EvidenceStatus.CANT_CHECK
    assert result.missing is MissingEvidence.SYMBOL_ABSENT


def test_a_truncated_index_never_reports_absence() -> None:
    """Valid JSON is not proof of completeness; the flag is."""

    session = FakeSession(["/src/lib/dynbuf.c"], [FUNCTION_TAG], truncated=lambda chunk: True)
    result = SymbolChecker().check(claim(), session)

    assert result.status is EvidenceStatus.CANT_CHECK
    assert result.missing is None
    assert session.seal_calls == []


def test_no_c_sources_is_cant_check_not_absence() -> None:
    session = FakeSession([], [])
    result = SymbolChecker().check(claim(), session)
    assert result.status is EvidenceStatus.CANT_CHECK
    assert result.missing is None


@pytest.mark.parametrize(
    "flags",
    [{"find_exit_code": 2}, {"find_timed_out": True, "find_exit_code": -9}],
)
def test_an_index_that_never_finished_is_cant_check_without_a_signal(
    flags: dict[str, object],
) -> None:
    """A failed listing is an analysis limit, not a rename to look up."""

    session = FakeSession(["/src/lib/dynbuf.c"], [], **flags)  # type: ignore[arg-type]
    result = SymbolChecker().check(claim(), session)

    assert result.status is EvidenceStatus.CANT_CHECK
    assert result.missing is None
    assert "could not index" in (result.summary or "")
    assert session.seal_calls == []


def test_an_escaping_file_hint_runs_nothing() -> None:
    session = FakeSession(["/src/lib/dynbuf.c"], [FUNCTION_TAG])
    result = SymbolChecker().check(claim(file_hint="../../etc/passwd"), session)

    assert result.status is EvidenceStatus.CANT_CHECK
    assert session.calls == []


def test_a_symbol_found_outside_the_hinted_file_is_still_confirmed() -> None:
    """A hint must never narrow the search into a false absence."""

    session = FakeSession(["/src/lib/dynbuf.c"], [FUNCTION_TAG])
    result = SymbolChecker().check(claim(file_hint="lib/elsewhere.c"), session)

    assert result.status is EvidenceStatus.CONFIRMED
    assert "lib/dynbuf.c:180" in (result.summary or "")
    assert "the claim points at" in (result.summary or "")


def test_the_hinted_definition_is_cited_when_several_match() -> None:
    tags = [
        _tag("Curl_dyn_add", "/src/lib/a.c", 10, "function"),
        _tag("Curl_dyn_add", "/src/lib/b.c", 20, "function"),
    ]
    session = FakeSession(["/src/lib/a.c", "/src/lib/b.c"], tags)
    result = SymbolChecker().check(claim(file_hint="lib/b.c"), session)

    assert result.status is EvidenceStatus.CONFIRMED
    assert "lib/b.c:20" in (result.summary or "")
    assert session.seal_calls[0][-1] == "/src/lib/b.c"


def test_a_failed_sealing_run_cannot_confirm() -> None:
    """A proof whose command failed is not proof, whatever it printed."""

    session = FakeSession(["/src/lib/dynbuf.c"], [FUNCTION_TAG], seal_exit_code=1)
    result = SymbolChecker().check(claim(), session)

    assert result.status is EvidenceStatus.CANT_CHECK
    assert "exited 1" in (result.summary or "")
    assert result.proof is not None
    assert result.proof.exit_code == 1


@pytest.mark.parametrize(
    ("flags", "note"),
    [
        ({"seal_timed_out": True, "seal_exit_code": -9}, "timed out"),
        ({"seal_truncated": True}, "truncated"),
    ],
)
def test_an_incomplete_sealing_run_cannot_confirm(
    flags: dict[str, object], note: str
) -> None:
    session = FakeSession(["/src/lib/dynbuf.c"], [FUNCTION_TAG], **flags)  # type: ignore[arg-type]
    result = SymbolChecker().check(claim(), session)

    assert result.status is EvidenceStatus.CANT_CHECK
    assert result.proof is not None
    # A timeout must be reported as one, not as a command that "exited -9".
    assert note in (result.summary or "")


@pytest.mark.parametrize(
    "syntax_tag",
    [
        _tag("Curl_dyn_adder", "/src/lib/dynbuf.c", 180, "function"),
        _tag("Curl_dyn_add", "/src/lib/dynbuf.c", 180, "macro"),
        _tag("Curl_dyn_add", "/src/lib/dynbuf.c", 179, "function"),
        _tag("Curl_dyn_add", "/src/lib/other.c", 180, "function"),
    ],
    ids=["name", "kind", "line", "file"],
)
def test_syntax_capture_must_bind_all_four_tag_fields(syntax_tag: dict[str, object]) -> None:
    stdout = f"{syntax_tag['path']}\n" + _syntax_match(syntax_tag)
    session = FakeSession(["/src/lib/dynbuf.c"], [FUNCTION_TAG], seal_stdout=stdout)
    result = SymbolChecker().check(claim(expected_present=False), session)

    assert result.status is EvidenceStatus.CANT_CHECK
    assert result.missing is None
    assert "did not uniquely confirm" in (result.summary or "")


@pytest.mark.parametrize(
    "capture",
    [
        "    capture: error, start: (179, 0), end: (181, 0)\n",
        "    capture: 8 - missing, start: (180, 0), end: (180, 0), text: ``\n",
    ],
    ids=["overlapping-error", "missing-final-token"],
)
def test_local_parser_recovery_blocks_a_terminal_verdict(capture: str) -> None:
    stdout = "/src/lib/dynbuf.c\n" + _syntax_match(FUNCTION_TAG) + "  pattern: 1\n" + capture
    session = FakeSession(["/src/lib/dynbuf.c"], [FUNCTION_TAG], seal_stdout=stdout)
    assert SymbolChecker().check(claim(), session).status is EvidenceStatus.CANT_CHECK


def test_duplicate_index_entries_are_not_unique_candidates() -> None:
    session = FakeSession(["/src/lib/dynbuf.c"], [FUNCTION_TAG, FUNCTION_TAG])
    assert SymbolChecker().check(claim(), session).status is EvidenceStatus.CANT_CHECK
    assert session.seal_calls == []


def test_ambiguous_syntax_matches_do_not_confirm() -> None:
    match = _syntax_match(FUNCTION_TAG)
    session = FakeSession(["/src/lib/dynbuf.c"], [FUNCTION_TAG], seal_stdout=(
        "/src/lib/dynbuf.c\n" + match + match.replace("(179, 4)", "(179, 5)")
    ))
    assert SymbolChecker().check(claim(), session).status is EvidenceStatus.CANT_CHECK


@pytest.mark.parametrize(
    ("path", "resolved"),
    [("/src/ambiguous*.c", None), ("/src/lib/dynbuf.c", "/etc/passwd")],
    ids=["glob", "symlink-escape"],
)
def test_only_an_exact_path_in_the_checkout_reaches_the_query(
    path: str, resolved: str | None
) -> None:
    session = FakeSession([path], [_tag("Curl_dyn_add", path, 180, "function")],
                          resolved_path=resolved)
    result = SymbolChecker().check(claim(), session)
    assert result.status is EvidenceStatus.CANT_CHECK
    assert session.seal_calls == []


@pytest.mark.parametrize(
    "symbol",
    ["--output=/tmp/pwn", "a b", "$(id)", "'; rm -rf /", "-R", "x\ninjected"],
)
def test_hostile_symbol_names_never_reach_a_command(symbol: str) -> None:
    hostile = FakeSession(["/src/lib/dynbuf.c"], [])
    SymbolChecker().check(claim(symbol=symbol), hostile)

    benign = FakeSession(["/src/lib/dynbuf.c"], [])
    SymbolChecker().check(claim(symbol="Curl_dyn_add"), benign)

    assert hostile.calls, "the checker should still search"
    # The name is only compared in Python, so the commands must be identical:
    # a name like "-R" is our own flag, and one that leaks would change argv.
    assert hostile.calls == benign.calls
