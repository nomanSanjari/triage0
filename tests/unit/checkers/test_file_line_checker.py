"""Unit tests for FileLineChecker against a fake sandbox; no Docker needed."""

from __future__ import annotations

import pytest

from triage0.checkers.file_line_checker import FileLineChecker
from triage0.core.evidence import (
    Argv,
    CommandResult,
    DeterministicProof,
    EvidenceStatus,
    MissingEvidence,
)
from triage0.core.interfaces import BaseClaimChecker
from triage0.core.models import ClaimKind, FileLineClaim, SymbolClaim

TREE_HASH = "a" * 40

PATH = "lib/dynbuf.c"
TARGET = "/src/lib/dynbuf.c"

# The same line numbers as the real integration anchors, so a failure here is
# easy to compare with curl 8.13.0.
SOURCE = [
    '#include "dynbuf.h"',
    "",
    "CURLcode Curl_dyn_add(struct dynbuf *s, const char *str)",
    "{",
    "  size_t n;",
    "  DEBUGASSERT(str);",
    "  n = strlen(str);",
    "  return dyn_nappend(s, (const unsigned char *)str, n);",
    "}",
]


class FakeSession:
    """Emulates ``sed -n A,Bp -- path`` and ``test -e/-r`` over a checkout."""

    tree_hash = TREE_HASH
    commit = "c" * 40

    def __init__(
        self,
        files: dict[str, list[str]] | None = None,
        *,
        read_exit_code: int = 0,
        read_truncated: bool = False,
        read_timed_out: bool = False,
        resolved_path: str | None = None,
        unreadable: set[str] | None = None,
        probe_exit_code: int | None = None,
        seal_exit_code: int = 0,
        seal_truncated: bool = False,
        seal_timed_out: bool = False,
        seal_stdout: str | None = None,
    ) -> None:
        self.files = {TARGET: SOURCE} if files is None else files
        self.read_exit_code = read_exit_code
        self.read_truncated = read_truncated
        self.read_timed_out = read_timed_out
        self.resolved_path = resolved_path
        self.unreadable = unreadable or set()
        self.probe_exit_code = probe_exit_code
        self.seal_exit_code = seal_exit_code
        self.seal_truncated = seal_truncated
        self.seal_timed_out = seal_timed_out
        self.seal_stdout = seal_stdout
        self.calls: list[tuple[str, ...]] = []
        self.seal_calls: list[tuple[str, ...]] = []

    def _read(
        self,
        argv: Argv,
        *,
        exit_code: int,
        truncated: bool,
        timed_out: bool,
        stdout: str | None,
    ) -> CommandResult:
        path = argv[list(argv).index("--") + 1]
        spec = argv[list(argv).index("-n") + 1]
        first, last = (int(part) for part in spec[:-1].split(","))
        if path not in self.files:
            return CommandResult(
                tool="sed", argv=argv, exit_code=2, stderr="sed: can't read"
            )
        if stdout is None:
            stdout = "".join(line + "\n" for line in self.files[path][first - 1 : last])
        return CommandResult(
            tool="sed",
            argv=argv,
            exit_code=exit_code,
            stdout=stdout,
            truncated=truncated,
            timed_out=timed_out,
        )

    def _test(self, argv: Argv) -> CommandResult:
        """``test -e`` on a listed file, ``test -r`` unless it is unreadable."""

        if self.probe_exit_code is not None:
            return CommandResult(tool="test", argv=argv, exit_code=self.probe_exit_code)
        flag, path = argv[1], argv[2]
        is_dir = path == "/src" or any(name.startswith(path + "/") for name in self.files)
        exists = is_dir or path in self.files
        if flag == "-e":
            answer = exists
        elif flag == "-L":
            answer = False
        elif flag in ("-d", "-x"):
            answer = is_dir and (flag != "-x" or path not in self.unreadable)
        else:
            answer = exists and path not in self.unreadable
        return CommandResult(tool="test", argv=argv, exit_code=0 if answer else 1)

    def run(self, argv: Argv, *, timeout_s: float | None = None) -> CommandResult:
        self.calls.append(tuple(argv))
        if argv[0] == "realpath":
            resolved = self.resolved_path if self.resolved_path is not None else argv[-1]
            return CommandResult(tool="realpath", argv=argv, exit_code=0, stdout=resolved + "\n")
        if argv[0] == "test":
            return self._test(argv)
        return self._read(
            argv,
            exit_code=self.read_exit_code,
            truncated=self.read_truncated,
            timed_out=self.read_timed_out,
            stdout=None,
        )

    def seal(self, argv: Argv, *, timeout_s: float | None = None) -> DeterministicProof:
        self.seal_calls.append(tuple(argv))
        result = self._read(
            argv,
            exit_code=self.seal_exit_code,
            truncated=self.seal_truncated,
            timed_out=self.seal_timed_out,
            stdout=self.seal_stdout,
        )
        return DeterministicProof.from_command_result(
            result, git_tree_hash=self.tree_hash, commit=self.commit
        )


def claim(**overrides: object) -> FileLineClaim:
    data: dict[str, object] = {
        "claim_id": "c1",
        "statement": "line 3 defines Curl_dyn_add",
        "path": PATH,
        "line_start": 3,
    }
    data.update(overrides)
    return FileLineClaim(**data)


def test_supports_only_file_line_claims() -> None:
    checker = FileLineChecker()

    assert isinstance(checker, BaseClaimChecker)
    assert checker.claim_kind is ClaimKind.FILE_LINE
    assert checker.supports(claim()) is True
    assert (
        checker.supports(
            SymbolClaim(claim_id="c2", statement="x", symbol="Curl_dyn_add")
        )
        is False
    )


def test_a_matching_snippet_is_confirmed_with_the_read_proof() -> None:
    session = FakeSession()
    result = FileLineChecker().check(
        claim(
            expected_snippet="CURLcode Curl_dyn_add(struct dynbuf *s, const char *str)"
        ),
        session,
    )

    assert result.status is EvidenceStatus.CONFIRMED
    assert result.proof is not None
    assert result.proof.argv == ("sed", "-n", "3,3p", "--", TARGET)
    assert result.proof.exit_code == 0
    assert not result.proof.truncated
    assert not result.proof.timed_out
    assert "Curl_dyn_add" in (result.proof.stdout or "")
    assert "lib/dynbuf.c:3-3" in (result.summary or "")
    assert len(session.seal_calls) == 1


def test_an_indented_multiline_snippet_survives_the_model_trim() -> None:
    """DomainModel trims the whole snippet, so line 1 loses its indentation."""

    result = FileLineChecker().check(
        claim(
            line_start=5,
            line_end=6,
            expected_snippet="      size_t n;\n    DEBUGASSERT(str);",
        ),
        FakeSession(),
    )

    assert result.status is EvidenceStatus.CONFIRMED


def test_a_fragment_from_the_middle_of_a_line_is_confirmed() -> None:
    result = FileLineChecker().check(
        claim(expected_snippet="Curl_dyn_add"), FakeSession()
    )

    assert result.status is EvidenceStatus.CONFIRMED


def test_a_whitespace_run_inside_a_line_is_collapsed() -> None:
    """The comparison rule is fixed: whitespace runs inside a line collapse."""

    result = FileLineChecker().check(
        claim(
            expected_snippet=(
                "CURLcode   Curl_dyn_add(struct  dynbuf *s, const char *str)"
            )
        ),
        FakeSession(),
    )

    assert result.status is EvidenceStatus.CONFIRMED


def test_line_breaks_are_preserved_so_a_join_does_not_match() -> None:
    """If newlines collapsed to spaces, two lines could match as one."""

    result = FileLineChecker().check(
        claim(line_start=3, line_end=4, expected_snippet="const char *str) {"),
        FakeSession(),
    )

    assert result.status is EvidenceStatus.CANT_CHECK
    assert result.missing is MissingEvidence.CONTENT_DIFFERS


def test_a_snippet_elsewhere_in_the_file_is_not_confirmed() -> None:
    """2.3c never looks outside the claimed range for the code."""

    result = FileLineChecker().check(
        claim(line_start=3, expected_snippet="DEBUGASSERT(str);"), FakeSession()
    )

    assert result.status is EvidenceStatus.CANT_CHECK
    assert result.missing is MissingEvidence.CONTENT_DIFFERS


def test_a_range_without_a_snippet_confirms_the_existing_lines() -> None:
    result = FileLineChecker().check(
        claim(line_start=3, line_end=4), FakeSession()
    )

    assert result.status is EvidenceStatus.CONFIRMED
    assert "exists" in (result.summary or "")


def test_a_single_line_claim_reads_only_that_line() -> None:
    session = FakeSession()
    result = FileLineChecker().check(
        claim(line_start=5, expected_snippet="size_t n;"), session
    )

    assert result.status is EvidenceStatus.CONFIRMED
    assert session.calls[-1] == ("sed", "-n", "5,5p", "--", TARGET)


def test_a_different_snippet_is_content_differs() -> None:
    session = FakeSession()
    result = FileLineChecker().check(
        claim(expected_snippet="CURLcode Curl_dyn_remove(struct dynbuf *s)"), session
    )

    assert result.status is EvidenceStatus.CANT_CHECK
    assert result.missing is MissingEvidence.CONTENT_DIFFERS
    # The summary shows what the lines actually read, without guessing.
    assert "Curl_dyn_add" in (result.summary or "")
    assert session.seal_calls == []


def test_a_mismatch_reads_only_the_claimed_lines() -> None:
    """2.3c never searches for where the code moved; that is history's job."""

    session = FakeSession()
    FileLineChecker().check(claim(expected_snippet="not here"), session)

    assert session.calls == [
        ("realpath", "-m", "--", TARGET),
        ("sed", "-n", "3,3p", "--", TARGET),
    ]
    assert session.seal_calls == []


def test_a_missing_path_is_file_absent() -> None:
    result = FileLineChecker().check(
        claim(path="lib/curl_config.h", line_start=1), FakeSession()
    )

    assert result.status is EvidenceStatus.CANT_CHECK
    assert result.missing is MissingEvidence.FILE_ABSENT


def test_a_present_but_unreadable_path_is_not_file_absent() -> None:
    """The file is there; a history check for a rename would be misled."""

    session = FakeSession(read_exit_code=2, unreadable={TARGET})
    result = FileLineChecker().check(claim(expected_snippet="Curl_dyn_add"), session)

    assert result.status is EvidenceStatus.CANT_CHECK
    assert result.missing is None
    assert "cannot be read" in (result.summary or "")
    assert session.seal_calls == []


def test_a_read_that_failed_for_another_reason_is_not_file_absent() -> None:
    session = FakeSession(read_exit_code=2)
    result = FileLineChecker().check(claim(expected_snippet="Curl_dyn_add"), session)

    assert result.status is EvidenceStatus.CANT_CHECK
    assert result.missing is None
    assert "sed exited 2" in (result.summary or "")
    assert session.seal_calls == []


def test_a_probe_that_cannot_answer_is_not_taken_as_absence() -> None:
    """``test`` exits 2 when it cannot tell; that must not become "missing"."""

    session = FakeSession(read_exit_code=2, probe_exit_code=2)
    result = FileLineChecker().check(claim(path="lib/other.c", line_start=1), session)

    assert result.status is EvidenceStatus.CANT_CHECK
    assert result.missing is None


def test_a_carriage_return_does_not_create_an_extra_sed_line() -> None:
    session = FakeSession({TARGET: ["first\rsecond"]})
    result = FileLineChecker().check(claim(line_start=1, line_end=2), session)

    assert result.status is EvidenceStatus.CANT_CHECK
    assert result.missing is MissingEvidence.LINE_OUT_OF_RANGE


def test_a_possibly_invalid_utf8_read_cannot_confirm() -> None:
    session = FakeSession({TARGET: ["replacement \ufffd"]})
    result = FileLineChecker().check(
        claim(line_start=1, expected_snippet="replacement \ufffd"), session
    )

    assert result.status is EvidenceStatus.CANT_CHECK
    assert result.missing is None
    assert session.seal_calls == []


def test_a_line_past_the_end_is_line_out_of_range() -> None:
    result = FileLineChecker().check(claim(line_start=99), FakeSession())

    assert result.status is EvidenceStatus.CANT_CHECK
    assert result.missing is MissingEvidence.LINE_OUT_OF_RANGE
    assert "no line 99" in (result.summary or "")


def test_a_range_that_runs_past_the_end_is_line_out_of_range() -> None:
    result = FileLineChecker().check(
        claim(line_start=5, line_end=99), FakeSession()
    )

    assert result.status is EvidenceStatus.CANT_CHECK
    assert result.missing is MissingEvidence.LINE_OUT_OF_RANGE
    assert "ends at line 9" in (result.summary or "")


@pytest.mark.parametrize(
    "flags",
    [{"read_truncated": True}, {"read_timed_out": True, "read_exit_code": -9}],
)
def test_an_incomplete_read_is_cant_check_without_a_signal(
    flags: dict[str, object],
) -> None:
    session = FakeSession(**flags)  # type: ignore[arg-type]
    result = FileLineChecker().check(claim(expected_snippet="Curl_dyn_add"), session)

    assert result.status is EvidenceStatus.CANT_CHECK
    assert result.missing is None
    assert session.seal_calls == []


def test_a_blank_snippet_is_refused_before_reading() -> None:
    """``"" in anything`` is True, so a blank snippet must never confirm."""

    session = FakeSession()
    result = FileLineChecker().check(claim(expected_snippet="   "), session)

    assert result.status is EvidenceStatus.CANT_CHECK
    assert result.missing is None
    assert session.calls == []


@pytest.mark.parametrize(
    "path",
    [
        "../../etc/passwd",
        "/etc/passwd",
        "lib\\dynbuf.c",
        "lib/\x00.c",
        "lib/a\tb.c",
        "lib/\x1b[31m.c",
        ".git",
        ".git/config",
    ],
)
def test_an_unusable_path_runs_nothing(path: str) -> None:
    session = FakeSession()
    result = FileLineChecker().check(claim(path=path), session)

    assert result.status is EvidenceStatus.CANT_CHECK
    assert session.calls == []


def test_symlink_resolving_outside_checkout_is_not_read() -> None:
    session = FakeSession(resolved_path="/etc/passwd")
    result = FileLineChecker().check(claim(), session)

    assert result.status is EvidenceStatus.CANT_CHECK
    assert result.missing is None
    assert [call[0] for call in session.calls] == ["realpath"]
    assert session.seal_calls == []


def test_symlink_to_git_administration_is_not_read() -> None:
    session = FakeSession(resolved_path="/src/.git")
    result = FileLineChecker().check(claim(), session)

    assert result.status is EvidenceStatus.CANT_CHECK
    assert result.missing is None
    assert [call[0] for call in session.calls] == ["realpath"]


def test_a_hostile_snippet_never_reaches_a_command() -> None:
    hostile_session = FakeSession()
    FileLineChecker().check(
        claim(expected_snippet="--help; $(id) ' \""), hostile_session
    )

    benign_session = FakeSession()
    FileLineChecker().check(
        claim(expected_snippet="Curl_dyn_add"), benign_session
    )

    # The snippet is only compared in Python, so the read must be identical.
    assert hostile_session.calls == benign_session.calls


def test_a_failed_sealing_read_cannot_confirm() -> None:
    session = FakeSession(seal_exit_code=1)
    result = FileLineChecker().check(
        claim(expected_snippet="Curl_dyn_add"), session
    )

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
def test_an_incomplete_sealing_read_cannot_confirm(
    flags: dict[str, object], note: str
) -> None:
    session = FakeSession(**flags)  # type: ignore[arg-type]
    result = FileLineChecker().check(
        claim(expected_snippet="Curl_dyn_add"), session
    )

    assert result.status is EvidenceStatus.CANT_CHECK
    assert result.proof is not None
    assert note in (result.summary or "")


def test_a_sealing_read_that_lost_the_lines_cannot_confirm() -> None:
    session = FakeSession(seal_stdout="")
    result = FileLineChecker().check(
        claim(expected_snippet="Curl_dyn_add"), session
    )

    assert result.status is EvidenceStatus.CANT_CHECK
    assert "no longer shows" in (result.summary or "")


def test_a_sealing_read_with_possibly_invalid_utf8_cannot_confirm() -> None:
    session = FakeSession(seal_stdout="Curl_dyn_add \ufffd\n")
    result = FileLineChecker().check(claim(expected_snippet="Curl_dyn_add"), session)

    assert result.status is EvidenceStatus.CANT_CHECK
    assert result.missing is None
    assert result.proof is not None


def test_a_sealing_read_with_different_content_cannot_confirm() -> None:
    session = FakeSession(seal_stdout="a different line\n")
    result = FileLineChecker().check(
        claim(expected_snippet="Curl_dyn_add"), session
    )

    assert result.status is EvidenceStatus.CANT_CHECK
    assert result.missing is None
    assert result.proof is not None
    assert "no longer shows" in (result.summary or "")


@pytest.mark.parametrize(
    ("overrides", "expected"),
    [
        ({"path": "lib/curl_config.h", "line_start": 1}, MissingEvidence.FILE_ABSENT),
        ({"line_start": 99}, MissingEvidence.LINE_OUT_OF_RANGE),
        (
            {"expected_snippet": "CURLcode Curl_dyn_remove(struct dynbuf *s)"},
            MissingEvidence.CONTENT_DIFFERS,
        ),
    ],
)
def test_a_miss_never_becomes_a_contradiction(
    overrides: dict[str, object], expected: MissingEvidence
) -> None:
    """The issue's Done-when rule: "not found" is never CONTRADICTED."""

    result = FileLineChecker().check(claim(**overrides), FakeSession())

    assert result.status is EvidenceStatus.CANT_CHECK
    assert result.missing is expected
