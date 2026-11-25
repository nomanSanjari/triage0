"""Unit tests for the ctags layer; no Docker needed."""

from __future__ import annotations

import json
from collections.abc import Callable

import pytest

from triage0.checkers.tools.ast_tools import (
    BadPathError,
    LineReadStatus,
    TagParseError,
    ctags_argv,
    find_tags,
    line_argv,
    parse_tags,
    read_lines,
    sandbox_target,
    symbol_kind,
)
from triage0.core.evidence import Argv, CommandResult, DeterministicProof
from triage0.core.models import SymbolKind

TREE_HASH = "a" * 40


def _tag(
    name: str = "f", path: str = "/src/lib/x.c", line: int = 3, kind: str = "function"
) -> dict[str, object]:
    return {"_type": "tag", "name": name, "path": path, "line": line, "kind": kind}


class FakeSession:
    """Serves a file listing, ctags output, path probes, and ``sed`` reads."""

    tree_hash = TREE_HASH
    commit = "c" * 40

    def __init__(
        self,
        files: list[str],
        *,
        tags: list[dict[str, object]] | None = None,
        ctags_stdout: str | None = None,
        ctags_stderr: str = "",
        truncated: Callable[[list[str]], bool] | None = None,
        exit_code: int = 0,
        timed_out: bool = False,
        find_truncated: bool = False,
        find_exit_code: int = 0,
        find_timed_out: bool = False,
        fail_after: int | None = None,
        content: dict[str, list[str]] | None = None,
        sed_truncated: bool = False,
        sed_timed_out: bool = False,
        sed_exit_code: int = 0,
        unreadable: set[str] | None = None,
        probe_exit_code: int | None = None,
    ) -> None:
        self.files = files
        self.tags = tags or []
        self.ctags_stdout = ctags_stdout
        self.ctags_stderr = ctags_stderr
        self.truncated = truncated or (lambda chunk: False)
        self.exit_code = exit_code
        self.timed_out = timed_out
        self.find_truncated = find_truncated
        self.find_exit_code = find_exit_code
        self.find_timed_out = find_timed_out
        self.fail_after = fail_after
        self.content = content or {}
        self.sed_truncated = sed_truncated
        self.sed_timed_out = sed_timed_out
        self.sed_exit_code = sed_exit_code
        self.unreadable = unreadable or set()
        self.probe_exit_code = probe_exit_code
        self.ctags_runs = 0
        self.calls: list[tuple[str, ...]] = []

    def run(self, argv: Argv, *, timeout_s: float | None = None) -> CommandResult:
        self.calls.append(tuple(argv))
        if argv[0] == "find":
            listed = "".join(name + "\x00" for name in self.files)
            return CommandResult(
                tool="find",
                argv=argv,
                exit_code=self.find_exit_code,
                stdout=listed,
                truncated=self.find_truncated,
                timed_out=self.find_timed_out,
            )
        if argv[0] == "sed":
            return self._sed(argv)
        if argv[0] == "test":
            return self._test(argv)
        targets = list(argv[list(argv).index("-") + 1 :])
        lines = [json.dumps(tag) for tag in self.tags if tag["path"] in targets]
        failing = self.fail_after is not None and self.ctags_runs >= self.fail_after
        self.ctags_runs += 1
        return CommandResult(
            tool="ctags",
            argv=argv,
            exit_code=2 if failing else self.exit_code,
            stdout=self.ctags_stdout if self.ctags_stdout is not None else "\n".join(lines),
            stderr=self.ctags_stderr,
            truncated=self.truncated(targets),
            timed_out=self.timed_out,
        )

    def _sed(self, argv: Argv) -> CommandResult:
        """Emulate GNU sed: a missing file exits 2, past-EOF prints nothing."""

        path = argv[list(argv).index("--") + 1]
        spec = argv[list(argv).index("-n") + 1]
        first, last = (int(part) for part in spec[:-1].split(","))
        if path not in self.content or path in self.unreadable:
            return CommandResult(
                tool="sed", argv=argv, exit_code=2, stderr="sed: can't read"
            )
        selected = self.content[path][first - 1 : last]
        return CommandResult(
            tool="sed",
            argv=argv,
            exit_code=-9 if self.sed_timed_out else self.sed_exit_code,
            stdout="".join(line + "\n" for line in selected),
            timed_out=self.sed_timed_out,
            truncated=self.sed_truncated,
        )

    def _test(self, argv: Argv) -> CommandResult:
        """Emulate ``test -e`` / ``test -r``; ``unreadable`` paths exist only."""

        if self.probe_exit_code is not None:
            return CommandResult(tool="test", argv=argv, exit_code=self.probe_exit_code)
        flag, path = argv[1], argv[2]
        is_dir = path == "/src" or any(name.startswith(path + "/") for name in self.content)
        exists = is_dir or path in self.content or path in self.unreadable
        if flag == "-e":
            answer = exists
        elif flag == "-L":
            answer = False
        elif flag in ("-d", "-x"):
            answer = is_dir and (flag != "-x" or path not in self.unreadable)
        else:
            answer = exists and path not in self.unreadable
        return CommandResult(tool="test", argv=argv, exit_code=0 if answer else 1)

    def seal(self, argv: Argv, *, timeout_s: float | None = None) -> DeterministicProof:
        return DeterministicProof.from_command_result(
            self.run(argv, timeout_s=timeout_s), git_tree_hash=self.tree_hash, commit=self.commit
        )


def _ctags_calls(session: FakeSession) -> list[tuple[str, ...]]:
    return [call for call in session.calls if call[0] == "ctags"]


def test_parse_tags_reads_json_lines() -> None:
    stdout = "\n".join([json.dumps(_tag()), json.dumps(_tag(name="g", kind="macro"))])
    tags = parse_tags(stdout)
    assert [(tag.name, tag.kind, tag.line) for tag in tags] == [
        ("f", "function", 3),
        ("g", "macro", 3),
    ]


def test_parse_tags_rejects_a_line_that_is_not_json() -> None:
    with pytest.raises(TagParseError, match="not JSON"):
        parse_tags('ctags: Warning: something\n{"name": "f"}')


@pytest.mark.parametrize(
    "broken",
    [
        {"name": "", "path": "/src/x.c", "line": 4, "kind": "function"},
        {"name": "f", "path": "/src/x.c", "line": 4, "kind": ""},
        {"name": "f", "path": "/src/x.c", "line": 0, "kind": "function"},
        {"name": "f", "path": "/src/x.c", "line": True, "kind": "function"},
        # The path may be sent to tree-sitter, so it must stay in the
        # mounted worktree, and it must never smuggle control characters
        # into a summary a report renderer will print.
        {"name": "f", "path": "/etc/passwd", "line": 4, "kind": "function"},
        {"name": "f", "path": "/src/../etc/passwd", "line": 4, "kind": "function"},
        {"name": "f", "path": "/src/lib/a\nb.c", "line": 4, "kind": "function"},
        {"name": "f", "path": "/src/lib/a\tb.c", "line": 4, "kind": "function"},
        {"name": "f", "path": "/src/lib/a\x00b.c", "line": 4, "kind": "function"},
    ],
)
def test_parse_tags_rejects_unusable_fields(broken: dict[str, object]) -> None:
    with pytest.raises(TagParseError, match="unusable name/path/kind/line"):
        parse_tags(json.dumps(broken))


def test_parse_tags_rejects_a_tag_without_a_line() -> None:
    with pytest.raises(TagParseError, match="unusable name/path/kind/line"):
        parse_tags(json.dumps({"_type": "tag", "name": "f", "path": "/src/x.c", "kind": "function"}))


@pytest.mark.parametrize(
    ("ctags_kind", "expected"),
    [
        ("function", SymbolKind.FUNCTION),
        ("macro", SymbolKind.MACRO),
        ("struct", SymbolKind.STRUCT),
        ("typedef", SymbolKind.TYPE),
        ("variable", SymbolKind.VARIABLE),
        ("member", None),
        ("enumerator", None),
    ],
)
def test_symbol_kind_maps_the_shared_enum(ctags_kind: str, expected: SymbolKind | None) -> None:
    assert symbol_kind(ctags_kind) is expected


@pytest.mark.parametrize(
    "hint",
    [
        "../etc/passwd",
        "lib/../../etc/passwd",
        "/etc/passwd",
        "~/.ssh/id_rsa",
        "lib\\dynbuf.c",
        "",
        "a/../../b",
        # A path reaches both a command argument and a summary, so control
        # characters are refused even when they stay inside /src.
        "lib/a\tb.c",
        "lib/\x1b[31m.c",
    ],
)
def test_sandbox_target_rejects_unusable_paths(hint: str) -> None:
    with pytest.raises(BadPathError):
        sandbox_target(hint)


@pytest.mark.parametrize(
    ("hint", "expected"),
    [
        ("lib/dynbuf.c", "/src/lib/dynbuf.c"),
        ("./lib/dynbuf.c", "/src/lib/dynbuf.c"),
        ("include/curl", "/src/include/curl"),
    ],
)
def test_sandbox_target_prefixes_the_worktree(hint: str, expected: str) -> None:
    assert sandbox_target(hint) == expected


def test_ctags_argv_keeps_the_output_on_stdout() -> None:
    argv = ctags_argv(["/src/a.c"])
    assert argv[:1] == ["ctags"]
    assert argv[-1] == "/src/a.c"
    assert argv[argv.index("-o") + 1] == "-"


def test_line_argv_puts_the_path_behind_a_double_dash() -> None:
    assert line_argv("/src/lib/-n", 3, 5) == [
        "sed",
        "-n",
        "3,5p",
        "--",
        "/src/lib/-n",
    ]


def test_read_lines_returns_the_requested_range() -> None:
    session = FakeSession(
        [], content={"/src/lib/x.c": ["one", "two", "three", "four"]}
    )
    read = read_lines(session, "/src/lib/x.c", 2, 3)

    assert read.status is LineReadStatus.OK
    assert read.lines == ("two", "three")


def test_read_lines_past_the_end_is_an_empty_ok_read() -> None:
    """GNU sed exits 0 with no output past EOF; the checker decides range."""

    session = FakeSession([], content={"/src/lib/x.c": ["one"]})
    read = read_lines(session, "/src/lib/x.c", 5, 6)

    assert read.status is LineReadStatus.OK
    assert read.lines == ()


def test_read_lines_on_a_missing_path_is_absent() -> None:
    read = read_lines(FakeSession([]), "/src/nope.c", 1, 1)

    assert read.status is LineReadStatus.ABSENT
    assert read.lines == ()
    assert "does not exist" in read.detail


def test_a_present_but_unreadable_path_is_not_absent() -> None:
    """sed fails the same way for both, and only one of them is absence."""

    session = FakeSession([], content={"/src/x.c": ["a"]}, unreadable={"/src/x.c"})
    read = read_lines(session, "/src/x.c", 1, 1)

    assert read.status is LineReadStatus.UNREADABLE


def test_an_inaccessible_parent_does_not_imply_absence() -> None:
    # test -e fails both for a missing file and an unsearchable parent.
    session = FakeSession(
        [], content={"/src/private/other.c": ["a"]}, unreadable={"/src/private"}
    )
    read = read_lines(session, "/src/private/x.c", 1, 1)

    assert read.status is LineReadStatus.FAILED


def test_a_read_failure_that_is_not_a_missing_path_is_not_absent() -> None:
    session = FakeSession([], content={"/src/x.c": ["a"]}, sed_exit_code=2)
    read = read_lines(session, "/src/x.c", 1, 1)

    assert read.status is LineReadStatus.FAILED
    assert "sed exited 2" in read.detail


def test_a_probe_that_cannot_answer_is_not_taken_as_absence() -> None:
    """``test`` exits 2 when it cannot tell; that must not become "missing"."""

    read = read_lines(FakeSession([], probe_exit_code=2), "/src/nope.c", 1, 1)

    assert read.status is LineReadStatus.FAILED


def test_a_truncated_read_is_incomplete_not_unreadable() -> None:
    session = FakeSession([], content={"/src/x.c": ["a"]}, sed_truncated=True)
    read = read_lines(session, "/src/x.c", 1, 1)

    assert read.status is LineReadStatus.INCOMPLETE


def test_a_timed_out_read_is_incomplete_not_unreadable() -> None:
    """A timeout reports a negative exit code; that must not look like absence."""

    session = FakeSession([], content={"/src/x.c": ["a"]}, sed_timed_out=True)
    read = read_lines(session, "/src/x.c", 1, 1)

    assert read.status is LineReadStatus.INCOMPLETE


def test_find_tags_reports_no_c_sources_as_incomplete() -> None:
    search = find_tags(FakeSession([]), "f")
    assert search.complete is False
    assert search.tags == ()


def test_find_tags_matches_the_name_exactly() -> None:
    tags = [_tag(name="f"), _tag(name="fx"), _tag(name="F")]
    search = find_tags(FakeSession(["/src/lib/x.c"], tags=tags), "f")
    assert [tag.name for tag in search.tags] == ["f"]
    assert search.complete is True


def test_a_truncated_chunk_is_split_until_it_fits() -> None:
    files = ["/src/a.c", "/src/b.c", "/src/c.c", "/src/d.c"]
    tags = [_tag(name="f", path="/src/b.c", line=7)]
    session = FakeSession(files, tags=tags, truncated=lambda chunk: len(chunk) > 2)

    search = find_tags(session, "f")

    assert search.complete is True
    assert [(tag.path, tag.line) for tag in search.tags] == [("/src/b.c", 7)]
    # One 4-file run truncated, then two 2-file runs over the same files.
    targets = [list(call[list(call).index("-") + 1 :]) for call in _ctags_calls(session)]
    assert targets == [files, files[:2], files[2:]]


def test_a_single_file_that_still_truncates_is_incomplete() -> None:
    """A huge file cannot be half-searched, and valid JSON is not proof."""

    session = FakeSession(["/src/a.c"], tags=[_tag()], truncated=lambda chunk: True)
    search = find_tags(session, "f")
    assert search.complete is False
    assert search.tags == ()
    assert "truncated" in search.detail


def test_a_truncated_file_listing_is_incomplete() -> None:
    """An unfinished listing must never look like an empty repository."""

    search = find_tags(FakeSession(["/src/a.c"], find_truncated=True), "f")
    assert search.complete is False
    assert search.tags == ()


def test_a_failed_file_listing_is_incomplete() -> None:
    search = find_tags(FakeSession(["/src/a.c"], find_exit_code=1), "f")
    assert search.complete is False
    assert "could not list" in search.detail


def test_a_timed_out_file_listing_is_incomplete() -> None:
    session = FakeSession(["/src/a.c"], find_timed_out=True)
    search = find_tags(session, "f")
    assert search.complete is False
    assert "could not list" in search.detail


def test_malformed_ctags_output_makes_the_search_incomplete() -> None:
    """A cut-off or garbled line means "cannot see everything", not "no tags"."""

    session = FakeSession(["/src/a.c"], ctags_stdout='{"name": ')
    search = find_tags(session, "f")

    assert search.complete is False
    assert search.tags == ()
    assert "not JSON" in search.detail


def test_a_timed_out_ctags_run_is_incomplete() -> None:
    session = FakeSession(["/src/a.c"], tags=[_tag()], timed_out=True)
    search = find_tags(session, "f")
    assert search.complete is False
    assert "failed" in search.detail


def test_ctags_failure_is_incomplete_not_absent() -> None:
    session = FakeSession(["/src/a.c"], exit_code=2)
    search = find_tags(session, "f")
    assert search.complete is False
    assert "failed" in search.detail


def test_ctags_warning_makes_the_search_incomplete_even_on_exit_zero() -> None:
    session = FakeSession(["/src/a.c"], ctags_stderr="ctags: Warning: cannot open input file")
    search = find_tags(session, "f")

    assert search.complete is False
    assert search.tags == ()
    assert "warning" in search.detail


def test_duplicate_names_across_chunks_are_all_kept() -> None:
    files = [f"/src/f{index:03d}.c" for index in range(600)]
    tags = [
        _tag(name="f", path="/src/f000.c", line=1),
        _tag(name="f", path="/src/f300.c", line=2),
        _tag(name="f", path="/src/f599.c", line=3),
    ]
    session = FakeSession(files, tags=tags)
    search = find_tags(session, "f")

    assert search.complete is True
    assert [tag.path for tag in search.tags] == ["/src/f000.c", "/src/f300.c", "/src/f599.c"]
    chunks = [call[call.index("-") + 1 :] for call in _ctags_calls(session)]
    assert [len(chunk) for chunk in chunks] == [256, 256, 88]


def test_a_later_failing_batch_reports_how_far_the_index_got() -> None:
    """The count is the only diagnostic a partial search leaves behind."""

    files = [f"/src/f{index:03d}.c" for index in range(600)]
    session = FakeSession(files, fail_after=1)
    search = find_tags(session, "f")

    assert search.complete is False
    assert "ctags failed on /src/f256.c (exit 2)" in search.detail
    assert "256 C files indexed so far" in search.detail


def test_the_symbol_never_reaches_the_command_line() -> None:
    session = FakeSession(["/src/a.c"], tags=[_tag(name="f")])
    hostile = "--output=/tmp/pwn; $(id) ' \""
    find_tags(session, hostile)
    for call in session.calls:
        assert all(hostile not in argument for argument in call)
