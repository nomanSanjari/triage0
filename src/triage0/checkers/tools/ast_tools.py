"""C symbol indexing and exact line reads, run inside the sandbox.

The symbol name is untrusted report text. It never becomes an argument: it is
only compared with tag names in Python, so a name containing spaces, quotes,
shell metacharacters or a leading "-" has no command to influence. The only
report-supplied value that reaches an argv is a path, and it is checked
against ``/src`` first (see ``sandbox_target``).

The file/line reader is the other half of this module. ``read_lines`` prints
an exact range with ``sed`` behind ``--``, so a path can never be read as an
option, and the read happens inside the sandbox: nothing here opens a worktree
file from the host. A read that timed out or hit the output cap is reported
as incomplete, never as an absent file.

Output is capped at 1 MiB per run by the sandbox, and a ctags index of a real
project is larger than that (curl 8.13.0 is 2.4 MiB / 12k tags). Truncated
output is detected through ``CommandResult.truncated``, never inferred from a
JSON parse: a cut after a complete JSON line still parses fine. Files are
indexed in bounded chunks, and a chunk that still truncates is split in half,
so the whole tree is read or the search is reported as incomplete.

Absence is only meaningful when the search was complete, and even then it is
a ``CANT_CHECK`` with a missing signal -- never a contradiction. For the same
reason the search is always the whole checkout: a report-supplied file hint
picks which definition to cite, it never narrows the index, because "not in
that one file" is not "not in the repository".

The file list comes from one ``find`` run, which is bounded by the same 1 MiB
cap. A checkout too large to enumerate that way is reported as an incomplete
search rather than as an empty one.
"""

from __future__ import annotations

import json
import posixpath
from dataclasses import dataclass
from enum import StrEnum

from triage0.core.evidence import Argv, CommandResult, DeterministicProof
from triage0.core.interfaces import SandboxSession
from triage0.core.models import SymbolKind
from triage0.sandbox.providers.docker_local import SANDBOX_SRC

__all__ = [
    "BadPathError",
    "LineRead",
    "LineReadStatus",
    "Tag",
    "TagIndexError",
    "TagParseError",
    "TagSearch",
    "ctags_argv",
    "find_tags",
    "line_argv",
    "output_lines",
    "parse_tags",
    "read_lines",
    "resolve_target",
    "sandbox_target",
    "seal_lines",
    "symbol_kind",
]

CTAGS_TIMEOUT_S = 120

# A line read is one sed invocation over a bounded range; it should finish in
# milliseconds. The limit only exists so a pathological file cannot hang the
# checker, and a timeout is reported as an incomplete read, not absence.
READ_TIMEOUT_S = 30

# Files per ctags run. On curl 8.13.0 one 256-file chunk is 0.3-1.1 MB of
# JSON, so the cap is reached in normal use and that chunk is halved; a
# larger batch would cost extra round trips without ever buying correctness.
_FILES_PER_RUN = 256

# --options=NONE: ignore ctags.conf/ctags.d so every image build indexes the
# same way. --map-C=+.h: the C map only knows *.c, and curl's macros (and
# most libraries' declarations) live in headers. --links=no keeps a symlink
# in the untrusted worktree from pulling files in from outside /src.
_CTAGS_OPTIONS = (
    "--options=NONE",
    "--quiet",
    "--output-format=json",
    "--fields=+nK",
    "--languages=C",
    "--map-C=+.h",
    "--exclude=.git",
    "--links=no",
    "-R",
    "-o",
    "-",
)

# These startup diagnostics do not indicate a skipped C input. Anything
# else on stderr prevents the search from claiming complete coverage.
_CTAGS_STARTUP_MESSAGES = {
    "ctags: Notice: No options will be read from files or environment",
    "ctags: Warning: Enabling Cargo subparser may enable TOML parser.",
    "ctags: Warning: The current implementation of the TOML parser is broken.",
}

# ctags kinds that map onto core.models.SymbolKind. Everything else
# (member, enumerator, prototype, ...) is reported as "kind unknown" rather
# than forced into a bucket the enum does not have.
_KINDS = {
    "function": SymbolKind.FUNCTION,
    "macro": SymbolKind.MACRO,
    "struct": SymbolKind.STRUCT,
    "typedef": SymbolKind.TYPE,
    "variable": SymbolKind.VARIABLE,
}


class TagParseError(Exception):
    """ctags wrote something that is not the JSON we asked for."""


class TagIndexError(Exception):
    """The index could not be read completely, so absence is unknowable."""


class BadPathError(Exception):
    """A report-supplied path does not stay inside the checkout."""


@dataclass(frozen=True)
class Tag:
    """One ctags definition tag."""

    name: str
    path: str  # absolute, as ctags printed it: /src/lib/dynbuf.c
    line: int
    kind: str  # ctags kind name: function, macro, struct, ...

    @property
    def rel_path(self) -> str:
        """Path relative to the mounted worktree, for summaries and evidence."""

        prefix = SANDBOX_SRC + "/"
        return self.path[len(prefix):] if self.path.startswith(prefix) else self.path


@dataclass(frozen=True)
class TagSearch:
    """Outcome of an index search.

    ``complete`` is False whenever the tool failed, timed out, or its output
    was truncated. Only a complete search may support a "not found" claim.
    ``files_indexed`` is only meaningful when ``complete`` is True; an
    incomplete search reports how far it got in ``detail`` instead.
    """

    tags: tuple[Tag, ...]
    complete: bool
    detail: str
    files_indexed: int


class LineReadStatus(StrEnum):
    """How a bounded line read ended."""

    OK = "ok"
    ABSENT = "absent"  # the path is definitely not there
    UNREADABLE = "unreadable"  # the path exists but the tool could not read it
    FAILED = "failed"  # the read failed for another reason; nothing is known
    INCOMPLETE = "incomplete"  # the run timed out or its output was capped


@dataclass(frozen=True)
class LineRead:
    """One read of ``start``..``end`` from a file under ``/src``.

    ``status`` separates "the path is not there" -- the absence signal a later
    history check follows up -- from "the output was cut short" or a read that
    failed for some other reason, which prove nothing either way.
    """

    lines: tuple[str, ...]
    status: LineReadStatus
    detail: str


def ctags_argv(targets: list[str]) -> list[str]:
    """Build the ctags invocation for one bounded set of absolute paths.

    Targets are absolute paths under /src, so none of them can be read as an
    option even if the repository contains a file called "--foo.c".
    """

    return ["ctags", *_CTAGS_OPTIONS, *targets]


def symbol_kind(ctags_kind: str) -> SymbolKind | None:
    """Map a ctags kind onto the shared enum, or None when there is no match."""

    return _KINDS.get(ctags_kind)


def sandbox_target(hint: str) -> str:
    """Turn a report-supplied path into an absolute path inside ``/src``.

    Validation runs on the raw components *before* normalising: ``normpath``
    would silently rewrite ``lib/../dynbuf.c`` to ``dynbuf.c`` and hide the
    traversal attempt. Any ``..`` component is refused outright, as are
    absolute paths, ``~`` and backslashes (which are not separators in the
    sandbox and usually mean a Windows path leaked into the report).
    """

    if not hint or _has_control_char(hint) or "\ufffd" in hint:
        raise BadPathError("path is empty or contains unusable characters")
    if hint.startswith(("/", "~")) or "\\" in hint:
        raise BadPathError(f"path must be relative and POSIX: {hint!r}")
    if any(part == ".." for part in hint.split("/")):
        raise BadPathError(f"path must not contain '..': {hint!r}")
    if ".git" in hint.split("/"):
        raise BadPathError("git administrative files are not part of the checkout tree")

    relative = posixpath.normpath(hint)
    if relative in (".", "..") or relative.startswith("../"):
        raise BadPathError(f"path does not name a file: {hint!r}")

    target = posixpath.normpath(f"{SANDBOX_SRC}/{relative}")
    if not target.startswith(SANDBOX_SRC + "/"):
        raise BadPathError(f"path escapes {SANDBOX_SRC}: {hint!r}")
    return target


def resolve_target(session: SandboxSession, target: str) -> str:
    """Resolve worktree symlinks in the sandbox before reading a path.

    Lexical validation alone cannot stop a repository symlink from pointing
    outside the mounted checkout. ``-m`` also resolves a missing final path,
    so a missing file can still be classified without reading outside /src.
    """
    result = session.run(["realpath", "-m", "--", target], timeout_s=READ_TIMEOUT_S)
    if result.timed_out or result.truncated or result.exit_code != 0:
        raise BadPathError("could not resolve the path inside the sandbox")
    resolved = result.stdout.removesuffix("\n")
    if (
        not _inside_worktree(resolved)
        or _has_control_char(resolved)
        or "\ufffd" in resolved
        or ".git" in resolved.split("/")
    ):
        raise BadPathError("resolved path is outside the checkout or is not a tree file")
    return resolved


def parse_tags(stdout: str) -> list[Tag]:
    """Parse ctags JSON output into tags we can trust.

    A line that is not a complete tag makes the whole run untrustworthy: the
    caller must treat it as an incomplete index, not as "no more tags". The
    fields are checked before use because ``path`` can become a tree-sitter
    argument, and the indexed repository is untrusted input.
    """

    tags: list[Tag] = []
    for line in stdout.splitlines():
        if not line.strip():
            continue
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as exc:
            raise TagParseError(f"line is not JSON: {line[:80]!r}") from exc
        if not isinstance(raw, dict):
            raise TagParseError(f"tag is not an object: {line[:80]!r}")
        name = raw.get("name")
        path = raw.get("path")
        kind = raw.get("kind")
        line_no = raw.get("line")
        if (
            not isinstance(name, str)
            or not name
            or not isinstance(kind, str)
            or not kind
            # `type(...) is int` on purpose: bool is a subclass of int, so
            # isinstance would accept "line": true.
            or type(line_no) is not int
            or line_no < 1
            or not isinstance(path, str)
            or not _inside_worktree(path)
            or _has_control_char(path)
        ):
            raise TagParseError(f"tag has an unusable name/path/kind/line: {line[:80]!r}")
        tags.append(Tag(name=name, path=posixpath.normpath(path), line=line_no, kind=kind))
    return tags


def _inside_worktree(path: str) -> bool:
    """Whether a path resolves to a file inside ``/src`` (``..`` cannot escape)."""

    return posixpath.normpath(path).startswith(SANDBOX_SRC + "/")


def _has_control_char(text: str) -> bool:
    """Whether text carries C0/DEL characters a filename should never expose."""

    return any(ord(ch) < 0x20 or ord(ch) == 0x7F for ch in text)


def find_tags(session: SandboxSession, symbol: str) -> TagSearch:
    """Find definition tags named exactly ``symbol`` anywhere under ``/src``.

    The whole checkout is always searched. Scoping the index to a hinted file
    would let "not in that file" masquerade as "not in the repository", which
    is a false absence signal for whoever follows up on it.
    """

    try:
        files = _c_source_files(session)
    except TagIndexError as exc:
        return TagSearch((), False, str(exc), 0)

    if not files:
        # Saying "absent" here would be wrong for a C++ or Python repo: this
        # checker only speaks C, so it has nothing to say at all.
        return TagSearch((), False, f"no C source files under {SANDBOX_SRC}", 0)

    try:
        matches, indexed = _scan(session, files, symbol)
    except TagIndexError as exc:
        return TagSearch((), False, str(exc), 0)

    matches.sort(key=lambda tag: (tag.path, tag.line))
    return TagSearch(
        tuple(matches), True, f"indexed {indexed} C files under {SANDBOX_SRC}", indexed
    )


def line_argv(path: str, start: int, end: int) -> list[str]:
    """Build one bounded read of absolute ``path``.

    The line numbers are ints and the path is absolute behind ``--``, so a
    report-supplied path can never be read as a sed option.
    """

    return ["sed", "-n", f"{start},{end}p", "--", path]


def output_lines(stdout: str) -> tuple[str, ...]:
    """Count the LF-delimited lines that sed reads, not Python's broader splitlines."""
    if not stdout:
        return ()
    lines = stdout.split("\n")
    if lines[-1] == "":
        lines.pop()
    return tuple(lines)


def read_lines(session: SandboxSession, path: str, start: int, end: int) -> LineRead:
    """Read lines ``start``..``end`` (1-based, inclusive) from ``path``.

    ``path`` must be the output of :func:`sandbox_target`: this wrapper does
    not re-validate it, because validation belongs once, at the checker
    boundary.
    """

    result = session.run(line_argv(path, start, end), timeout_s=READ_TIMEOUT_S)
    lines = output_lines(result.stdout)
    if result.timed_out or result.truncated:
        return LineRead(lines, LineReadStatus.INCOMPLETE, "the read was cut short")
    # The sandbox decodes bytes with replacement. U+FFFD could therefore be a
    # decoding error rather than source text, so it cannot support evidence.
    if "\ufffd" in result.stdout:
        return LineRead((), LineReadStatus.FAILED, "the read may contain invalid UTF-8")
    if result.exit_code == 0:
        return LineRead(lines, LineReadStatus.OK, "the read completed")
    return _classify_failed_read(session, path, result)


def seal_lines(
    session: SandboxSession, path: str, start: int, end: int
) -> DeterministicProof:
    """Re-run the decisive line read and seal it."""

    return session.seal(line_argv(path, start, end), timeout_s=READ_TIMEOUT_S)


def _classify_failed_read(
    session: SandboxSession, path: str, result: CommandResult
) -> LineRead:
    """Tell a missing path from a present one instead of guessing from ``sed``.

    ``sed`` exits with the same status for every failure, so its exit code
    cannot say whether the file is gone, unreadable, or something else went
    wrong. Only a definitive "not there" is reported as absent: that is the
    one classification a history check acts on.
    """

    exists = _probe(session, "-e", path)
    if exists is False and _confirm_absent(session, path):
        return LineRead((), LineReadStatus.ABSENT, "the path does not exist")
    if exists is True and _probe(session, "-r", path) is False:
        return LineRead(
            (), LineReadStatus.UNREADABLE, "the path exists but cannot be read"
        )
    return LineRead((), LineReadStatus.FAILED, f"sed exited {result.exit_code}")


def _confirm_absent(session: SandboxSession, path: str) -> bool:
    # test -e also returns false when an ancestor is inaccessible or a symlink
    # dangles. Only mark absence when the nearest existing parent is searchable.
    current = path
    # Bound Docker round trips for a maliciously deep report-supplied path.
    for _ in range(32):
        if current == SANDBOX_SRC:
            return False
        if _probe(session, "-L", current) is not False:
            return False
        parent = posixpath.dirname(current)
        exists = _probe(session, "-e", parent)
        if exists is None:
            return False
        if exists:
            return _probe(session, "-d", parent) is True and _probe(session, "-x", parent) is True
        current = parent
    return False


def _probe(session: SandboxSession, flag: str, path: str) -> bool | None:
    """Ask ``test`` about ``path``; None when the probe itself cannot answer.

    A clean false status is not by itself proof of absence: inaccessible
    ancestors and dangling symlinks also make ``test -e`` return false.
    """

    result = session.run(["test", flag, path], timeout_s=READ_TIMEOUT_S)
    if result.timed_out or result.truncated or result.exit_code not in (0, 1):
        return None
    return result.exit_code == 0


def _c_source_files(session: SandboxSession) -> list[str]:
    """List the files ctags will treat as C (``*.c`` and ``*.h``), sorted.

    The extensions mirror ``--languages=C --map-C=+.h`` so the enumeration
    and the parser always agree on what counts as a source file.
    """

    argv: Argv = [
        "find",
        SANDBOX_SRC,
        "-type",
        "f",
        "(",
        "-name",
        "*.c",
        "-o",
        "-name",
        "*.h",
        ")",
        "-print0",
    ]
    result = session.run(argv, timeout_s=CTAGS_TIMEOUT_S)
    if result.exit_code != 0 or result.truncated or result.timed_out:
        raise TagIndexError(f"could not list the C sources under {SANDBOX_SRC}")
    return sorted(name for name in result.stdout.split("\x00") if name)


def _scan(session: SandboxSession, files: list[str], symbol: str) -> tuple[list[Tag], int]:
    """Index ``files`` in bounded chunks, halving any chunk that truncates."""

    matches: list[Tag] = []
    indexed = 0

    def run(chunk: list[str]) -> None:
        nonlocal indexed
        result = session.run(ctags_argv(chunk), timeout_s=CTAGS_TIMEOUT_S)
        # The count is what makes a failure diagnosable: it says how much of
        # the tree was actually read before the search gave up.
        progress = f"; {indexed} C files indexed so far"
        if result.exit_code != 0 or result.timed_out:
            raise TagIndexError(f"ctags failed on {chunk[0]} (exit {result.exit_code}){progress}")
        # ctags can exit zero after failing to open an input file. A warning
        # means the search may not cover the whole checkout.
        if set(result.stderr.splitlines()) - _CTAGS_STARTUP_MESSAGES:
            raise TagIndexError(f"ctags reported a warning on {chunk[0]}{progress}")
        if result.truncated:
            if len(chunk) == 1:
                # One file whose tags exceed the output cap: we cannot see
                # all of it, so the search is incomplete rather than empty.
                raise TagIndexError(f"ctags output for {chunk[0]} was truncated{progress}")
            middle = len(chunk) // 2
            run(chunk[:middle])
            run(chunk[middle:])
            return
        try:
            tags = parse_tags(result.stdout)
        except TagParseError as exc:
            raise TagIndexError(f"{exc}{progress}") from exc
        indexed += len(chunk)
        matches.extend(tag for tag in tags if tag.name == symbol)

    for start in range(0, len(files), _FILES_PER_RUN):
        run(files[start : start + _FILES_PER_RUN])
    return matches, indexed
