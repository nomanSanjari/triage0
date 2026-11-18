"""Search a repository's git history (roadmap task 2.4b).

These run on the host against the cached bare repo, through the hardened
``run_git``: git can't run inside the sandbox (the worktree's ``.git`` points
at a host path), and reading history executes no repository code.

Symbol names and paths come from bug reports, so they are attacker-controlled.
Each is checked here, once, before it can reach git, and then travels as a
single argument that can't be read as an option (rule 2). A value that fails
the check raises ``ValueError`` and git is never called.

Every query asks git for almost nothing back (one commit id, a yes/no, or a
range of at most ``MAX_RANGE_LINES`` lines), so a huge history can't flood the
caller; the timeout stops one that is merely slow. A git failure or timeout
propagates (``CalledProcessError`` / ``TimeoutExpired``): callers turn it
into ``CANT_CHECK``, never into a verdict.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path, PurePosixPath

from triage0.checkers.tools.git_tools import run_git

__all__ = [
    "HISTORY_TIMEOUT_SECONDS",
    "MAX_RANGE_LINES",
    "blame_lines",
    "last_change",
    "line_history",
    "parents",
    "path_in_tree",
    "symbol_in_tree",
    "tree_hash",
]

# Walking all of curl's history takes seconds on a laptop; this leaves room
# for a slower machine without letting a run hang.
HISTORY_TIMEOUT_SECONDS = 120.0
MAX_RANGE_LINES = 200
MAX_COMMITS = 50

# A C identifier: no spaces, quotes, shell characters or leading "-".
_SYMBOL_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
# Full object ids only (what git itself prints), never a branch name or an
# abbreviation that could be read as something else.
_COMMIT_RE = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})")
_BLAME_HUNK_RE = re.compile(r"([0-9a-f]{40}|[0-9a-f]{64}) \d+ (\d+) (\d+)")


def _check_symbol(symbol: str) -> str:
    if not _SYMBOL_RE.fullmatch(symbol):
        raise ValueError(f"not a C identifier: {symbol!r}")
    return symbol


def _check_commit(commit: str) -> str:
    if not _COMMIT_RE.fullmatch(commit):
        raise ValueError(f"not a full commit id: {commit!r}")
    return commit


def _check_path(path: str) -> str:
    """A path inside the repository: relative, no ``..``, no leading ``-``."""

    parts = PurePosixPath(path).parts
    if (
        not path
        or path.startswith(("/", "-"))
        or ".." in parts
        or any(ch in path for ch in "\0\n")
    ):
        raise ValueError(f"not a safe repository path: {path!r}")
    return path


def _check_range(start: int, end: int) -> None:
    if start < 1 or end < start or end - start + 1 > MAX_RANGE_LINES:
        raise ValueError(
            f"line range must be 1-based, ordered, and at most "
            f"{MAX_RANGE_LINES} lines: {start}-{end}"
        )


def symbol_in_tree(repo: Path, commit: str, symbol: str) -> bool:
    """Whether ``symbol`` appears, as a whole word, in any file at ``commit``."""

    try:
        run_git(
            ["grep", "-q", "-w", "-F", "-e", _check_symbol(symbol), _check_commit(commit)],
            repo=repo,
        )
    except subprocess.CalledProcessError as error:
        if error.returncode != 1:  # 1 means "no match"; anything else is a real error
            raise
        return False
    return True


def last_change(repo: Path, symbol: str, commit: str | None = None) -> str | None:
    """The newest commit that added or removed ``symbol``, or None if none did.

    ``commit`` limits the search to history up to it; None searches every
    ref. Searching more can only find more, so None can only make a "never
    existed" conclusion rarer, never wrong.

    This is ``git log -S``, a plain *text* match, because the whole-word
    (regex) version is far too slow: over 120 s on curl against about 8 s. So
    ``curlx_dyn_ad`` matches ``curlx_dyn_add`` and a hit doesn't prove the
    name was ever a whole word. None does prove it never appeared at all; a
    caller that needs the commit where a word disappeared checks the hit with
    ``symbol_in_tree`` on the commit and its ``parents``.
    """

    rev = "--all" if commit is None else _check_commit(commit)
    output = run_git(
        [
            "log",
            f"-S{_check_symbol(symbol)}",
            "--format=%H",
            "--max-count=1",
            rev,
        ],
        repo=repo,
        timeout=HISTORY_TIMEOUT_SECONDS,
    ).stdout.strip()
    return output or None


def path_in_tree(repo: Path, commit: str, path: str) -> bool:
    """Whether ``path`` is a file at ``commit`` (a directory doesn't count)."""

    # One tree entry at most, so the output is a single line.
    entry = run_git(
        ["ls-tree", _check_commit(commit), "--", _check_path(path)], repo=repo
    ).stdout.split()
    return len(entry) >= 2 and entry[1] == "blob"


def tree_hash(repo: Path, commit: str) -> str:
    """The git tree id of ``commit``, which a proof records."""

    return run_git(
        ["rev-parse", "--verify", f"{_check_commit(commit)}^{{tree}}"], repo=repo
    ).stdout.strip()


def parents(repo: Path, commit: str) -> tuple[str, ...]:
    """The parent commits of ``commit`` (none for the first commit)."""

    # One line: the commit, then its parents.
    line = run_git(
        ["rev-list", "--parents", "--max-count=1", _check_commit(commit)], repo=repo
    ).stdout.split()
    return tuple(line[1:])


def line_history(repo: Path, commit: str, path: str, start: int, end: int) -> list[str]:
    """Commits up to ``commit`` that changed lines ``start``-``end``, newest first.

    This is ``git log -L``, which follows the lines through edits; at most
    ``MAX_COMMITS`` come back.
    """

    _check_range(start, end)
    output = run_git(
        [
            "log",
            f"-L{start},{end}:{_check_path(path)}",
            "--format=%H",
            "--no-patch",
            f"--max-count={MAX_COMMITS}",
            _check_commit(commit),
        ],
        repo=repo,
        timeout=HISTORY_TIMEOUT_SECONDS,
    ).stdout
    return output.split()


def blame_lines(repo: Path, commit: str, path: str, start: int, end: int) -> dict[int, str]:
    """For each line ``start``-``end`` of ``path`` at ``commit``, the commit that last wrote it.

    ``--incremental`` reports who wrote each block of lines without repeating
    the lines' text, so a minified file with huge lines can't flood the output.
    """

    _check_range(start, end)
    output = run_git(
        [
            "blame",
            "--incremental",
            f"-L{start},{end}",
            _check_commit(commit),
            "--",
            _check_path(path),
        ],
        repo=repo,
        timeout=HISTORY_TIMEOUT_SECONDS,
    ).stdout
    blamed: dict[int, str] = {}
    for line in output.splitlines():
        hunk = _BLAME_HUNK_RE.fullmatch(line)
        if hunk:
            first, count = int(hunk.group(2)), int(hunk.group(3))
            blamed.update({first + offset: hunk.group(1) for offset in range(count)})
    return blamed
