"""Confirm one ctags candidate against an error-free tree-sitter definition."""

from __future__ import annotations

import re
from dataclasses import dataclass

from triage0.checkers.tools.ast_tools import BadPathError, Tag, resolve_target, sandbox_target
from triage0.core.evidence import DeterministicProof
from triage0.core.interfaces import SandboxSession

_QUERY = "/opt/triage0/symbols.scm"
_KINDS = frozenset({"function", "macro", "struct", "typedef", "variable"})
_PATTERN = re.compile(r"^  pattern: [0-9]+\n", re.MULTILINE)
_CAPTURE = re.compile(
    r"    capture: (?:[0-9]+ - )?([a-z_]+), "
    r"start: \(([0-9]{1,10}), ([0-9]{1,10})\), end: \(([0-9]{1,10}), ([0-9]{1,10})\)"
    r"(?:, text: `(.*)`)?"
)

_Point = tuple[int, int]


@dataclass(frozen=True)
class _Capture:
    label: str
    start: _Point
    end: _Point
    text: str | None


def _query_argv(path: str) -> tuple[str, ...]:
    return (
        "tree-sitter", "query", "--config-path", "/opt/triage0/config.json",
        "--scope", "source.c", _QUERY, path,
    )


def seal_definition(session: SandboxSession, tag: Tag) -> DeterministicProof:
    """Run only the image's fixed query on the exact, non-symlink C path."""

    # A tree-sitter pathname can instead be a glob; never let it expand.
    if (
        any(char in tag.path for char in "*?[\ufffd")
        or sandbox_target(tag.rel_path) != tag.path
        or not tag.path.endswith((".c", ".h"))
        or resolve_target(session, tag.path) != tag.path
    ):
        raise BadPathError("the indexed path is not an exact C file in the checkout")
    return session.seal(_query_argv(tag.path), timeout_s=30)


def confirms_definition(proof: DeterministicProof, tag: Tag) -> bool:
    """Match the name in its own definition and reject local parser recovery.

    Query output is grouped by match; do not pair captures across groups.
    Tree-sitter prints capture text only for single-line nodes. Split on LF
    only, so source text cannot insert new records via other line separators.
    """

    if (
        proof.tool != "tree-sitter"
        or proof.argv != _query_argv(tag.path)
        or proof.exit_code != 0
        or proof.timed_out
        or proof.truncated
        or proof.stderr.strip()
        or "\ufffd" in proof.stdout
    ):
        return False
    source, *groups = _PATTERN.split(proof.stdout)
    if source != tag.path + "\n":
        return False

    definitions: list[_Capture] = []
    errors: list[_Capture] = []
    for group in groups:
        captures: dict[str, _Capture] = {}
        for line in group.removesuffix("\n").split("\n"):
            match = _CAPTURE.fullmatch(line)
            if match is None:
                return False
            label, sr, sc, er, ec, text = match.groups()
            start = (int(sr), int(sc))
            end = (int(er), int(ec))
            if label in captures or end < start:
                return False
            captures[label] = _Capture(label, start, end, text)
        if len(captures) == 1 and (captures.keys() & {"error", "missing"}):
            errors.extend(captures.values())
            continue
        name = captures.pop("name", None)
        if name is None or len(captures) != 1:
            return False
        definition = next(iter(captures.values()))
        if definition.label not in _KINDS:
            return False
        if (
            definition.label == tag.kind
            and name.start[0] == tag.line - 1 == name.end[0]
            and name.text == tag.name
            and definition.start <= name.start < name.end <= definition.end
        ):
            definitions.append(definition)

    if len(definitions) != 1:
        return False
    definition = definitions[0]
    return not any(_overlaps(error, definition) for error in errors)


def _overlaps(error: _Capture, definition: _Capture) -> bool:
    if error.start == error.end:
        # A missing final token (e.g. a brace) can sit at the node's end.
        return definition.start <= error.start <= definition.end
    return error.start < definition.end and definition.start < error.end
