"""Check that a file at the resolved tag has the claimed lines.

Policy, in one sentence: the checker reports what the claimed lines contain at
the resolved tag; a miss is a signal for a history check, never a
contradiction.

* The claimed range matches the snippet -> ``CONFIRMED``, with a proof sealed
  from the exact read.
* The path is missing, the range is past the end of the file, or the snippet
  is not in the range -> ``CANT_CHECK`` plus a signal for the caller that can
  follow up. This checker never contradicts: code moves and reports cite stale
  lines, and only history checking may turn "not here" into a contradiction.
* A read that timed out or was capped is ``CANT_CHECK`` with no signal: it
  proves nothing either way. The same goes for a path that is present but
  cannot be read, or a read that failed for another reason: history cannot
  explain either, so reporting the file as missing would mislead it.

Comparison is mechanical: whitespace runs inside a line are collapsed and
line ends are stripped (``DomainModel`` trims the snippet, so an indented
first line loses its indentation), and line breaks are preserved. There is no
fuzzy matching, no search outside the claimed range, and no judgement about
whether two different snippets "mean" the same thing.
"""

from __future__ import annotations

from collections.abc import Sequence

from triage0.checkers.tools.ast_tools import (
    BadPathError,
    LineRead,
    LineReadStatus,
    output_lines,
    read_lines,
    resolve_target,
    sandbox_target,
    seal_lines,
)
from triage0.checkers.tools.report_text import quote
from triage0.core.evidence import EvidenceResult, EvidenceStatus, MissingEvidence
from triage0.core.interfaces import SandboxSession
from triage0.core.models import AnyClaim, ClaimKind, FileLineClaim

__all__ = ["FileLineChecker"]


class FileLineChecker:
    """``BaseClaimChecker`` for ``ClaimKind.FILE_LINE``."""

    claim_kind = ClaimKind.FILE_LINE

    def supports(self, claim: AnyClaim) -> bool:
        return isinstance(claim, FileLineClaim)

    def check(self, claim: AnyClaim, sandbox: SandboxSession) -> EvidenceResult:
        if not isinstance(claim, FileLineClaim):
            raise TypeError(f"FileLineChecker cannot check a {claim.kind} claim")

        # The path is untrusted and, unlike a symbol name, it has to reach the
        # command line. Validate it once here; the tool wrapper trusts this.
        try:
            target = sandbox_target(claim.path)
        except BadPathError as exc:
            return EvidenceResult(
                status=EvidenceStatus.CANT_CHECK,
                summary=f"unusable path: {exc}",
            )

        snippet = claim.expected_snippet
        if snippet is not None and not snippet.strip():
            # `"" in anything` is True: an all-whitespace snippet would confirm
            # every file, so refuse to compare instead.
            return EvidenceResult(
                status=EvidenceStatus.CANT_CHECK,
                summary="the expected snippet is empty after trimming",
            )

        try:
            target = resolve_target(sandbox, target)
        except BadPathError as exc:
            return EvidenceResult(
                status=EvidenceStatus.CANT_CHECK,
                summary=f"unusable path: {exc}",
            )

        start = claim.line_start
        end = claim.effective_line_end
        read = read_lines(sandbox, target, start, end)

        if read.status is LineReadStatus.ABSENT:
            return EvidenceResult(
                status=EvidenceStatus.CANT_CHECK,
                summary=f"could not read {claim.path}: {read.detail}",
                missing=MissingEvidence.FILE_ABSENT,
            )
        if read.status is LineReadStatus.UNREADABLE or read.status is LineReadStatus.FAILED:
            # A present file the sandbox could not read, or a read that failed
            # for another reason, is an analysis limit: git history cannot
            # explain it, so there is no missing signal to hand over.
            return EvidenceResult(
                status=EvidenceStatus.CANT_CHECK,
                summary=f"could not read {claim.path}: {read.detail}",
            )
        if read.status is LineReadStatus.INCOMPLETE:
            return EvidenceResult(
                status=EvidenceStatus.CANT_CHECK,
                summary=f"the read of {claim.path} did not complete: {read.detail}",
            )

        if len(read.lines) < end - start + 1:
            return EvidenceResult(
                status=EvidenceStatus.CANT_CHECK,
                summary=_range_summary(claim, read),
                missing=MissingEvidence.LINE_OUT_OF_RANGE,
            )

        if snippet is not None and not _contains(read.lines, snippet):
            return EvidenceResult(
                status=EvidenceStatus.CANT_CHECK,
                summary=(
                    f"{claim.path}:{start}-{end} does not contain {quote(snippet)}; "
                    f"it reads {quote(_join(read.lines))}"
                ),
                missing=MissingEvidence.CONTENT_DIFFERS,
            )

        summary = (
            f"{claim.path}:{start}-{end} contains {quote(snippet)}"
            if snippet is not None
            else f"{claim.path}:{start}-{end} exists"
        )
        return self._seal(claim, sandbox, target, summary=summary)

    def _seal(
        self,
        claim: FileLineClaim,
        sandbox: SandboxSession,
        target: str,
        *,
        summary: str,
    ) -> EvidenceResult:
        """Seal the decisive read, and only then return ``CONFIRMED``.

        Same three gates as the symbol checker: ``downgrade`` turns a
        timed-out or truncated proof into ``CANT_CHECK``; a nonzero exit code
        means the tool did not succeed; and the re-run must still show the
        claimed lines.
        """

        proof = seal_lines(sandbox, target, claim.line_start, claim.effective_line_end)
        result = EvidenceResult.downgrade(
            status=EvidenceStatus.CONFIRMED, proof=proof, summary=summary
        )
        if not result.is_terminal:
            return result
        if proof.exit_code != 0:
            return EvidenceResult(
                status=EvidenceStatus.CANT_CHECK,
                proof=proof,
                summary=f"the sealing read exited {proof.exit_code}",
            )
        if "\ufffd" in proof.stdout or not _matches(claim, output_lines(proof.stdout)):
            return EvidenceResult(
                status=EvidenceStatus.CANT_CHECK,
                proof=proof,
                summary="the sealing read no longer shows the claimed lines",
            )
        return result


def _range_summary(claim: FileLineClaim, read: LineRead) -> str:
    """Say exactly how far the file got, without guessing where the code went."""

    if not read.lines:
        return f"{claim.path} has no line {claim.line_start}"
    last = claim.line_start + len(read.lines) - 1
    return (
        f"{claim.path} ends at line {last}, before the claimed line "
        f"{claim.effective_line_end}"
    )


def _matches(claim: FileLineClaim, lines: Sequence[str]) -> bool:
    """Whether ``lines`` still support the confirmed claim."""

    if len(lines) < claim.effective_line_end - claim.line_start + 1:
        return False
    snippet = claim.expected_snippet
    return snippet is None or _contains(lines, snippet)


def _contains(lines: Sequence[str], snippet: str) -> bool:
    """Whether the block contains the snippet under the fixed whitespace rule."""

    return _normalized(snippet) in _normalized(_join(lines))


def _join(lines: Sequence[str]) -> str:
    return "\n".join(lines)


def _normalized(text: str) -> str:
    """Collapse whitespace runs per line while preserving the line breaks.

    ``DomainModel`` trims the whole snippet, so an indented first line loses
    its indentation but interior lines keep theirs. ``str.split`` is
    Python-defined, so this is a fixed mechanical rule, not a similarity
    judgement.
    """

    return "\n".join(" ".join(line.split()) for line in text.split("\n")).strip()
