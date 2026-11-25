"""Check that a named C symbol exists at the resolved tag.

Policy: ctags finds candidates, but only tree-sitter syntax confirms a
definition. Finding nothing is never proof that something is absent.

* Found with the claimed kind -> ``CONFIRMED``, with a proof sealed from a
  tree-sitter query over the one file that defines it.
* The claim says the symbol can't exist, but the same syntax confirms it
  -> ``CONTRADICTED``. This is source existence, not build activity.
* Not found anywhere in the checkout -> ``CANT_CHECK`` plus the
  ``SYMBOL_ABSENT`` signal, which a later history check can follow up as a
  rename. This checker never contradicts on absence.
* Found, but no tag has the claimed kind -> ``CANT_CHECK``. Kind labels differ
  between parsers (prototypes, macro-generated code, duplicate tags), so a
  mismatch is a reason to abstain, not to contradict. This holds for both
  directions: a macro that shares the name of a claimed-missing function is
  not that function, so it cannot contradict the "cannot exist" claim.

The search always covers the whole checkout. ``file_hint`` only picks which
definition to cite when a name has several, and a citation outside the hinted
file is reported in the summary: scoping the search to the hint would turn
"not in that file" into a false "not in the repository".
"""

from __future__ import annotations

from collections.abc import Sequence

from triage0.checkers.tools.ast_tools import (
    BadPathError,
    Tag,
    find_tags,
    sandbox_target,
    symbol_kind,
)
from triage0.checkers.tools.report_text import quote
from triage0.checkers.tools.syntax_tools import confirms_definition, seal_definition
from triage0.core.evidence import EvidenceResult, EvidenceStatus, MissingEvidence
from triage0.core.interfaces import SandboxSession
from triage0.core.models import AnyClaim, ClaimKind, SymbolClaim, SymbolKind

__all__ = ["SymbolChecker"]


class SymbolChecker:
    """``BaseClaimChecker`` for ``ClaimKind.SYMBOL`` (C sources only)."""

    claim_kind = ClaimKind.SYMBOL

    def supports(self, claim: AnyClaim) -> bool:
        return isinstance(claim, SymbolClaim)

    def check(self, claim: AnyClaim, sandbox: SandboxSession) -> EvidenceResult:
        if not isinstance(claim, SymbolClaim):
            raise TypeError(f"SymbolChecker cannot check a {claim.kind} claim")

        # The hint is untrusted and only chooses which definition to cite, but
        # a malformed one makes the claim itself unusable, so refuse it before
        # running anything.
        try:
            cited = sandbox_target(claim.file_hint) if claim.file_hint else None
        except BadPathError as exc:
            return EvidenceResult(
                status=EvidenceStatus.CANT_CHECK,
                summary=f"unusable file hint: {exc}",
            )

        search = find_tags(sandbox, claim.symbol)
        if not search.complete:
            return EvidenceResult(
                status=EvidenceStatus.CANT_CHECK,
                summary=f"could not index the code: {search.detail}",
            )

        if not search.tags:
            return EvidenceResult(
                status=EvidenceStatus.CANT_CHECK,
                summary=f"no ctags tag named {quote(claim.symbol)}; {search.detail}",
                missing=MissingEvidence.SYMBOL_ABSENT,
            )

        # Kind scoping applies in both directions: a macro that happens to
        # carry the name of a claimed-missing function is a different symbol,
        # so it cannot contradict the "cannot exist" claim either.
        matching = [tag for tag in search.tags if _kind_matches(claim.symbol_kind, tag.kind)]
        if not matching:
            found = ", ".join(sorted({tag.kind for tag in search.tags}))
            return EvidenceResult(
                status=EvidenceStatus.CANT_CHECK,
                summary=f"{quote(claim.symbol)} is indexed as {found}, not as "
                f"{claim.symbol_kind.value}; a kind mismatch is not a contradiction",
            )

        tag = _cite(matching, cited)
        if matching.count(tag) != 1:
            return EvidenceResult(
                status=EvidenceStatus.CANT_CHECK,
                summary=f"ctags did not uniquely identify {quote(tag.name)} at "
                f"{tag.rel_path}:{tag.line}",
            )
        summary = _citation(tag, claim, cited)
        status = EvidenceStatus.CONFIRMED
        if not claim.expected_present:
            status = EvidenceStatus.CONTRADICTED
            summary += "; the claim says it cannot exist"

        try:
            proof = seal_definition(sandbox, tag)
        except BadPathError as exc:
            return EvidenceResult(status=EvidenceStatus.CANT_CHECK, summary=str(exc))
        if proof.timed_out or proof.truncated:
            return EvidenceResult.downgrade(
                status=status, proof=proof, summary="the tree-sitter query did not complete"
            )
        if proof.exit_code != 0:
            return EvidenceResult(
                status=EvidenceStatus.CANT_CHECK,
                proof=proof,
                summary=f"the sealing tree-sitter query exited {proof.exit_code}",
            )
        if not confirms_definition(proof, tag):
            return EvidenceResult(
                status=EvidenceStatus.CANT_CHECK,
                proof=proof,
                summary=f"tree-sitter did not uniquely confirm the indexed {tag.kind} "
                f"{quote(tag.name)} at {tag.rel_path}:{tag.line}",
            )
        return EvidenceResult(status=status, proof=proof, summary=summary)


def _kind_matches(wanted: SymbolKind, ctags_kind: str) -> bool:
    """Whether a tag's kind satisfies the claim's kind.

    ``UNKNOWN`` means the report did not say what the symbol is, so any kind
    is acceptable; otherwise the ctags kind must map onto the claimed one.
    """

    if wanted is SymbolKind.UNKNOWN:
        return True
    return symbol_kind(ctags_kind) is wanted


def _cite(tags: Sequence[Tag], wanted: str | None) -> Tag:
    """Pick the definition to cite: the hinted file if present, else the first.

    ``tags`` is already sorted by (path, line), so the choice is deterministic
    even when a name is defined in several files (ordinary for ``static``
    helpers in C).
    """

    if wanted is not None:
        for tag in tags:
            if tag.path == wanted:
                return tag
    return tags[0]


def _citation(tag: Tag, claim: SymbolClaim, cited: str | None) -> str:
    """Describe where the symbol was found, noting a hint that points elsewhere."""

    text = (
        f"{quote(claim.symbol)} is defined as a {tag.kind} in source at {tag.rel_path}:{tag.line}"
    )
    hint = claim.file_hint
    if cited is not None and hint is not None and tag.path != cited:
        text += f"; the claim points at {quote(hint)}"
    return text
