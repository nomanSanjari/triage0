"""Load the shared golden fixtures in ``tests/fixtures/golden/``.

Each fixture is a small curl report plus the verdict a correct pipeline should
reach for each claim in it, checked by hand against real curl at a stated tag.
The labels live in this wrapper, not in ``claim.result``: core's
``EvidenceResult`` only accepts CONFIRMED or CONTRADICTED with a real
``DeterministicProof``, and a hand-checked label is not one.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from triage0.core.evidence import EvidenceStatus
from triage0.core.models import AnyClaim, NormalizedReport

FIXTURES_DIR = Path(__file__).parent / "fixtures" / "golden"


class _FixtureModel(BaseModel):
    # extra="forbid" turns a mistyped field name into an error instead of a
    # silently ignored key. No whitespace stripping: evidence text must match
    # the source byte for byte.
    model_config = ConfigDict(frozen=True, extra="forbid")


class LineEvidence(_FixtureModel):
    """One source line, exactly as it reads at the case's commit.

    ``commit`` points at a different commit instead, for evidence about the
    past, such as where a since-renamed symbol used to be defined.
    """

    kind: Literal["line"]
    path: str = Field(min_length=1)
    line: int = Field(ge=1)
    text: str
    commit: str | None = Field(default=None, pattern=r"^[0-9a-f]{40}$")
    note: str | None = None


class AbsentEvidence(_FixtureModel):
    """Proof by absence: ``pattern`` appears nowhere in ``scope``.

    ``tree`` is every file at the case's commit; ``history`` is every commit
    up to it. A renamed symbol is absent from the tree but not the history,
    which is how the two cases are told apart.
    """

    kind: Literal["absent"]
    pattern: str = Field(min_length=1)
    scope: Literal["tree", "history"]
    note: str | None = None


Evidence = Annotated[LineEvidence | AbsentEvidence, Field(discriminator="kind")]


class ExpectedClaim(_FixtureModel):
    claim: AnyClaim
    expected_status: EvidenceStatus
    evidence: tuple[Evidence, ...] = ()

    @model_validator(mode="after")
    def _terminal_labels_need_evidence(self) -> Self:
        # Same idea as rule 1: a CONFIRMED/CONTRADICTED label must point at
        # source someone can re-check, never rest on memory.
        if self.expected_status is not EvidenceStatus.CANT_CHECK and not self.evidence:
            raise ValueError(
                f"{self.claim.claim_id}: {self.expected_status.value} needs evidence"
            )
        return self


class GoldenCase(_FixtureModel):
    case_id: str = Field(min_length=1)
    description: str = Field(min_length=1)
    repo_url: str = Field(min_length=1)
    # Null only for a report with nothing to check, such as one that names no
    # version; see _claims_need_a_version.
    tag: str | None = Field(min_length=1)
    commit: str | None = Field(pattern=r"^[0-9a-f]{40}$")
    report: NormalizedReport
    requested_version: str | None
    expected_claims: tuple[ExpectedClaim, ...]

    @model_validator(mode="after")
    def _claims_need_a_version(self) -> Self:
        # A label is only true at one version, so it is meaningless without
        # the commit it was checked at.
        if self.expected_claims and (self.tag is None or self.commit is None):
            raise ValueError("cases with claims need a tag and commit")
        return self

    @model_validator(mode="after")
    def _claims_come_from_the_report(self) -> Self:
        # #8 drops claims whose excerpt isn't in the report text, so a fixture
        # that broke this would expect a claim the decomposer must not return.
        seen: set[str] = set()
        for expected in self.expected_claims:
            claim = expected.claim
            if claim.claim_id in seen:
                raise ValueError(f"duplicate claim_id: {claim.claim_id}")
            seen.add(claim.claim_id)
            if not claim.source_excerpt or claim.source_excerpt not in self.report.body:
                raise ValueError(
                    f"{claim.claim_id}: source_excerpt not found in report body"
                )
        return self


def load_all() -> tuple[GoldenCase, ...]:
    """Every fixture, sorted by file name.

    ``case_id`` must equal the file name so a failing test names a file you
    can open.
    """

    cases = []
    for path in sorted(FIXTURES_DIR.glob("*.json")):
        case = GoldenCase.model_validate_json(path.read_text(encoding="utf-8"))
        if case.case_id != path.stem:
            raise ValueError(f"{path.name}: case_id is {case.case_id!r}")
        cases.append(case)
    return tuple(cases)
