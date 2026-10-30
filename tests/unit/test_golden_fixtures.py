"""Unit tests for the golden fixture loader and the fixtures it loads."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from tests import golden_fixtures
from tests.golden_fixtures import (
    AbsentEvidence,
    ExpectedClaim,
    GoldenCase,
    LineEvidence,
    load_all,
)
from triage0.core.evidence import EvidenceStatus
from triage0.core.models import (
    BuildFlagClaim,
    CallPathClaim,
    ClaimBase,
    ClaimKind,
    NormalizedReport,
    SymbolClaim,
)


def make_case(**overrides: Any) -> dict[str, Any]:
    """A minimal valid fixture, built here so guard tests don't depend on real data."""

    case: dict[str, Any] = {
        "case_id": "sample",
        "description": "test case",
        "repo_url": "https://github.com/curl/curl",
        "tag": "curl-8_23_0",
        "commit": "a" * 40,
        "report": {
            "report_id": "r1",
            "source": "markdown",
            "body": "The function `curl_easy_init` exists.",
        },
        "requested_version": None,
        "expected_claims": [
            {
                "claim": {
                    "kind": "symbol",
                    "claim_id": "c1",
                    "statement": "curl_easy_init exists",
                    "source_excerpt": "`curl_easy_init`",
                    "symbol": "curl_easy_init",
                },
                "expected_status": "CONFIRMED",
                "evidence": [
                    {"kind": "line", "path": "lib/easy.c", "line": 1, "text": "x"}
                ],
            }
        ],
    }
    case.update(overrides)
    return case


def test_every_fixture_loads_into_core_models() -> None:
    cases = load_all()

    assert cases
    for case in cases:
        assert isinstance(case.report, NormalizedReport)
        for expected in case.expected_claims:
            assert isinstance(expected.claim, ClaimBase)


def test_terminal_label_without_evidence_is_rejected() -> None:
    case = make_case()
    case["expected_claims"][0]["evidence"] = []

    with pytest.raises(ValidationError, match="CONFIRMED needs evidence"):
        GoldenCase.model_validate(case)


def test_cant_check_label_needs_no_evidence() -> None:
    case = make_case()
    case["expected_claims"][0]["expected_status"] = "CANT_CHECK"
    case["expected_claims"][0]["evidence"] = []

    GoldenCase.model_validate(case)


def test_excerpt_missing_from_report_is_rejected() -> None:
    case = make_case()
    case["expected_claims"][0]["claim"]["source_excerpt"] = "not in the report"

    with pytest.raises(ValidationError, match="source_excerpt not found"):
        GoldenCase.model_validate(case)


def test_duplicate_claim_id_is_rejected() -> None:
    case = make_case()
    case["expected_claims"].append(case["expected_claims"][0])

    with pytest.raises(ValidationError, match="duplicate claim_id: c1"):
        GoldenCase.model_validate(case)


def test_claims_without_a_version_are_rejected() -> None:
    case = make_case(tag=None, commit=None)

    with pytest.raises(ValidationError, match="cases with claims need a tag and commit"):
        GoldenCase.model_validate(case)


def test_case_id_must_match_file_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "other_name.json").write_text(json.dumps(make_case()))
    monkeypatch.setattr(golden_fixtures, "FIXTURES_DIR", tmp_path)

    with pytest.raises(ValueError, match="other_name.json: case_id is 'sample'"):
        load_all()


# The coverage checklist: what #8, #10, #11 and #12 rely on the fixtures to
# contain. Each row is checked across all cases together, so a case can be
# edited or removed freely as long as every row stays covered.

Rule = Callable[[tuple[GoldenCase, ...]], bool]


def _all_claims(cases: tuple[GoldenCase, ...]) -> list[ExpectedClaim]:
    return [expected for case in cases for expected in case.expected_claims]


def _kind_with_status(kind: ClaimKind, status: EvidenceStatus) -> Rule:
    return lambda cases: any(
        e.claim.kind == kind and e.expected_status is status
        for e in _all_claims(cases)
    )


def _is_negated(claim: ClaimBase) -> bool:
    return (
        (isinstance(claim, SymbolClaim) and not claim.expected_present)
        or (isinstance(claim, CallPathClaim) and not claim.expected_reachable)
        or (isinstance(claim, BuildFlagClaim) and not claim.expected_enabled)
    )


def _is_rename(expected: ExpectedClaim) -> bool:
    # Gone from the tree, but a line at an older commit shows it existed.
    return isinstance(expected.claim, SymbolClaim) and any(
        isinstance(ev, AbsentEvidence) and ev.scope == "tree" for ev in expected.evidence
    ) and any(
        isinstance(ev, LineEvidence) and ev.commit is not None for ev in expected.evidence
    )


def _is_made_up(expected: ExpectedClaim) -> bool:
    return (
        isinstance(expected.claim, SymbolClaim)
        and expected.expected_status is EvidenceStatus.CONTRADICTED
    ) and any(
        isinstance(ev, AbsentEvidence) and ev.scope == "history"
        for ev in expected.evidence
    )


def _is_real_true_report(case: GoldenCase) -> bool:
    statuses = {e.expected_status for e in case.expected_claims}
    return (
        case.report.source_url is not None
        and EvidenceStatus.CONFIRMED in statuses
        and EvidenceStatus.CONTRADICTED not in statuses
    )


COVERAGE: dict[str, Rule] = {
    **{
        f"{kind.value} {status.value}": _kind_with_status(kind, status)
        for kind in (
            ClaimKind.SYMBOL,
            ClaimKind.FILE_LINE,
            ClaimKind.CALL_PATH,
            ClaimKind.BUILD_FLAG,
        )
        for status in (EvidenceStatus.CONFIRMED, EvidenceStatus.CONTRADICTED)
    },
    "negated claim": lambda cases: any(
        _is_negated(e.claim) for e in _all_claims(cases)
    ),
    "precise CANT_CHECK claim": lambda cases: any(
        e.expected_status is EvidenceStatus.CANT_CHECK for e in _all_claims(cases)
    ),
    "renamed symbol": lambda cases: any(_is_rename(e) for e in _all_claims(cases)),
    "made-up symbol": lambda cases: any(_is_made_up(e) for e in _all_claims(cases)),
    "real public report that must not be CONTRADICTED": lambda cases: any(
        _is_real_true_report(case) for case in cases
    ),
    "prompt injection": lambda cases: any(
        "ignore previous instructions" in case.report.body.lower() for case in cases
    ),
    "vague report with no claims and no version": lambda cases: any(
        not case.expected_claims and case.requested_version is None for case in cases
    ),
}


@pytest.mark.parametrize("row", COVERAGE)
def test_fixtures_cover(row: str) -> None:
    assert COVERAGE[row](load_all()), f"no golden case covers: {row}"
