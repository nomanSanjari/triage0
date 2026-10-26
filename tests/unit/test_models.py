"""Unit tests for the shared report, claim, and plan models."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from pydantic import TypeAdapter, ValidationError

from triage0.core.evidence import DeterministicProof, EvidenceResult, EvidenceStatus
from triage0.core.models import (
    AnyClaim,
    BuildFlagClaim,
    BuildSystem,
    CallPathClaim,
    ClaimKind,
    FileLineClaim,
    NormalizedReport,
    PoCClaim,
    ReportSource,
    SymbolClaim,
    SymbolKind,
    VerificationPlan,
)

TREE_HASH = "a" * 40


def make_proof() -> DeterministicProof:
    return DeterministicProof(
        tool="git",
        argv=("git", "grep", "-n", "curl_easy_init"),
        exit_code=0,
        git_tree_hash=TREE_HASH,
    )


def claims() -> tuple[AnyClaim, ...]:
    return (
        SymbolClaim(
            claim_id="c-symbol",
            statement="curl_easy_init is defined",
            symbol="curl_easy_init",
            symbol_kind=SymbolKind.FUNCTION,
        ),
        FileLineClaim(
            claim_id="c-file",
            statement="The bounds check is missing",
            path="lib/url.c",
            line_start=100,
            line_end=110,
        ),
        CallPathClaim(
            claim_id="c-call",
            statement="main reaches the vulnerable helper",
            entrypoint="main",
            target="parse_url",
        ),
        BuildFlagClaim(
            claim_id="c-build",
            statement="ASan is enabled by default",
            build_system=BuildSystem.CMAKE,
            flag="-fsanitize=address",
        ),
        PoCClaim(
            claim_id="c-poc",
            statement="The provided payload crashes curl",
            steps=("curl --data @payload.txt https://example.test",),
            expected_outcome="heap-buffer-overflow",
            sanitizer="address",
        ),
    )


def make_plan(**overrides: object) -> VerificationPlan:
    data: dict[str, object] = {
        "plan_id": "plan-1",
        "report_id": "report-1",
        "repo_url": "https://github.com/curl/curl.git",
        "project": "curl",
        "requested_version": "curl 8.13.0",
        "resolved_ref": "curl-8_13_0",
        "claims": claims(),
    }
    data.update(overrides)
    return VerificationPlan(**data)


def test_normalized_report_minimal() -> None:
    report = NormalizedReport(
        report_id="report-1",
        source=ReportSource.MARKDOWN,
        body="curl crashes when...",
    )
    assert report.project is None
    assert report.title is None
    assert report.content_hash == report.content_hash


def test_normalized_report_accepts_enum_strings() -> None:
    report = NormalizedReport(
        report_id="report-1",
        source="hackerone",
        body="body",
    )
    assert report.source is ReportSource.HACKERONE


def test_content_hash_is_stable_and_body_sensitive() -> None:
    first = NormalizedReport(
        report_id="a", source=ReportSource.MARKDOWN, body="same body"
    )
    second = NormalizedReport(
        report_id="b", source=ReportSource.EMAIL, body="same body"
    )
    third = NormalizedReport(
        report_id="a", source=ReportSource.MARKDOWN, body="different body"
    )
    assert first.content_hash == second.content_hash
    assert first.content_hash != third.content_hash
    assert first.model_dump()["content_hash"] == first.content_hash


def test_report_body_must_not_be_blank() -> None:
    with pytest.raises(ValidationError, match="body must contain report text"):
        NormalizedReport(
            report_id="report-1", source=ReportSource.MARKDOWN, body="   "
        )


def test_normalized_report_round_trips_through_json() -> None:
    report = NormalizedReport(
        report_id="report-1",
        source=ReportSource.MARKDOWN,
        body="body",
    )
    restored = NormalizedReport.model_validate_json(report.model_dump_json())
    assert restored == report
    assert restored.content_hash == report.content_hash


def test_submitted_at_must_be_timezone_aware() -> None:
    with pytest.raises(ValidationError, match="timezone-aware"):
        NormalizedReport(
            report_id="report-1",
            source=ReportSource.MARKDOWN,
            body="body",
            submitted_at=datetime(2026, 1, 1, 12, 0, 0),
        )


def test_verification_plan_round_trips_through_json() -> None:
    plan = make_plan()
    restored = VerificationPlan.model_validate_json(plan.model_dump_json())

    assert restored == plan
    assert isinstance(restored.claims[0], SymbolClaim)
    assert isinstance(restored.claims[1], FileLineClaim)
    assert isinstance(restored.claims[2], CallPathClaim)
    assert isinstance(restored.claims[3], BuildFlagClaim)
    assert isinstance(restored.claims[4], PoCClaim)


def test_claim_union_discriminates_on_kind() -> None:
    adapter: TypeAdapter[AnyClaim] = TypeAdapter(AnyClaim)
    claim = adapter.validate_python(
        {
            "kind": "file_line",
            "claim_id": "c1",
            "statement": "line exists",
            "path": "src/a.c",
            "line_start": 4,
        }
    )
    assert isinstance(claim, FileLineClaim)
    assert claim.effective_line_end == 4


def test_claim_kind_values() -> None:
    assert {kind.value for kind in ClaimKind} == {
        "symbol",
        "file_line",
        "call_path",
        "build_flag",
        "poc",
    }
    assert [claim.kind for claim in claims()] == [
        "symbol",
        "file_line",
        "call_path",
        "build_flag",
        "poc",
    ]


def test_file_line_range_must_be_ordered() -> None:
    with pytest.raises(ValidationError, match="line_end must be greater"):
        FileLineClaim(
            claim_id="c1",
            statement="bad range",
            path="src/a.c",
            line_start=10,
            line_end=5,
        )


def test_file_line_end_defaults_to_start() -> None:
    claim = FileLineClaim(
        claim_id="c1",
        statement="one line",
        path="src/a.c",
        line_start=7,
    )
    assert claim.line_end is None
    assert claim.effective_line_end == 7


def test_poc_requires_at_least_one_step() -> None:
    with pytest.raises(ValidationError):
        PoCClaim(
            claim_id="c1",
            statement="poc",
            steps=(),
            expected_outcome="crash",
        )


def test_poc_rejects_blank_steps() -> None:
    with pytest.raises(ValidationError, match="must not be blank"):
        PoCClaim(
            claim_id="c1",
            statement="poc",
            steps=("   ",),
            expected_outcome="crash",
        )


def test_extra_fields_are_rejected() -> None:
    with pytest.raises(ValidationError):
        SymbolClaim(
            claim_id="c1",
            statement="symbol",
            symbol="main",
            unexpected="nope",
        )


def test_duplicate_claim_ids_are_rejected() -> None:
    duplicate = claims()[0].model_copy(update={"statement": "duplicate id"})
    with pytest.raises(ValidationError, match="duplicate claim_id"):
        make_plan(claims=(claims()[0], duplicate))


def test_with_result_returns_typed_copy() -> None:
    original = claims()[0]
    result = EvidenceResult(status=EvidenceStatus.CONFIRMED, proof=make_proof())

    updated = original.with_result(result)

    assert isinstance(updated, SymbolClaim)
    assert updated.result == result
    assert original.result is None
    assert updated.claim_id == original.claim_id


def test_verification_plan_requires_repo_url() -> None:
    with pytest.raises(ValidationError):
        make_plan(repo_url="")


def test_verification_plan_created_at_defaults_to_utc() -> None:
    plan = make_plan()
    assert plan.created_at.tzinfo is not None


def test_verification_plan_rejects_naive_created_at() -> None:
    with pytest.raises(ValidationError, match="timezone-aware"):
        make_plan(created_at=datetime(2026, 1, 1, 12, 0, 0))


def test_models_are_frozen() -> None:
    plan = make_plan()
    with pytest.raises(ValidationError, match="frozen"):
        plan.notes = "changed"  # type: ignore[misc]
