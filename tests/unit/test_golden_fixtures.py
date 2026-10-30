"""Unit tests for the golden fixture loader and the fixtures it loads."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from tests import golden_fixtures
from tests.golden_fixtures import GoldenCase, load_all
from triage0.core.models import ClaimBase, NormalizedReport


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


def test_case_id_must_match_file_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "other_name.json").write_text(json.dumps(make_case()))
    monkeypatch.setattr(golden_fixtures, "FIXTURES_DIR", tmp_path)

    with pytest.raises(ValueError, match="other_name.json: case_id is 'sample'"):
        load_all()
