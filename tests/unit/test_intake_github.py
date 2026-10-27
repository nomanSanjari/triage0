"""Unit tests for the GitHub security advisory intake adapter."""

from __future__ import annotations

import copy
import json
from datetime import datetime, timezone
from typing import Any

import pytest

from triage0.core.interfaces import IntakeAdapter
from triage0.core.models import NormalizedReport, ReportSource
from triage0.intake import GitHubAdvisoryAdapter, IntakeError

BODY = (
    "Affected version: 8.13.0\n\n"
    "Calling curl_easy_escape with a long string overflows the heap."
)

REST_ADVISORY: dict[str, Any] = {
    "ghsa_id": "GHSA-xxxx-yyyy-zzzz",
    "cve_id": "CVE-2026-12345",
    "summary": "Heap overflow in curl_easy_escape",
    "description": BODY,
    "severity": "high",
    "state": "published",
    "html_url": (
        "https://github.com/curl/curl/security/advisories/"
        "GHSA-xxxx-yyyy-zzzz"
    ),
    "published_at": "2026-10-05T14:30:00Z",
    "identifiers": [{"type": "GHSA", "value": "GHSA-xxxx-yyyy-zzzz"}],
    "cwes": [{"cwe_id": "CWE-122", "name": "Heap-based Buffer Overflow"}],
    "vulnerabilities": [
        {
            "package": {"ecosystem": "other", "name": "curl"},
            "vulnerable_version_range": ">= 8.0.0, < 8.13.1",
            "patched_versions": "8.13.1",
        }
    ],
}

GRAPHQL_ADVISORY: dict[str, Any] = {
    "ghsaId": "GHSA-aaaa-bbbb-cccc",
    "summary": "Out-of-bounds read in libfoo",
    "description": BODY,
    "severity": "MODERATE",
    "permalink": (
        "https://github.com/example/libfoo/security/advisories/"
        "GHSA-aaaa-bbbb-cccc"
    ),
    "publishedAt": "2026-10-06T09:15:00Z",
    "identifiers": [{"type": "GHSA", "value": "GHSA-aaaa-bbbb-cccc"}],
    "vulnerabilities": {
        "nodes": [
            {
                "package": {"name": "libfoo", "ecosystem": "OTHER"},
                "vulnerableVersionRange": "< 2.4.2",
                "firstPatchedVersion": {"identifier": "2.4.2"},
            }
        ]
    },
}


def rest(**overrides: Any) -> dict[str, Any]:
    payload = copy.deepcopy(REST_ADVISORY)
    payload.update(overrides)
    return payload


def graphql(**overrides: Any) -> dict[str, Any]:
    payload = copy.deepcopy(GRAPHQL_ADVISORY)
    payload.update(overrides)
    return payload


def test_adapter_declares_github_source() -> None:
    assert (
        GitHubAdvisoryAdapter().source is ReportSource.GITHUB_ADVISORY
    )


def test_rest_advisory_is_normalized() -> None:
    report = GitHubAdvisoryAdapter().normalize(rest())

    assert isinstance(report, NormalizedReport)
    assert report.source is ReportSource.GITHUB_ADVISORY
    assert report.report_id == "GHSA-xxxx-yyyy-zzzz"
    assert report.title == "Heap overflow in curl_easy_escape"
    assert report.project == "curl"
    assert report.reported_version == ">= 8.0.0, < 8.13.1"
    assert report.source_url == REST_ADVISORY["html_url"]
    assert report.submitted_at == datetime(
        2026, 10, 5, 14, 30, tzinfo=timezone.utc
    )
    assert report.body == BODY


def test_graphql_repository_advisory_is_unwrapped() -> None:
    payload = {
        "data": {"repository": {"securityAdvisory": graphql()}}
    }

    report = GitHubAdvisoryAdapter().normalize(payload)

    assert report.report_id == "GHSA-aaaa-bbbb-cccc"
    assert report.project == "libfoo"
    assert report.reported_version == "< 2.4.2"
    assert report.source_url == GRAPHQL_ADVISORY["permalink"]


def test_graphql_root_advisory_is_unwrapped() -> None:
    payload = {"data": {"securityAdvisory": graphql()}}

    report = GitHubAdvisoryAdapter().normalize(payload)

    assert report.report_id == "GHSA-aaaa-bbbb-cccc"


def test_single_connection_node_is_unwrapped() -> None:
    payload = {
        "data": {
            "repository": {
                "securityAdvisories": {"nodes": [graphql()]}
            }
        }
    }

    report = GitHubAdvisoryAdapter().normalize(payload)

    assert report.report_id == "GHSA-aaaa-bbbb-cccc"


def test_repository_description_is_not_mistaken_for_the_advisory() -> None:
    advisory = graphql()
    payload = {
        "data": {
            "repository": {
                "description": "The curl repository",
                "securityAdvisories": {"nodes": [advisory]},
            }
        }
    }

    report = GitHubAdvisoryAdapter().normalize(payload)

    assert report.report_id == advisory["ghsaId"]
    assert report.title == advisory["summary"]
    assert report.body == BODY
    assert "curl repository" not in report.body


def test_empty_connection_is_rejected() -> None:
    payload = {
        "data": {"repository": {"securityAdvisories": {"nodes": []}}}
    }

    with pytest.raises(IntakeError, match="found 0"):
        GitHubAdvisoryAdapter().normalize(payload)


def test_ambiguous_connection_is_rejected() -> None:
    payload = {
        "data": {
            "repository": {
                "securityAdvisories": {"nodes": [graphql(), graphql()]}
            }
        }
    }

    with pytest.raises(IntakeError, match="found 2"):
        GitHubAdvisoryAdapter().normalize(payload)


def test_version_uses_first_patched_version_without_a_range() -> None:
    advisory = graphql()
    del advisory["vulnerabilities"]["nodes"][0]["vulnerableVersionRange"]

    report = GitHubAdvisoryAdapter().normalize(advisory)

    assert report.reported_version == "2.4.2"


def test_version_falls_back_to_a_body_label() -> None:
    report = GitHubAdvisoryAdapter().normalize(
        rest(vulnerabilities=[])
    )

    assert report.reported_version == "8.13.0"


def test_project_falls_back_to_repository_name() -> None:
    advisory = rest(vulnerabilities=[])
    advisory["repository"] = {"nameWithOwner": "curl/curl"}

    report = GitHubAdvisoryAdapter().normalize(advisory)

    assert report.project == "curl/curl"


def test_report_id_is_derived_without_a_ghsa_id() -> None:
    advisory = rest(ghsa_id=None)

    report = GitHubAdvisoryAdapter().normalize(advisory)

    assert report.report_id.startswith("gh-")
    assert report.report_id != "gh-"


def test_report_id_can_be_overridden() -> None:
    report = GitHubAdvisoryAdapter().normalize(
        rest(), report_id="advisory-7"
    )

    assert report.report_id == "advisory-7"


def test_missing_description_raises() -> None:
    advisory = rest()
    del advisory["description"]

    with pytest.raises(IntakeError, match="no description"):
        GitHubAdvisoryAdapter().normalize(advisory)


def test_blank_description_raises() -> None:
    with pytest.raises(IntakeError, match="no description"):
        GitHubAdvisoryAdapter().normalize(rest(description="   \n\n"))


def test_title_falls_back_to_first_description_line() -> None:
    advisory = rest()
    del advisory["summary"]

    report = GitHubAdvisoryAdapter().normalize(advisory)

    assert report.title == "Affected version: 8.13.0"


def test_camel_case_fields_are_read() -> None:
    report = GitHubAdvisoryAdapter().normalize(graphql())

    assert report.submitted_at == datetime(
        2026, 10, 6, 9, 15, tzinfo=timezone.utc
    )


def test_metadata_captures_advisory_context() -> None:
    report = GitHubAdvisoryAdapter().normalize(rest())

    assert report.metadata["adapter"] == "github"
    assert report.metadata["severity"] == "high"
    assert report.metadata["cve_id"] == "CVE-2026-12345"
    assert report.metadata["state"] == "published"
    assert report.metadata["identifiers"] == [
        {"type": "GHSA", "value": "GHSA-xxxx-yyyy-zzzz"}
    ]
    assert report.metadata["cwes"] == [
        {"cwe_id": "CWE-122", "name": "Heap-based Buffer Overflow"}
    ]
    assert report.metadata["affected"] == [
        {
            "name": "curl",
            "ecosystem": "other",
            "vulnerable_version_range": ">= 8.0.0, < 8.13.1",
        }
    ]


def test_json_text_is_accepted() -> None:
    report = GitHubAdvisoryAdapter().normalize(json.dumps(rest()))

    assert report.report_id == "GHSA-xxxx-yyyy-zzzz"


def test_invalid_json_raises() -> None:
    with pytest.raises(IntakeError, match="not valid JSON"):
        GitHubAdvisoryAdapter().normalize("{nope}")


def test_payload_without_an_advisory_raises() -> None:
    with pytest.raises(IntakeError, match="does not contain a security"):
        GitHubAdvisoryAdapter().normalize({"data": {"repository": None}})


def test_adapter_satisfies_intake_protocol() -> None:
    assert isinstance(GitHubAdvisoryAdapter(), IntakeAdapter)
