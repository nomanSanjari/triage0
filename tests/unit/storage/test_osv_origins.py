"""Tests for origin matching and real-record assembly."""

from __future__ import annotations

from typing import Any

import pytest

from triage0.storage.indexers import osv_origins
from triage0.storage.indexers.h1_reports import FetchPolicy, H1Error, H1Report
from triage0.storage.indexers.osv_advisories import (
    OsvVulnerability,
    advisory_report_text,
)
from triage0.storage.indexers.osv_origins import (
    Origin,
    _origin_for,
    build_real_record,
)
from triage0.storage.indexers.osv_refs import RefResolution
from triage0.storage.indexers.osv_sync import _Pacer

FIX_SHA = "d7546809d5bd548b69ec49fca26565c8f6f0d344"
INTRO_SHA = "ce0881edee3c78609eae49665fb70264d8786d29"


def osv_payload(**overrides: Any) -> dict[str, Any]:
    data: dict[str, Any] = {
        "schema_version": "1.5.0",
        "id": "CURL-CVE-2026-92391",
        "aliases": ["CVE-2026-92391"],
        "summary": "curl_mime_free stack overflow by recursion",
        "details": "A deeply nested multipart MIME structure crashes on cleanup.",
        "published": "2026-10-14T08:00:00.00Z",
        "modified": "2026-10-24T23:17:46.00Z",
        "database_specific": {
            "package": "curl",
            "affects": "lib",
            "URL": "https://curl.se/docs/CVE-2026-92391.json",
            "www": "https://curl.se/docs/CVE-2026-92391.html",
            "issue": "https://hackerone.com/reports/4024222",
            "CWE": {"id": "CWE-674", "desc": "Uncontrolled Recursion"},
            "last_affected": "8.22.0",
            "severity": "Low",
        },
        "affected": [
            {
                "ranges": [
                    {
                        "type": "SEMVER",
                        "events": [
                            {"introduced": "7.56.0"},
                            {"fixed": "8.14.3"},
                            {"introduced": "8.15.0"},
                            {"fixed": "8.16.2"},
                            {"introduced": "8.17.0"},
                            {"fixed": "8.20.2"},
                            {"introduced": "8.21.0"},
                            {"fixed": "8.23.0"},
                        ],
                    },
                    {
                        "type": "GIT",
                        "repo": "https://github.com/curl/curl.git",
                        "events": [
                            {"introduced": INTRO_SHA},
                            {"fixed": FIX_SHA},
                        ],
                    },
                ],
                "versions": ["8.22.0"],
            }
        ],
    }
    data.update(overrides)
    return data


def osv(**overrides: Any) -> OsvVulnerability:
    return OsvVulnerability.model_validate(osv_payload(**overrides))


def osv_with_issue(issue: str) -> OsvVulnerability:
    payload = osv_payload()
    payload["database_specific"]["issue"] = issue
    return OsvVulnerability.model_validate(payload)


def hackerone_origin() -> Origin:
    return Origin(
        source="hackerone",
        source_url="https://hackerone.com/reports/4024222",
        title="CVE-2026-92391: curl_mime_free stack overflow by recursion",
        body="Full public report body.",
        h1={"id": 4024222, "public": True},
    )


# --- origin matching -------------------------------------------------------


@pytest.mark.parametrize(
    "issue",
    [
        "https://hackerone.com/reports/4024222",
        "https://hackerone.com/reports/4024222/",
    ],
)
def test_origin_for_matches_hackerone_links(
    monkeypatch: pytest.MonkeyPatch, issue: str
) -> None:
    report = H1Report(
        id=4024222, title="t", body="b", public=True, team_handle="curl"
    )
    monkeypatch.setattr(osv_origins, "fetch_public_report", lambda *a, **k: report)

    origin = _origin_for(
        osv_with_issue(issue), FetchPolicy(delay_seconds=0), _Pacer(0).wait
    )

    assert origin.source == "hackerone"
    assert origin.source_url == "https://hackerone.com/reports/4024222"


def test_origin_for_matches_github_links(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        osv_origins,
        "fetch_public_json",
        lambda *a, **k: {
            "number": 2093,
            "title": "t",
            "body": "b",
            "state": "open",
        },
    )

    origin = _origin_for(
        osv_with_issue("https://github.com/curl/curl/issues/2093"),
        FetchPolicy(delay_seconds=0),
        _Pacer(0).wait,
    )

    assert origin.source == "github_issue"
    assert origin.source_url == "https://github.com/curl/curl/issues/2093"


def test_origin_for_falls_back_for_an_unsupported_link() -> None:
    origin = _origin_for(
        osv_with_issue("https://example.test/report/1"),
        FetchPolicy(delay_seconds=0),
        _Pacer(0).wait,
    )

    assert origin.source == "osv_advisory"
    assert origin.fallback_reason == "unrecognized_issue_link"


def test_origin_for_falls_back_when_the_report_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unavailable(*args: object, **kwargs: object) -> None:
        raise H1Error("HTTP 404")

    monkeypatch.setattr(osv_origins, "fetch_public_report", unavailable)

    origin = _origin_for(
        osv_with_issue("https://hackerone.com/reports/4024222"),
        FetchPolicy(delay_seconds=0),
        _Pacer(0).wait,
    )

    assert origin.source == "osv_advisory"
    assert origin.fallback_reason == "h1_report_unavailable"


def test_origin_for_rejects_a_boolean_github_number(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        osv_origins,
        "fetch_public_json",
        lambda *a, **k: {
            "number": True,
            "title": "t",
            "body": "b",
            "state": "open",
        },
    )

    origin = _origin_for(
        osv_with_issue("https://github.com/curl/curl/issues/1"),
        FetchPolicy(delay_seconds=0),
        _Pacer(0).wait,
    )

    assert origin.source == "osv_advisory"
    assert origin.fallback_reason == "github_issue_unavailable"


# --- record assembly -------------------------------------------------------


def test_build_real_record_pins_the_last_affected_commit() -> None:
    ref = RefResolution(
        tag="curl-8_22_0",
        commit="a" * 40,
        reported_version="8.22.0",
        ref_source="last_affected",
    )
    record = build_real_record(osv(), ref, hackerone_origin(), commit_verified=True)

    assert record["id"] == "CVE-2026-92391"
    assert record["ground_truth"] == "real"
    assert record["tag"] == "curl-8_22_0"
    assert record["commit"] == "a" * 40
    assert record["reported_version"] == "8.22.0"
    assert record["source"] == "hackerone"
    assert isinstance(record["report"], str)
    assert record["report"].startswith("CVE-2026-92391:")
    assert record["provenance"]["ref_source"] == "last_affected"
    assert record["provenance"]["ref_verified"] is True
    assert record["provenance"]["osv"]["fix_commits"] == [FIX_SHA]


def test_build_real_record_marks_advisory_fallbacks() -> None:
    ref = RefResolution(
        tag=None,
        commit=None,
        reported_version="8.22.0",
        ref_source="unresolved",
    )
    origin = Origin(
        source="osv_advisory",
        source_url="https://curl.se/docs/CVE-2026-92391.html",
        title="curl_mime_free stack overflow by recursion",
        body=advisory_report_text(osv()),
        fallback_reason="no_original_report_link",
    )
    record = build_real_record(osv(), ref, origin)

    assert record["source"] == "osv_advisory"
    assert record["tag"] is None
    assert record["commit"] is None
    assert record["provenance"]["fallback_reason"] == "no_original_report_link"
    assert record["provenance"]["ref_verified"] is None
    assert "h1" not in record["provenance"]


def test_build_real_record_does_not_claim_an_unverified_ref() -> None:
    ref = RefResolution(
        tag="curl-8_22_0",
        commit="a" * 40,
        reported_version="8.22.0",
        ref_source="last_affected",
    )
    record = build_real_record(osv(), ref, hackerone_origin())

    assert record["commit"] == "a" * 40
    assert record["provenance"]["ref_verified"] is None
    assert record["provenance"]["commit_verified"] is None


def test_build_real_record_keeps_official_hashes_after_failed_verification() -> None:
    ref = RefResolution(
        tag="curl-8_22_0",
        commit="a" * 40,
        reported_version="8.22.0",
        ref_source="last_affected",
    )
    record = build_real_record(
        osv(), ref, hackerone_origin(), commit_verified=False
    )

    assert record["commit"] == "a" * 40
    assert record["provenance"]["ref_verified"] is False
    assert record["provenance"]["osv"]["fix_commits"] == [FIX_SHA]
    assert record["provenance"]["osv"]["introduced_commits"] == [INTRO_SHA]
