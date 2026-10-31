"""Tests for OSV advisory parsing, version ranges, and advisory text."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from triage0.storage.indexers import osv_advisories
from triage0.storage.indexers.h1_reports import FetchPolicy
from triage0.storage.indexers.osv_advisories import (
    OsvError,
    OsvVulnerability,
    advisory_report_text,
    cve_id,
    fix_commits,
    fixed_versions,
    introduced_commits,
    introduced_versions,
    load_vuln_json,
    parse_vuln_bundle,
    semver_ranges,
)

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


# --- parsing ---------------------------------------------------------------


def test_parse_vuln_bundle_accepts_array_and_wrapper() -> None:
    payload = osv_payload()
    assert len(parse_vuln_bundle(json.dumps([payload]))) == 1
    assert len(parse_vuln_bundle(json.dumps({"vulnerabilities": [payload]}))) == 1


def test_parse_vuln_bundle_requires_last_affected() -> None:
    payload = osv_payload()
    del payload["database_specific"]["last_affected"]
    with pytest.raises(OsvError, match="last_affected"):
        parse_vuln_bundle(json.dumps([payload]))


def test_parse_vuln_bundle_rejects_duplicate_cves() -> None:
    payload = osv_payload()
    with pytest.raises(OsvError, match="duplicate advisory"):
        parse_vuln_bundle(json.dumps([payload, payload]))


def test_parse_vuln_bundle_rejects_a_blank_summary() -> None:
    payload = osv_payload(summary="   ")
    with pytest.raises(OsvError, match="summary"):
        parse_vuln_bundle(json.dumps([payload]))


@pytest.mark.parametrize(
    "text",
    [
        "not json",
        '{"vulnerabilities": "nope"}',
        "[]",
    ],
)
def test_parse_vuln_bundle_rejects_a_malformed_bundle(text: str) -> None:
    with pytest.raises(OsvError):
        parse_vuln_bundle(text)


def test_load_vuln_json_reads_a_local_snapshot_without_network(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    path = tmp_path / "vuln.json"
    path.write_text(json.dumps([osv_payload()]), encoding="utf-8")

    def fail(*args: object, **kwargs: object) -> None:
        raise AssertionError("a local snapshot must not use the network")

    monkeypatch.setattr(osv_advisories, "fetch_public_text", fail)

    loaded = load_vuln_json(path, FetchPolicy(delay_seconds=0))
    assert [cve_id(osv) for osv in loaded] == ["CVE-2026-92391"]


def test_cve_id_requires_a_cve_alias() -> None:
    with pytest.raises(OsvError, match="no CVE alias"):
        cve_id(osv(aliases=["CURL-CVE-2026-92391"]))


# --- version ranges --------------------------------------------------------


def test_semver_ranges_preserve_backports() -> None:
    assert semver_ranges(osv()) == [
        ("7.56.0", "8.14.3"),
        ("8.15.0", "8.16.2"),
        ("8.17.0", "8.20.2"),
        ("8.21.0", "8.23.0"),
    ]


def test_fixed_and_introduced_versions_keep_order() -> None:
    assert fixed_versions(osv()) == ["8.14.3", "8.16.2", "8.20.2", "8.23.0"]
    assert introduced_versions(osv()) == ["7.56.0", "8.15.0", "8.17.0", "8.21.0"]


def test_fix_commits_are_empty_without_a_git_range() -> None:
    payload = osv_payload()
    payload["affected"][0]["ranges"] = payload["affected"][0]["ranges"][:1]
    assert fix_commits(OsvVulnerability.model_validate(payload)) == []
    assert introduced_commits(OsvVulnerability.model_validate(payload)) == []


def test_fix_commits_are_collected_from_git_ranges() -> None:
    assert fix_commits(osv()) == [FIX_SHA]
    assert introduced_commits(osv()) == [INTRO_SHA]


# --- advisory text ---------------------------------------------------------


def test_advisory_report_text_is_built_only_from_advisory_fields() -> None:
    text = advisory_report_text(osv())

    assert text.startswith("curl_mime_free stack overflow by recursion")
    assert "introduced 7.56.0, fixed in 8.14.3" in text
    assert "Last affected: 8.22.0" in text
    assert "Fixed in: 8.14.3, 8.16.2, 8.20.2, 8.23.0" in text
    assert "CWE: CWE-674 Uncontrolled Recursion" in text
    assert "https://curl.se/docs/CVE-2026-92391.html" in text
    assert text.endswith("crashes on cleanup.")
