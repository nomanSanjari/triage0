"""Origin matching and real-record construction.

An advisory's ``database_specific.issue`` link decides where the report text
comes from: a disclosed HackerOne report, a GitHub issue, or a fallback built
from the official advisory fields alone. This module also assembles the flat
dataset record the baseline loader reads.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from triage0.storage.indexers.h1_reports import (
    CURL_REPO_URL,
    GITHUB_HOSTS,
    VULN_JSON_URL,
    FetchPolicy,
    H1Error,
    fetch_public_json,
    fetch_public_report,
    report_sha256,
    report_text,
)
from triage0.storage.indexers.osv_advisories import (
    OsvError,
    OsvVulnerability,
    advisory_report_text,
    cve_id,
    osv_provenance,
)
from triage0.storage.indexers.osv_refs import RefResolution

__all__ = [
    "Origin",
    "build_real_record",
]

_H1_ISSUE = re.compile(r"https://hackerone\.com/reports/(\d+)/?")
_GITHUB_ISSUE = re.compile(r"https://github\.com/curl/curl/issues/(\d+)")


@dataclass(frozen=True)
class Origin:
    source: str
    source_url: str
    title: str
    body: str
    h1: Mapping[str, Any] | None = None
    github: Mapping[str, Any] | None = None
    fallback_reason: str | None = None


def _describe(cve: str, origin: Origin, ref: RefResolution) -> str:
    if origin.source == "hackerone" and origin.h1 is not None:
        return (
            f"Real curl CVE {cve}; original HackerOne report {origin.h1.get('id')}; "
            f"checked at last affected version {ref.reported_version}."
        )
    if origin.source == "github_issue" and origin.github is not None:
        return (
            f"Real curl CVE {cve}; original GitHub issue {origin.github.get('number')}; "
            f"checked at last affected version {ref.reported_version}."
        )
    return (
        f"Real curl CVE {cve}; no original report link, text derived from curl's "
        f"OSV advisory; checked at last affected version {ref.reported_version}."
    )


def _fallback_origin(osv: OsvVulnerability, reason: str) -> Origin:
    # The reason is recorded, not swallowed, so evaluation can stratify
    # original-report records from advisory-derived ones.
    cve = cve_id(osv)
    return Origin(
        source="osv_advisory",
        source_url=osv.database_specific.www or f"https://curl.se/docs/{cve}.html",
        title=osv.summary.strip(),
        body=advisory_report_text(osv),
        fallback_reason=reason,
    )


def _github_issue_fields(payload: Any, expected_number: int) -> tuple[str, str]:
    """Extract the report title and body from the GitHub API response.

    The payload is untrusted external JSON, so shape and blankness are
    validated here once rather than trusted downstream.
    """

    if not isinstance(payload, Mapping):
        raise OsvError(f"GitHub issue {expected_number} is not a JSON object")
    number = payload.get("number")
    title = payload.get("title")
    body = payload.get("body")
    # bool is an int subclass, so `True == 1` must not pass as issue 1.
    if (
        isinstance(number, bool)
        or not isinstance(number, int)
        or number != expected_number
    ):
        raise OsvError(f"GitHub issue {expected_number} returned a different number")
    if not isinstance(title, str) or not title.strip():
        raise OsvError(f"GitHub issue {expected_number} has no title")
    if not isinstance(body, str) or not body.strip():
        raise OsvError(f"GitHub issue {expected_number} has no body")
    return title.strip(), body.strip()


def _origin_for(
    osv: OsvVulnerability, policy: FetchPolicy, wait: Callable[[], None]
) -> Origin:
    """Resolve the report origin for one advisory.

    ``wait`` paces serial requests (the coordinator passes ``pacer.wait``);
    this module never imports the pacer itself.
    """

    cve = cve_id(osv)
    issue = osv.database_specific.issue
    if not issue:
        return _fallback_origin(osv, "no_original_report_link")

    stripped = issue.strip()
    h1_match = _H1_ISSUE.fullmatch(stripped)
    if h1_match is not None:
        report_id = int(h1_match.group(1))
        wait()
        try:
            report = fetch_public_report(report_id, policy)
        except H1Error as exc:
            print(f"  ! {cve}: HackerOne report {report_id} unavailable: {exc}")
            return _fallback_origin(osv, "h1_report_unavailable")
        return Origin(
            source="hackerone",
            source_url=f"https://hackerone.com/reports/{report_id}",
            title=report.title,
            body=report.body,
            h1=report.provenance(),
        )

    github_match = _GITHUB_ISSUE.fullmatch(stripped)
    if github_match is not None:
        number = int(github_match.group(1))
        wait()
        try:
            payload = fetch_public_json(
                f"https://api.github.com/repos/curl/curl/issues/{number}",
                policy,
                allowed_hosts=GITHUB_HOSTS,
            )
            title, body = _github_issue_fields(payload, number)
        except (H1Error, OsvError) as exc:
            print(f"  ! {cve}: GitHub issue {number} unavailable: {exc}")
            return _fallback_origin(osv, "github_issue_unavailable")
        return Origin(
            source="github_issue",
            source_url=f"https://github.com/curl/curl/issues/{number}",
            title=title,
            body=body,
            github={"number": number, "title": title, "state": payload.get("state")},
        )

    print(f"  ! {cve}: unsupported issue link {stripped!r}")
    return _fallback_origin(osv, "unrecognized_issue_link")


def build_real_record(
    osv: OsvVulnerability,
    ref: RefResolution,
    origin: Origin,
    *,
    commit_verified: bool | None = None,
) -> dict[str, Any]:
    """Assemble one flat dataset record for a real curl CVE.

    The record is flat because the baseline loader only reads a string
    ``report`` field. In provenance, ``ref_verified``/``commit_verified`` are
    ``None`` when no verification ran and ``False`` when the commit was
    checked and found missing; a False is never upgraded to True.
    """

    cve = cve_id(osv)
    text = report_text(origin.title, origin.body)
    # A tag without a verification result is still unverified; never claim it.
    ref_verified = commit_verified if ref.commit is not None else None

    provenance: dict[str, Any] = {
        "label_source": VULN_JSON_URL,
        "origin": origin.source,
        "ref_source": ref.ref_source,
        "ref_verified": ref_verified,
        "commit_verified": ref_verified,
        "fallback_reason": origin.fallback_reason,
        "report_sha256": report_sha256(text),
        "osv": osv_provenance(osv),
    }
    if origin.h1 is not None:
        provenance["h1"] = dict(origin.h1)
    if origin.github is not None:
        provenance["github"] = dict(origin.github)

    return {
        "id": cve,
        "case_id": cve,
        "description": _describe(cve, origin, ref),
        "ground_truth": "real",
        "repo_url": CURL_REPO_URL,
        "tag": ref.tag,
        "commit": ref.commit,
        "report": text,
        "report_id": cve,
        "source": origin.source,
        "project": "curl",
        "title": origin.title,
        "reported_version": ref.reported_version,
        "source_url": origin.source_url,
        "provenance": provenance,
    }
