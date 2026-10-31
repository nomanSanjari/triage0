"""OSV advisory models, parsing, and version-range extraction.

curl publishes every advisory as OSV at ``https://curl.se/docs/vuln.json``.
The issue comment for this dataset is explicit: the version to check is
``database_specific.last_affected``, never a ``fixed`` version. A single
advisory can contain several introduced/fixed pairs because fixes were
backported, so "the last fixed event" is not a boundary.

This module owns the advisory schema. It imports nothing from the other
indexer modules, so everything else can depend on it.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
)

from triage0.storage.indexers.h1_reports import (
    CURL_HOSTS,
    VULN_JSON_URL,
    FetchPolicy,
    fetch_public_text,
)

__all__ = [
    "OsvAffected",
    "OsvCwe",
    "OsvDatabaseSpecific",
    "OsvError",
    "OsvEvent",
    "OsvRange",
    "OsvVulnerability",
    "advisory_report_text",
    "cve_id",
    "fix_commits",
    "fixed_versions",
    "introduced_commits",
    "introduced_versions",
    "load_vuln_json",
    "osv_git_commits",
    "osv_provenance",
    "parse_vuln_bundle",
    "semver_ranges",
]

# Git hashes in OSV GIT ranges: SHA-1 (40 chars) or SHA-256 (40 + 24 = 64).
# OSV publishes both forms depending on how the advisory was authored.
SHA = re.compile(r"[0-9a-f]{40}(?:[0-9a-f]{24})?")


class OsvError(RuntimeError):
    """The advisory file or a git reference did not have the expected shape."""


class OsvEvent(BaseModel):
    model_config = ConfigDict(frozen=True, extra="allow")

    introduced: str | None = None
    fixed: str | None = None


class OsvRange(BaseModel):
    model_config = ConfigDict(frozen=True, extra="allow")

    type: str
    events: list[OsvEvent] = Field(default_factory=list)


class OsvAffected(BaseModel):
    model_config = ConfigDict(frozen=True, extra="allow")

    ranges: list[OsvRange] = Field(default_factory=list)


class OsvCwe(BaseModel):
    model_config = ConfigDict(frozen=True, extra="allow")

    id: str | None = None
    desc: str | None = None


class OsvDatabaseSpecific(BaseModel):
    model_config = ConfigDict(frozen=True, extra="allow", populate_by_name=True)

    last_affected: str
    issue: str | None = None
    www: str | None = None
    severity: str | None = None
    affects: str | None = None
    # vuln.json spells the key uppercase, so accept "CWE" as well as "cwe".
    cwe: OsvCwe | None = Field(default=None, alias="CWE")

    @field_validator("last_affected")
    @classmethod
    def _last_affected_not_blank(cls, value: str) -> str:
        text = value.strip()
        if not text:
            raise ValueError("last_affected must not be blank")
        return text


class OsvVulnerability(BaseModel):
    model_config = ConfigDict(frozen=True, extra="allow")

    id: str
    aliases: list[str] = Field(min_length=1)
    summary: str = Field(min_length=1)
    details: str = ""
    published: str | None = None
    modified: str | None = None
    database_specific: OsvDatabaseSpecific
    affected: list[OsvAffected] = Field(default_factory=list)

    @field_validator("summary")
    @classmethod
    def _summary_not_blank(cls, value: str) -> str:
        text = value.strip()
        if not text:
            raise ValueError("summary must not be blank")
        return text


def _ordered_unique(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return ordered


def cve_id(osv: OsvVulnerability) -> str:
    for alias in osv.aliases:
        if alias.startswith("CVE-"):
            return alias
    raise OsvError(f"{osv.id} has no CVE alias")


def semver_ranges(osv: OsvVulnerability) -> list[tuple[str | None, str | None]]:
    """Return every introduced/fixed pair, preserving backports.

    Only completed pairs are returned: an introduced event whose range has no
    fix is dropped, because ``last_affected`` is the evaluation boundary and
    an unfixed range contributes no version information.
    """

    ranges: list[tuple[str | None, str | None]] = []
    for affected in osv.affected:
        for version_range in affected.ranges:
            if version_range.type != "SEMVER":
                continue
            introduced: str | None = None
            for event in version_range.events:
                if event.introduced is not None:
                    introduced = event.introduced
                elif event.fixed is not None:
                    ranges.append((introduced, event.fixed))
                    introduced = None
    return ranges


def fixed_versions(osv: OsvVulnerability) -> list[str]:
    return _ordered_unique(
        fixed for _, fixed in semver_ranges(osv) if fixed is not None
    )


def introduced_versions(osv: OsvVulnerability) -> list[str]:
    return _ordered_unique(
        introduced
        for introduced, _ in semver_ranges(osv)
        # OSV uses "0" for "before any release"; listing it is noise.
        if introduced is not None and introduced != "0"
    )


def fix_commits(osv: OsvVulnerability) -> list[str]:
    return _ordered_unique(
        event.fixed
        for affected in osv.affected
        for version_range in affected.ranges
        if version_range.type == "GIT"
        for event in version_range.events
        if event.fixed is not None and SHA.fullmatch(event.fixed)
    )


def introduced_commits(osv: OsvVulnerability) -> list[str]:
    return _ordered_unique(
        event.introduced
        for affected in osv.affected
        for version_range in affected.ranges
        if version_range.type == "GIT"
        for event in version_range.events
        if event.introduced is not None and SHA.fullmatch(event.introduced)
    )


def osv_git_commits(osv: OsvVulnerability) -> list[str]:
    """Return every unique GIT hash the advisory publishes, in OSV order."""

    return _ordered_unique([*fix_commits(osv), *introduced_commits(osv)])


def advisory_report_text(osv: OsvVulnerability) -> str:
    """Build report text from official advisory fields only.

    Used when no original report is available. Every line comes from the OSV
    record; nothing is inferred.
    """

    lines = [osv.summary.strip(), ""]
    ranges = semver_ranges(osv)
    if ranges:
        lines.append(
            "Version ranges: "
            + "; ".join(
                f"introduced {introduced or 'unknown'}, fixed in {fixed or 'unfixed'}"
                for introduced, fixed in ranges
            )
        )
    lines.append(f"Last affected: {osv.database_specific.last_affected}")
    fixed = fixed_versions(osv)
    if fixed:
        lines.append("Fixed in: " + ", ".join(fixed))
    cwe = osv.database_specific.cwe
    if cwe is not None:
        label = " ".join(part for part in (cwe.id, cwe.desc) if part)
        if label:
            lines.append(f"CWE: {label}")
    if osv.database_specific.www:
        lines.append(f"Advisory: {osv.database_specific.www}")
    details = osv.details.strip()
    if details:
        lines.extend(["", details])
    return "\n".join(lines).strip()


def osv_provenance(osv: OsvVulnerability) -> dict[str, Any]:
    """Copy the advisory fields the dataset needs for audit.

    The raw ``ranges`` are kept as published, not just the derived versions, so
    a reviewer can confirm every backported introduced/fixed pair survived.
    """

    cwe = osv.database_specific.cwe
    return {
        "id": osv.id,
        "aliases": list(osv.aliases),
        "summary": osv.summary,
        "published": osv.published,
        "modified": osv.modified,
        "severity": osv.database_specific.severity,
        "cwe": None if cwe is None else {"id": cwe.id, "desc": cwe.desc},
        "affects": osv.database_specific.affects,
        "last_affected": osv.database_specific.last_affected,
        "fixed_versions": fixed_versions(osv),
        "introduced_versions": introduced_versions(osv),
        "fix_commits": fix_commits(osv),
        "introduced_commits": introduced_commits(osv),
        "ranges": [
            version_range.model_dump()
            for affected in osv.affected
            for version_range in affected.ranges
        ],
        "advisory_url": osv.database_specific.www,
    }


def _error_summary(exc: ValidationError) -> str:
    return "; ".join(
        f"{'.'.join(str(part) for part in error.get('loc', ())) or '<root>'}: "
        f"{error.get('msg', 'invalid value')}"
        for error in exc.errors()[:4]
    )


def parse_vuln_bundle(text: str) -> list[OsvVulnerability]:
    """Validate curl's ``vuln.json`` and return one record per CVE."""

    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise OsvError(f"invalid vuln.json: {exc}") from exc

    if isinstance(data, dict):
        entries = data.get("vulnerabilities")
        if not isinstance(entries, list):
            raise OsvError("vuln.json has no 'vulnerabilities' array")
    elif isinstance(data, list):
        entries = data
    else:
        raise OsvError("vuln.json is not an array or object")

    vulnerabilities: list[OsvVulnerability] = []
    seen: set[str] = set()
    for entry in entries:
        try:
            osv = OsvVulnerability.model_validate(entry)
        except ValidationError as exc:
            raise OsvError(f"invalid advisory: {_error_summary(exc)}") from exc
        identifier = cve_id(osv)
        if identifier in seen:
            raise OsvError(f"duplicate advisory for {identifier}")
        seen.add(identifier)
        vulnerabilities.append(osv)
    if not vulnerabilities:
        raise OsvError("vuln.json contains no advisories")
    return vulnerabilities


def load_vuln_json(
    path: Path | None, policy: FetchPolicy
) -> list[OsvVulnerability]:
    """Load advisories from a local snapshot or from curl.se."""

    if path is not None:
        try:
            text = path.read_text(encoding="utf-8")
        except OSError as exc:
            raise OsvError(f"cannot read {path}: {exc}") from exc
    else:
        text = fetch_public_text(VULN_JSON_URL, policy, allowed_hosts=CURL_HOSTS)
    return parse_vuln_bundle(text)
