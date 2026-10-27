"""Intake adapter for GitHub security advisory payloads.

GitHub publishes the same advisory through REST (snake_case) and GraphQL
(camelCase), and GraphQL wraps it in one of several envelopes:

    {"ghsa_id": ...}                                    REST advisory
    {"repositoryAdvisory": {...}}                       GraphQL node
    {"data": {"securityAdvisory": {...}}}               GraphQL query
    {"data": {"repository": {"securityAdvisories":
        {"nodes": [...]}}}}                             GraphQL connection

One adapter unwraps every shape and reads fields by alias, so callers do not
need to know which API produced the export.  A connection with zero or several
advisories is rejected rather than guessed at: normalizing the wrong advisory
would attach evidence to the wrong report.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from triage0.core.models import NormalizedReport, ReportSource
from triage0.intake._common import (
    IntakeError,
    decode_json,
    derive_report_id,
    first_line,
    first_raw_text,
    first_text,
    first_timestamp,
    json_safe,
    labelled_value,
)

__all__ = ["GitHubAdvisoryAdapter"]

_BODY_KEYS = ("description", "body")
_TITLE_KEYS = ("summary", "title")
_ID_KEYS = ("ghsa_id", "ghsaId", "advisory_id")
_URL_KEYS = ("html_url", "htmlUrl", "permalink", "url")
_DATE_KEYS = ("published_at", "publishedAt", "created_at", "createdAt")

_VERSION_LABELS = (
    "affected version",
    "affected versions",
    "vulnerable version",
    "vulnerable versions",
    "product version",
    "version",
)

_WRAPPER_KEYS = (
    "data",
    "repository",
    "securityAdvisory",
    "repositoryAdvisory",
    "securityAdvisories",
)

# Keys that identify an object as the advisory itself.  Unwrapping stops here so
# that an advisory's own ``repository`` field is not mistaken for another
# envelope.
_ADVISORY_KEYS = (
    "ghsa_id",
    "ghsaId",
    "summary",
    "description",
    "cve_id",
    "cveId",
    "vulnerabilities",
    "identifiers",
)

_METADATA_FIELDS: dict[str, tuple[str, ...]] = {
    "severity": ("severity",),
    "state": ("state",),
    "cve_id": ("cve_id", "cveId"),
    "published_at": ("published_at", "publishedAt"),
    "updated_at": ("updated_at", "updatedAt"),
    "identifiers": ("identifiers",),
    "cwes": ("cwes",),
}


def _looks_like_advisory(data: Mapping[str, Any]) -> bool:
    # A repository envelope can carry its own ``description`` next to
    # ``securityAdvisories``; an object that still wraps an advisory is never
    # the advisory itself, so keep descending.
    if "securityAdvisories" in data or "repositoryAdvisory" in data:
        return False
    return any(key in data for key in _ADVISORY_KEYS)


def _single_node(nodes: list[Any]) -> Mapping[str, Any]:
    if len(nodes) != 1:
        raise IntakeError(
            "expected exactly one security advisory, "
            f"found {len(nodes)}"
        )
    node = nodes[0]
    if not isinstance(node, Mapping):
        raise IntakeError("security advisory node is not a JSON object")
    return node


def _unwrap(data: dict[str, Any]) -> dict[str, Any]:
    """Return the advisory object from any supported payload envelope."""

    current: Mapping[str, Any] = data
    for _ in range(2 * len(_WRAPPER_KEYS)):
        if _looks_like_advisory(current):
            return dict(current)

        nodes = current.get("nodes")
        if isinstance(nodes, list):
            current = _single_node(nodes)
            continue

        for key in _WRAPPER_KEYS:
            nested = current.get(key)
            if isinstance(nested, Mapping):
                current = nested
                break
        else:
            break

    if not isinstance(current, Mapping) or not _looks_like_advisory(current):
        raise IntakeError(
            "GitHub payload does not contain a security advisory"
        )
    return dict(current)


def _vulnerability_items(value: object) -> list[Mapping[str, Any]]:
    """Return advisory vulnerability entries from a list or GraphQL connection."""

    if isinstance(value, Mapping):
        nodes = value.get("nodes")
        if isinstance(nodes, list):
            return [node for node in nodes if isinstance(node, Mapping)]
        return []
    if isinstance(value, list):
        return [item for item in value if isinstance(item, Mapping)]
    return []


def _first_across(
    items: Sequence[Mapping[str, Any]], keys: Sequence[str]
) -> str | None:
    for item in items:
        value = first_text(item, keys)
        if value is not None:
            return value
    return None


def _project(advisory: Mapping[str, Any]) -> str | None:
    items = _vulnerability_items(advisory.get("vulnerabilities"))
    for item in items:
        package = item.get("package")
        if isinstance(package, Mapping):
            name = package.get("name")
            if isinstance(name, str) and name.strip():
                return name.strip()

    repository = advisory.get("repository")
    if isinstance(repository, Mapping):
        owner = first_text(
            repository, ("nameWithOwner", "full_name", "name_with_owner")
        )
        if owner is not None:
            return owner

    return first_text(advisory, ("project", "product"))


def _version(advisory: Mapping[str, Any], body: str) -> str | None:
    items = _vulnerability_items(advisory.get("vulnerabilities"))

    affected = _first_across(
        items, ("vulnerable_version_range", "vulnerableVersionRange")
    )
    if affected is not None:
        return affected

    for item in items:
        patched = first_text(item, ("patched_versions", "patchedVersions"))
        if patched is not None:
            return patched
        first_patched = item.get("firstPatchedVersion")
        if isinstance(first_patched, Mapping):
            identifier = first_text(first_patched, ("identifier",))
            if identifier is not None:
                return identifier

    return labelled_value(_VERSION_LABELS, body)


def _metadata(advisory: Mapping[str, Any]) -> dict[str, Any]:
    metadata: dict[str, Any] = {"adapter": "github"}

    for name, keys in _METADATA_FIELDS.items():
        for key in keys:
            if advisory.get(key) is not None:
                metadata[name] = json_safe(advisory[key])
                break

    affected: list[dict[str, Any]] = []
    for item in _vulnerability_items(advisory.get("vulnerabilities")):
        entry: dict[str, Any] = {}
        package = item.get("package")
        if isinstance(package, Mapping):
            for key in ("name", "ecosystem"):
                if package.get(key) is not None:
                    entry[key] = json_safe(package[key])
        for key in ("vulnerable_version_range", "vulnerableVersionRange"):
            if item.get(key) is not None:
                entry["vulnerable_version_range"] = json_safe(item[key])
                break
        if entry:
            affected.append(entry)
    if affected:
        metadata["affected"] = affected

    return metadata


class GitHubAdvisoryAdapter:
    """Normalize a GitHub security advisory into a :class:`NormalizedReport`."""

    source = ReportSource.GITHUB_ADVISORY

    def normalize(
        self,
        payload: str | bytes | Mapping[str, Any],
        *,
        report_id: str | None = None,
        forced_source: ReportSource | None = None,
    ) -> NormalizedReport:
        advisory = _unwrap(decode_json(payload))

        body = first_raw_text(advisory, _BODY_KEYS)
        if body is None:
            raise IntakeError("GitHub advisory has no description")

        if report_id is None:
            external_id = first_text(advisory, _ID_KEYS)
            report_id = (
                external_id
                if external_id is not None
                else derive_report_id("gh", body)
            )

        return NormalizedReport(
            report_id=report_id,
            source=forced_source if forced_source is not None else self.source,
            project=_project(advisory),
            title=first_text(advisory, _TITLE_KEYS) or first_line(body),
            body=body,
            reported_version=_version(advisory, body),
            submitted_at=first_timestamp(advisory, _DATE_KEYS),
            source_url=first_text(advisory, _URL_KEYS),
            metadata=_metadata(advisory),
        )
