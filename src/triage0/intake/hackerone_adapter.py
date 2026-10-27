"""Intake adapter for HackerOne JSON exports.

HackerOne exposes the same report in several shapes depending on how it was
exported: a flat object, a REST-style ``{"report": ...}`` envelope, or a
GraphQL ``{"data": {"report": ...}}`` envelope.  All three describe the same
fields, so one adapter unwraps the envelope and then reads consistent keys.

The vulnerability description lives in ``vulnerability_information``.  When a
report omits a structured version, the version is recovered from a labelled
``Version:`` line in that description, which is how many hand-written
submissions record it.
"""

from __future__ import annotations

from collections.abc import Mapping
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

__all__ = ["HackerOneAdapter"]

_BODY_KEYS = ("vulnerability_information", "body", "description")
_TITLE_KEYS = ("title", "summary")
_ID_KEYS = ("id", "report_id", "database_id")
_URL_KEYS = ("url", "report_url", "html_url")
_DATE_KEYS = ("created_at", "submitted_at", "reported_at")
_VERSION_KEYS = (
    "reported_version",
    "version",
    "affected_version",
    "affected_versions",
)
_PROJECT_KEYS = ("program", "product")

_VERSION_LABELS = (
    "affected version",
    "affected versions",
    "versions affected",
    "product version",
    "version",
)

# Custom fields arrive either as a mapping or as a list of ``{field, value}``
# entries; both spellings are normalized to these names.
_VERSION_FIELDS = frozenset(
    {
        "version",
        "reported_version",
        "affected_version",
        "affected_versions",
        "product_version",
    }
)

_METADATA_KEYS = (
    "severity",
    "state",
    "substate",
    "weakness",
    "disclosed_at",
    "bounty_amount",
    "has_bounty",
)


def _unwrap(data: dict[str, Any]) -> dict[str, Any]:
    """Return the report object from any supported export envelope."""

    candidate: object = data
    envelope = data.get("data")
    if isinstance(envelope, Mapping):
        candidate = envelope
    if isinstance(candidate, Mapping):
        report = candidate.get("report")
        if isinstance(report, Mapping):
            candidate = report
    if not isinstance(candidate, Mapping):
        raise IntakeError(
            "HackerOne payload does not contain a report object"
        )
    return dict(candidate)


def _project(data: Mapping[str, Any]) -> str | None:
    team = data.get("team")
    if isinstance(team, Mapping):
        for key in ("handle", "name", "slug"):
            value = team.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()

    program = data.get("program")
    if isinstance(program, Mapping):
        for key in ("handle", "name"):
            value = program.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()

    return first_text(data, _PROJECT_KEYS)


def _normalize_field_name(name: str) -> str:
    return name.strip().lower().replace(" ", "_").replace("-", "_")


def _custom_field_version(value: object) -> str | None:
    if isinstance(value, Mapping):
        found = first_text(value, _VERSION_KEYS)
        if found is not None:
            return found
        for key, item in value.items():
            if isinstance(key, str) and _normalize_field_name(key) in _VERSION_FIELDS:
                candidate = first_text({"value": item}, ("value",))
                if candidate is not None:
                    return candidate
        return None

    if isinstance(value, list):
        for item in value:
            if not isinstance(item, Mapping):
                continue
            name = item.get("field") or item.get("name") or item.get("key")
            if not isinstance(name, str):
                continue
            if _normalize_field_name(name) not in _VERSION_FIELDS:
                continue
            found = first_text(item, ("value", "values", "text"))
            if found is not None:
                return found
    return None


def _version(data: Mapping[str, Any], body: str) -> str | None:
    direct = first_text(data, _VERSION_KEYS)
    if direct is not None:
        return direct

    for key in ("custom_fields", "custom_field_values"):
        found = _custom_field_version(data.get(key))
        if found is not None:
            return found

    return labelled_value(_VERSION_LABELS, body)


def _metadata(data: Mapping[str, Any]) -> dict[str, Any]:
    metadata: dict[str, Any] = {"adapter": "hackerone"}
    for key in _METADATA_KEYS:
        if data.get(key) is not None:
            metadata[key] = json_safe(data[key])

    team = data.get("team")
    if isinstance(team, Mapping):
        handle = team.get("handle")
        if isinstance(handle, str) and handle.strip():
            metadata["team"] = handle.strip()

    scope = data.get("structured_scope")
    if isinstance(scope, Mapping):
        asset = scope.get("asset_identifier")
        if isinstance(asset, str) and asset.strip():
            metadata["asset"] = asset.strip()

    return metadata


class HackerOneAdapter:
    """Normalize a HackerOne report export into a :class:`NormalizedReport`."""

    source = ReportSource.HACKERONE

    def normalize(
        self,
        payload: str | bytes | Mapping[str, Any],
        *,
        report_id: str | None = None,
        forced_source: ReportSource | None = None,
    ) -> NormalizedReport:
        report = _unwrap(decode_json(payload))

        body = first_raw_text(report, _BODY_KEYS)
        if body is None:
            raise IntakeError(
                "HackerOne payload has no vulnerability_information"
            )

        external_id = first_text(report, _ID_KEYS)
        if report_id is None:
            report_id = (
                f"h1-{external_id}"
                if external_id is not None
                else derive_report_id("h1", body)
            )

        return NormalizedReport(
            report_id=report_id,
            source=forced_source if forced_source is not None else self.source,
            project=_project(report),
            title=first_text(report, _TITLE_KEYS) or first_line(body),
            body=body,
            reported_version=_version(report, body),
            submitted_at=first_timestamp(report, _DATE_KEYS),
            source_url=first_text(report, _URL_KEYS),
            metadata=_metadata(report),
        )
