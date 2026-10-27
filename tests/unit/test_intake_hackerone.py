"""Unit tests for the HackerOne intake adapter."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

import pytest

from triage0.core.interfaces import IntakeAdapter
from triage0.core.models import NormalizedReport, ReportSource
from triage0.intake import HackerOneAdapter, IntakeError

BODY = (
    "Affected version: 8.13.0\n\n"
    "Calling curl_easy_escape with a long string overflows the heap."
)


def make_report(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "id": 1234567,
        "title": "Heap overflow in curl_easy_escape",
        "vulnerability_information": BODY,
        "team": {"handle": "curl", "name": "curl"},
        "created_at": "2026-10-05T14:30:00.000Z",
        "url": "https://hackerone.com/reports/1234567",
        "severity": "high",
        "state": "triaged",
        "weakness": "CWE-122",
        "disclosed_at": "2026-10-06T00:00:00Z",
        "bounty_amount": "500.00",
    }
    payload.update(overrides)
    return payload


def test_adapter_declares_hackerone_source() -> None:
    assert HackerOneAdapter().source is ReportSource.HACKERONE


def test_flat_export_is_normalized() -> None:
    report = HackerOneAdapter().normalize(make_report())

    assert isinstance(report, NormalizedReport)
    assert report.source is ReportSource.HACKERONE
    assert report.report_id == "h1-1234567"
    assert report.title == "Heap overflow in curl_easy_escape"
    assert report.project == "curl"
    assert report.reported_version == "8.13.0"
    assert report.source_url == "https://hackerone.com/reports/1234567"
    assert report.submitted_at == datetime(
        2026, 10, 5, 14, 30, tzinfo=timezone.utc
    )
    assert report.body == BODY


def test_rest_wrapped_export_is_unwrapped() -> None:
    report = HackerOneAdapter().normalize({"report": make_report()})

    assert report.report_id == "h1-1234567"
    assert report.project == "curl"


def test_graphql_wrapped_export_is_unwrapped() -> None:
    payload = {"data": {"report": make_report()}}

    report = HackerOneAdapter().normalize(payload)

    assert report.report_id == "h1-1234567"
    assert report.project == "curl"


def test_json_text_and_bytes_are_accepted() -> None:
    payload = json.dumps(make_report())

    from_text = HackerOneAdapter().normalize(payload)
    from_bytes = HackerOneAdapter().normalize(
        b"\xef\xbb\xbf" + payload.encode("utf-8")
    )

    assert from_text.report_id == from_bytes.report_id


def test_report_id_is_derived_when_absent() -> None:
    payload = make_report()
    del payload["id"]

    report = HackerOneAdapter().normalize(payload)

    assert report.report_id.startswith("h1-")
    assert report.report_id != "h1-"


def test_report_id_can_be_overridden() -> None:
    report = HackerOneAdapter().normalize(
        make_report(), report_id="h1-custom"
    )

    assert report.report_id == "h1-custom"


def test_missing_vulnerability_information_raises() -> None:
    payload = make_report()
    del payload["vulnerability_information"]

    with pytest.raises(IntakeError, match="vulnerability_information"):
        HackerOneAdapter().normalize(payload)


def test_blank_vulnerability_information_raises() -> None:
    with pytest.raises(IntakeError, match="vulnerability_information"):
        HackerOneAdapter().normalize(
            make_report(vulnerability_information="   \n\n")
        )


def test_title_falls_back_to_first_body_line() -> None:
    payload = make_report()
    del payload["title"]

    report = HackerOneAdapter().normalize(payload)

    assert report.title == "Affected version: 8.13.0"


def test_team_handle_is_preferred_over_name() -> None:
    payload = make_report(team={"handle": "curl", "name": "curl project"})

    report = HackerOneAdapter().normalize(payload)

    assert report.project == "curl"


def test_team_name_is_used_without_handle() -> None:
    report = HackerOneAdapter().normalize(
        make_report(team={"name": "curl project"})
    )

    assert report.project == "curl project"


def test_program_is_used_without_team() -> None:
    payload = make_report(program="curl")
    del payload["team"]

    report = HackerOneAdapter().normalize(payload)

    assert report.project == "curl"


def test_version_comes_from_custom_fields_mapping() -> None:
    payload = make_report(custom_fields={"version": "8.12.1"})

    report = HackerOneAdapter().normalize(payload)

    assert report.reported_version == "8.12.1"


def test_version_comes_from_custom_field_list() -> None:
    payload = make_report(
        custom_field_values=[{"field": "Affected Version", "value": "8.12.2"}]
    )

    report = HackerOneAdapter().normalize(payload)

    assert report.reported_version == "8.12.2"


def test_version_comes_from_non_canonical_custom_field_key() -> None:
    payload = make_report(custom_fields={"Affected Version": "8.12.3"})

    report = HackerOneAdapter().normalize(payload)

    assert report.reported_version == "8.12.3"


def test_version_falls_back_to_body_label() -> None:
    report = HackerOneAdapter().normalize(
        make_report(
            vulnerability_information=(
                "**Affected version:** **8.11.0**\n\nDetails."
            )
        )
    )

    assert report.reported_version == "8.11.0"


def test_submitted_at_accepts_offset_less_timestamps() -> None:
    payload = make_report(created_at="2026-10-05T14:30:00")

    report = HackerOneAdapter().normalize(payload)

    assert report.submitted_at == datetime(
        2026, 10, 5, 14, 30, tzinfo=timezone.utc
    )


def test_submitted_at_uses_submitted_at_alias() -> None:
    payload = make_report(submitted_at="2026-10-07T09:00:00Z")
    del payload["created_at"]

    report = HackerOneAdapter().normalize(payload)

    assert report.submitted_at == datetime(
        2026, 10, 7, 9, 0, tzinfo=timezone.utc
    )


def test_metadata_captures_source_fields_without_the_body() -> None:
    payload = make_report(
        structured_scope={"asset_identifier": "https://curl.se"}
    )

    report = HackerOneAdapter().normalize(payload)

    assert report.metadata["adapter"] == "hackerone"
    assert report.metadata["severity"] == "high"
    assert report.metadata["state"] == "triaged"
    assert report.metadata["weakness"] == "CWE-122"
    assert report.metadata["bounty_amount"] == "500.00"
    assert report.metadata["team"] == "curl"
    assert report.metadata["asset"] == "https://curl.se"
    assert "vulnerability_information" not in report.metadata


def test_invalid_json_raises_intake_error() -> None:
    with pytest.raises(IntakeError, match="not valid JSON"):
        HackerOneAdapter().normalize("{not json}")


def test_non_object_json_raises_intake_error() -> None:
    with pytest.raises(IntakeError, match="JSON array payload contains 3 reports"):
        HackerOneAdapter().normalize("[1, 2, 3]")


def test_adapter_satisfies_intake_protocol() -> None:
    assert isinstance(HackerOneAdapter(), IntakeAdapter)
