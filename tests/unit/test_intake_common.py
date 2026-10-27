"""Unit tests for the shared intake helpers."""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from triage0.intake._common import (
    IntakeError,
    decode_json,
    decode_text,
    derive_report_id,
    json_safe,
    normalize_newlines,
    parse_timestamp,
)


def test_decode_text_strips_utf8_bom() -> None:
    assert decode_text(b"\xef\xbb\xbf# Title") == "# Title"


def test_decode_text_keeps_plain_text_untouched() -> None:
    assert decode_text("  leading and trailing  ") == "  leading and trailing  "


def test_decode_text_rejects_invalid_utf8() -> None:
    with pytest.raises(IntakeError, match="not valid UTF-8"):
        decode_text(b"\xff\xfe\x00")


def test_decode_text_rejects_non_text() -> None:
    with pytest.raises(IntakeError, match="expected text or bytes"):
        decode_text(42)  # type: ignore[arg-type]


def test_decode_json_accepts_text_bytes_and_mapping() -> None:
    payload = {"id": 7, "title": "Report"}
    assert decode_json(payload) == payload
    assert decode_json(json.dumps(payload)) == payload
    assert decode_json(json.dumps(payload).encode("utf-8")) == payload


def test_decode_json_reports_position_on_invalid_json() -> None:
    with pytest.raises(IntakeError, match=r"not valid JSON \(.*line 1"):
        decode_json('{"id": }')


def test_decode_json_allows_single_element_array() -> None:
    assert decode_json('[{"id": 1}]') == {"id": 1}


def test_decode_json_rejects_multi_element_array() -> None:
    with pytest.raises(IntakeError, match="JSON array payload contains 3 reports"):
        decode_json("[1, 2, 3]")


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (
            "2026-10-05T14:30:00Z",
            datetime(2026, 10, 5, 14, 30, tzinfo=timezone.utc),
        ),
        (
            "2026-10-05T14:30:00z",
            datetime(2026, 10, 5, 14, 30, tzinfo=timezone.utc),
        ),
        (
            "2026-10-05T16:30:00+02:00",
            datetime(2026, 10, 5, 14, 30, tzinfo=timezone.utc),
        ),
        (
            "Mon, 5 Oct 2026 14:30:00 +0000",
            datetime(2026, 10, 5, 14, 30, tzinfo=timezone.utc),
        ),
        (0, datetime(1970, 1, 1, tzinfo=timezone.utc)),
    ],
)
def test_parse_timestamp_understands_common_formats(
    value: object, expected: datetime
) -> None:
    parsed = parse_timestamp(value)
    assert parsed is not None
    assert parsed == expected
    assert parsed.utcoffset() == timezone.utc.utcoffset(None)


def test_parse_timestamp_attaches_utc_to_naive_values() -> None:
    parsed = parse_timestamp("2026-10-05T14:30:00")
    assert parsed is not None
    assert parsed.tzinfo is timezone.utc


def test_parse_timestamp_keeps_aware_datetime() -> None:
    aware = datetime(2026, 10, 5, 14, 30, tzinfo=timezone.utc)
    assert parse_timestamp(aware) is aware


@pytest.mark.parametrize(
    "value",
    [None, True, False, "", "   ", "not a date", "2026-13-40T99:00:00", object()],
)
def test_parse_timestamp_returns_none_for_unusable_values(
    value: object,
) -> None:
    assert parse_timestamp(value) is None


def test_derive_report_id_is_prefix_scoped_and_stable() -> None:
    first = derive_report_id("md", "  report body  \r\n")
    second = derive_report_id("md", "report body")

    assert first == second
    assert first.startswith("md-")
    assert len(first) == len("md-") + 12


def test_derive_report_id_changes_with_body() -> None:
    assert derive_report_id("md", "one") != derive_report_id("md", "two")


def test_derive_report_id_changes_with_prefix() -> None:
    assert derive_report_id("md", "body") != derive_report_id("h1", "body")


def test_normalize_newlines_handles_crlf_and_cr() -> None:
    assert normalize_newlines("a\r\nb\rc\nd") == "a\nb\nc\nd"


def test_normalize_newlines_leaves_other_whitespace_alone() -> None:
    assert normalize_newlines("  a\tb  ") == "  a\tb  "


def test_json_safe_coerces_nested_values() -> None:
    payload = {
        "id": 1,
        "flags": (True, False),
        "nested": {"when": object()},
        "items": [1, "two", None],
    }

    assert json_safe(payload) == {
        "id": 1,
        "flags": [True, False],
        "nested": {"when": str(payload["nested"]["when"])},
        "items": [1, "two", None],
    }


def test_json_safe_output_round_trips_through_json() -> None:
    payload = {"nested": {"value": object()}, "keys": (1, 2)}

    rendered = json.dumps(json_safe(payload))

    assert json.loads(rendered) == json_safe(payload)
