"""Tests for public HackerOne report acquisition and record writing."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from triage0.storage.indexers import h1_reports
from triage0.storage.indexers.h1_reports import (
    EXPECTED_SLOP_COUNT,
    FetchPolicy,
    H1Error,
    build_fake_record,
    fetch_public_json,
    fetch_public_report,
    parse_slop_list,
    report_text,
    write_json_record,
)


class FakeResponse:
    def __init__(
        self,
        status: int,
        body: bytes = b"",
        headers: dict[str, str] | None = None,
    ) -> None:
        self.status = status
        self._body = body
        self._headers = headers or {}

    def read(self, amount: int) -> bytes:
        return self._body[:amount]

    def getheader(self, name: str) -> str | None:
        return self._headers.get(name)


def install_responses(
    monkeypatch: pytest.MonkeyPatch, *responses: FakeResponse
) -> list[tuple[str, str]]:
    """Replace HTTPSConnection with a queue of canned responses."""

    queue = list(responses)
    calls: list[tuple[str, str]] = []

    class FakeConnection:
        def __init__(
            self, host: str, port: int | None = None, timeout: float | None = None
        ) -> None:
            self.host = host

        def request(
            self, method: str, target: str, headers: dict[str, str] | None = None
        ) -> None:
            calls.append((self.host, target))

        def getresponse(self) -> FakeResponse:
            if not queue:
                raise AssertionError("unexpected HTTP request")
            return queue.pop(0)

        def close(self) -> None:
            return None

    monkeypatch.setattr(h1_reports, "HTTPSConnection", FakeConnection)
    return calls


def payload(report_id: int = 2199174, **overrides: Any) -> bytes:
    data: dict[str, Any] = {
        "id": report_id,
        "title": "Buffer Overflow Vulnerability in WebSocket Handling",
        "vulnerability_information": "curl 8.13.0 is affected.\n\nDetails.",
        "public": True,
        "team": {"handle": "curl"},
        "state": "Closed",
        "substate": "not-applicable",
        "severity_rating": "none",
        "weakness": None,
        "created_at": "2026-01-01T00:00:00Z",
        "disclosed_at": "2026-01-02T00:00:00Z",
        "cve_ids": [],
        "reporter": {"username": "researcher"},
        "attachments": [{"file_name": "proof.png"}],
        "has_bounty?": True,
        "bounty_amount": "500.00",
        "voters": [{"username": "someone"}],
    }
    data.update(overrides)
    return json.dumps(data).encode("utf-8")


def gist_with(ids: list[int]) -> str:
    lines = ["# Slop", "", "## Reports", ""]
    for position, report_id in enumerate(ids, 1):
        lines.append(
            f"{position}. A report [#{report_id}](https://hackerone.com/reports/{report_id})"
        )
    return "\n".join(lines)


# --- slop list -------------------------------------------------------------


def test_parse_slop_list_returns_49_unique_ids() -> None:
    ids = list(range(1000, 1000 + EXPECTED_SLOP_COUNT))
    assert parse_slop_list(gist_with(ids)) == ids


def test_parse_slop_list_rejects_count_drift() -> None:
    ids = list(range(1000, 1000 + EXPECTED_SLOP_COUNT - 1))
    with pytest.raises(H1Error, match="expected 49"):
        parse_slop_list(gist_with(ids))


def test_parse_slop_list_rejects_duplicate_links() -> None:
    ids = list(range(1000, 1000 + EXPECTED_SLOP_COUNT))
    text = gist_with(ids) + "\n[again](https://hackerone.com/reports/1000)"
    with pytest.raises(H1Error, match="repeats report ids"):
        parse_slop_list(text)


# --- report fetch ----------------------------------------------------------


def test_fetch_public_report_validates_and_strips_private_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = install_responses(monkeypatch, FakeResponse(200, payload()))
    report = fetch_public_report(2199174, FetchPolicy())

    assert calls == [("hackerone.com", "/reports/2199174.json")]
    assert report.id == 2199174
    assert report.team_handle == "curl"
    assert report.body.startswith("curl 8.13.0")
    assert report.provenance() == {
        "id": 2199174,
        "state": "Closed",
        "substate": "not-applicable",
        "severity_rating": "none",
        "weakness": None,
        "created_at": "2026-01-01T00:00:00Z",
        "disclosed_at": "2026-01-02T00:00:00Z",
        "public": True,
        "cve_ids": [],
    }


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"public": False}, "not public"),
        ({"team": {"handle": "another-project"}}, "not in the curl program"),
        ({"vulnerability_information": "   "}, "empty body"),
        ({"id": 42}, "different id"),
    ],
)
def test_fetch_public_report_rejects_unusable_reports(
    monkeypatch: pytest.MonkeyPatch, overrides: dict[str, Any], message: str
) -> None:
    install_responses(monkeypatch, FakeResponse(200, payload(**overrides)))
    with pytest.raises(H1Error, match=message):
        fetch_public_report(2199174, FetchPolicy())


@pytest.mark.parametrize("report_id", [0, -1, True])
def test_fetch_public_report_rejects_invalid_ids(report_id: Any) -> None:
    with pytest.raises(H1Error, match="invalid HackerOne report id"):
        fetch_public_report(report_id, FetchPolicy())


def test_fetch_public_report_keeps_the_cwe_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_responses(
        monkeypatch,
        FakeResponse(
            200,
            payload(weakness={"id": 18, "name": "Information Disclosure"}),
        ),
    )
    report = fetch_public_report(2199174, FetchPolicy())

    assert report.weakness == "Information Disclosure"


def test_fetch_public_json_rejects_non_https() -> None:
    with pytest.raises(H1Error, match="non-HTTPS"):
        fetch_public_json(
            "http://hackerone.com/reports/1.json",
            FetchPolicy(),
            allowed_hosts=h1_reports.H1_HOSTS,
        )


def test_fetch_public_json_rejects_unlisted_host() -> None:
    with pytest.raises(H1Error, match="refusing host"):
        fetch_public_json(
            "https://example.test/reports/1.json",
            FetchPolicy(),
            allowed_hosts=h1_reports.H1_HOSTS,
        )


def test_fetch_public_json_rejects_a_nonstandard_port(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    connections: list[str] = []

    class Recorder:
        def __init__(self, *args: object, **kwargs: object) -> None:
            connections.append("connected")

    monkeypatch.setattr(h1_reports, "HTTPSConnection", Recorder)

    with pytest.raises(H1Error, match="non-standard HTTPS port"):
        fetch_public_json(
            "https://hackerone.com:8443/reports/1.json",
            FetchPolicy(),
            allowed_hosts=h1_reports.H1_HOSTS,
        )
    assert connections == []


def test_fetch_public_json_rejects_embedded_credentials() -> None:
    with pytest.raises(H1Error, match="embedded credentials"):
        fetch_public_json(
            "https://user:secret@hackerone.com/reports/1.json",
            FetchPolicy(),
            allowed_hosts=h1_reports.H1_HOSTS,
        )


def test_fetch_public_json_converts_transport_errors_to_h1error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempts: list[str] = []

    def fail(url: str, *args: object, **kwargs: object) -> bytes:
        attempts.append(url)
        raise OSError("temporary DNS failure")

    monkeypatch.setattr(h1_reports, "_https_get", fail)
    monkeypatch.setattr(h1_reports.time, "sleep", lambda _: None)

    with pytest.raises(H1Error, match="giving up"):
        fetch_public_json(
            "https://hackerone.com/reports/1.json",
            FetchPolicy(retries=2, delay_seconds=0),
            allowed_hosts=h1_reports.H1_HOSTS,
        )

    assert len(attempts) == 3


def test_fetch_public_json_refuses_redirects(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_responses(monkeypatch, FakeResponse(302, headers={"Location": "https://x"}))
    with pytest.raises(H1Error, match="refusing redirect"):
        fetch_public_json(
            "https://hackerone.com/reports/1.json",
            FetchPolicy(),
            allowed_hosts=h1_reports.H1_HOSTS,
        )


def test_fetch_public_json_enforces_size_cap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_responses(monkeypatch, FakeResponse(200, b"x" * 11))
    with pytest.raises(H1Error, match="exceeds 10 bytes"):
        fetch_public_json(
            "https://hackerone.com/reports/1.json",
            FetchPolicy(max_bytes=10),
            allowed_hosts=h1_reports.H1_HOSTS,
        )


def test_fetch_public_json_rejects_client_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_responses(monkeypatch, FakeResponse(404))
    with pytest.raises(H1Error, match="HTTP 404"):
        fetch_public_json(
            "https://hackerone.com/reports/1.json",
            FetchPolicy(),
            allowed_hosts=h1_reports.H1_HOSTS,
        )


def test_fetch_public_json_retries_429_with_retry_after(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_responses(
        monkeypatch,
        FakeResponse(429, headers={"Retry-After": "2"}),
        FakeResponse(200, b'{"ok": true}'),
    )
    sleeps: list[float] = []
    monkeypatch.setattr(h1_reports.time, "sleep", sleeps.append)

    assert fetch_public_json(
        "https://hackerone.com/reports/1.json",
        FetchPolicy(retries=1),
        allowed_hosts=h1_reports.H1_HOSTS,
    ) == {"ok": True}
    assert sleeps == [2.0]


def test_fetch_public_json_ignores_non_finite_retry_after(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_responses(
        monkeypatch,
        FakeResponse(429, headers={"Retry-After": "nan"}),
        FakeResponse(200, b'{"ok": true}'),
    )
    sleeps: list[float] = []
    monkeypatch.setattr(h1_reports.time, "sleep", sleeps.append)

    assert fetch_public_json(
        "https://hackerone.com/reports/1.json",
        FetchPolicy(retries=1, delay_seconds=1.2),
        allowed_hosts=h1_reports.H1_HOSTS,
    ) == {"ok": True}
    assert sleeps == [1.2]


def test_fetch_public_json_gives_up_after_retries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_responses(
        monkeypatch,
        FakeResponse(503),
        FakeResponse(503),
    )
    monkeypatch.setattr(h1_reports.time, "sleep", lambda _: None)

    with pytest.raises(H1Error, match="giving up"):
        fetch_public_json(
            "https://hackerone.com/reports/1.json",
            FetchPolicy(retries=1),
            allowed_hosts=h1_reports.H1_HOSTS,
        )


def test_fetch_public_json_rejects_invalid_json(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_responses(monkeypatch, FakeResponse(200, b"not json"))
    with pytest.raises(H1Error, match="invalid JSON"):
        fetch_public_json(
            "https://hackerone.com/reports/1.json",
            FetchPolicy(),
            allowed_hosts=h1_reports.H1_HOSTS,
        )


# --- record assembly -------------------------------------------------------


def test_report_text_prepends_title_once() -> None:
    assert report_text("A bug", "Details.") == "A bug\n\nDetails."
    assert report_text("A bug", "A bug\n\nDetails.") == "A bug\n\nDetails."


def test_report_text_does_not_treat_a_word_prefix_as_the_title() -> None:
    assert (
        report_text("A bug", "A buggy parser crashes.")
        == "A bug\n\nA buggy parser crashes."
    )


def test_report_text_without_a_title_is_just_the_body() -> None:
    assert report_text("   ", "Details.") == "Details."


def test_report_text_rejects_an_empty_body() -> None:
    with pytest.raises(H1Error, match="body is empty"):
        report_text("A bug", "   ")


def test_build_fake_record_is_flat_and_has_no_private_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_responses(monkeypatch, FakeResponse(200, payload()))
    report = fetch_public_report(2199174, FetchPolicy())

    record = build_fake_record(
        report,
        tag="curl-8_13_0",
        commit="a" * 40,
        reported_version="8.13.0",
    )

    assert record["id"] == "h1-2199174"
    assert record["case_id"] == record["id"]
    assert record["ground_truth"] == "fake"
    assert isinstance(record["report"], str)
    assert record["report"].startswith(report.title)
    assert record["tag"] == "curl-8_13_0"
    assert record["commit"] == "a" * 40
    assert record["reported_version"] == "8.13.0"
    assert record["provenance"]["ref_source"] == "report_version"

    serialized = json.dumps(record)
    for forbidden in ("reporter", "attachments", "bounty_amount", "voters"):
        assert f'"{forbidden}"' not in serialized


def test_build_fake_record_requires_tag_and_commit_together(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_responses(monkeypatch, FakeResponse(200, payload()))
    report = fetch_public_report(2199174, FetchPolicy())

    with pytest.raises(H1Error, match="tag and commit"):
        build_fake_record(
            report, tag="curl-8_13_0", commit=None, reported_version="8.13.0"
        )
    with pytest.raises(H1Error, match="tag and commit"):
        build_fake_record(
            report, tag=None, commit="a" * 40, reported_version=None
        )


def test_build_fake_record_without_ref_stays_unresolved(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_responses(monkeypatch, FakeResponse(200, payload()))
    report = fetch_public_report(2199174, FetchPolicy())

    record = build_fake_record(
        report, tag=None, commit=None, reported_version=None
    )

    assert record["tag"] is None
    assert record["commit"] is None
    assert record["reported_version"] is None
    assert record["provenance"]["ref_source"] == "unresolved"
    assert record["provenance"]["ref_verified"] is None


def test_build_fake_record_does_not_claim_an_unverified_ref(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_responses(monkeypatch, FakeResponse(200, payload()))
    report = fetch_public_report(2199174, FetchPolicy())

    record = build_fake_record(
        report,
        tag="curl-8_13_0",
        commit="a" * 40,
        reported_version="8.13.0",
    )

    assert record["provenance"]["ref_verified"] is None
    assert record["provenance"]["commit_verified"] is None


# --- writer ----------------------------------------------------------------


def test_write_json_record_is_deterministic(tmp_path: Path) -> None:
    record = {"id": "h1-1", "b": 2, "a": 1, "report": "line\nline"}
    path = write_json_record(tmp_path, record)
    first = path.read_bytes()

    assert path.name == "h1-1.json"
    assert first.endswith(b"\n")
    assert b"\r\n" not in first
    assert first.index(b'"a"') < first.index(b'"b"')
    assert json.loads(first) == record

    write_json_record(tmp_path, record)
    assert path.read_bytes() == first


def test_write_json_record_leaves_no_temporary_file(tmp_path: Path) -> None:
    path = write_json_record(tmp_path, {"id": "h1-1", "report": "x"})

    assert path.exists()
    assert list(tmp_path.glob("*.tmp")) == []


def test_write_json_record_rejects_path_traversal(tmp_path: Path) -> None:
    with pytest.raises(H1Error, match="invalid record id"):
        write_json_record(tmp_path, {"id": "../escape"})
