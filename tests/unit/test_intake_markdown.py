"""Unit tests for the Markdown and email intake adapter."""

from __future__ import annotations

from datetime import datetime, timezone
from email.message import EmailMessage

import pytest

from triage0.core.interfaces import IntakeAdapter
from triage0.core.models import NormalizedReport, ReportSource
from triage0.intake import MarkdownAdapter
from triage0.intake._common import IntakeError

MARKDOWN_REPORT = """# Heap overflow in curl_easy_escape

Project: curl
Affected version: 8.13.0

Calling `curl_easy_escape` with a very long string overflows the heap.
"""


def email_with_body(body: str, **headers: str) -> str:
    lines = [f"{name.title()}: {value}" for name, value in headers.items()]
    lines.append('Content-Type: text/plain; charset="utf-8"')
    return "\n".join(lines) + "\n\n" + body


def multipart_email() -> str:
    message = EmailMessage()
    message["From"] = "Researcher <researcher@example.com>"
    message["Subject"] = "Multipart report"
    message["Date"] = "Mon, 5 Oct 2026 14:30:00 +0000"
    message.set_content("Project: curl\n\nPlain text body.\n")
    message.add_alternative(
        "<html><body><p>HTML body.</p></body></html>", subtype="html"
    )
    return message.as_string()


def html_only_email() -> str:
    message = EmailMessage()
    message["From"] = "researcher@example.com"
    message["Subject"] = "HTML report"
    message["Date"] = "Mon, 5 Oct 2026 14:30:00 +0000"
    message.set_content(
        "<html><body><p>Heap <b>overflow</b> &amp; more</p></body></html>",
        subtype="html",
    )
    return message.as_string()


def test_adapter_declares_markdown_source() -> None:
    assert MarkdownAdapter().source is ReportSource.MARKDOWN


def test_markdown_normalizes_heading_project_and_version() -> None:
    report = MarkdownAdapter().normalize(MARKDOWN_REPORT)

    assert isinstance(report, NormalizedReport)
    assert report.source is ReportSource.MARKDOWN
    assert report.title == "Heap overflow in curl_easy_escape"
    assert report.project == "curl"
    assert report.reported_version == "8.13.0"
    assert "overflows the heap" in report.body
    assert report.metadata["content_type"] == "markdown"


def test_title_falls_back_to_first_line() -> None:
    report = MarkdownAdapter().normalize(
        "Heap overflow in libfoo\n\nDetails follow.\n"
    )

    assert report.title == "Heap overflow in libfoo"


def test_project_falls_back_to_github_url() -> None:
    report = MarkdownAdapter().normalize(
        "# Report\n\nSee https://github.com/curl/curl.git for the source.\n"
    )

    assert report.project == "curl/curl"


def test_absent_version_and_project_are_none() -> None:
    report = MarkdownAdapter().normalize("# Report\n\nNo metadata here.\n")

    assert report.project is None
    assert report.reported_version is None
    assert report.submitted_at is None


def test_bold_labels_and_values_are_cleaned() -> None:
    report = MarkdownAdapter().normalize(
        "# Report\n\n"
        "**Project:** **curl**\n"
        "**Affected version:** **8.13.0**\n"
        "**Date:** 2026-10-05\n"
    )

    assert report.project == "curl"
    assert report.reported_version == "8.13.0"
    assert report.submitted_at == datetime(
        2026, 10, 5, tzinfo=timezone.utc
    )


def test_bulleted_labels_are_recognized() -> None:
    report = MarkdownAdapter().normalize(
        "# Report\n\n- Project: curl\n- Version: 8.13.0\n"
    )

    assert report.project == "curl"
    assert report.reported_version == "8.13.0"


def test_report_id_is_derived_and_stable_across_whitespace() -> None:
    adapter = MarkdownAdapter()

    first = adapter.normalize("# Report\n\nBody text\n")
    second = adapter.normalize("# Report\n\nBody text")

    assert first.report_id == second.report_id
    assert first.report_id.startswith("md-")
    assert first.content_hash == second.content_hash


def test_crlf_and_lf_payloads_share_content_hash() -> None:
    adapter = MarkdownAdapter()

    unix = adapter.normalize("# Report\n\nBody text\n")
    windows = adapter.normalize("# Report\r\n\r\nBody text\r\n")

    assert windows.content_hash == unix.content_hash
    assert windows.report_id == unix.report_id


def test_report_id_can_be_overridden() -> None:
    report = MarkdownAdapter().normalize(MARKDOWN_REPORT, report_id="report-42")

    assert report.report_id == "report-42"


def test_bytes_payload_with_bom_is_accepted() -> None:
    payload = b"\xef\xbb\xbf# Report\n\nBody text\n"

    report = MarkdownAdapter().normalize(payload)

    assert report.title == "Report"
    assert "Body text" in report.body


def test_blank_payload_raises() -> None:
    with pytest.raises(IntakeError, match="no report text"):
        MarkdownAdapter().normalize("   \n\n\t\n")


def test_mapping_payload_is_rejected() -> None:
    with pytest.raises(IntakeError, match="not a mapping"):
        MarkdownAdapter().normalize({"body": "text"})


def test_single_header_document_is_not_treated_as_email() -> None:
    report = MarkdownAdapter().normalize("Subject: not mail\n\nBody text\n")

    assert report.source is ReportSource.MARKDOWN


def test_markdown_heading_prevents_email_detection() -> None:
    report = MarkdownAdapter().normalize("# Title\n\nFrom: nobody\n\nBody\n")

    assert report.source is ReportSource.MARKDOWN


def test_email_source_subject_and_headers() -> None:
    payload = email_with_body(
        "Hi team,\n\nProject: curl\n\nDetails here.\n",
        **{
            "from": "Researcher <researcher@example.com>",
            "to": "security@example.org",
            "subject": "Heap overflow in curl_easy_escape",
            "date": "Mon, 5 Oct 2026 14:30:00 +0000",
        },
    )

    report = MarkdownAdapter().normalize(payload)

    assert report.source is ReportSource.EMAIL
    assert report.title == "Heap overflow in curl_easy_escape"
    assert report.project == "curl"
    assert report.submitted_at == datetime(
        2026, 10, 5, 14, 30, tzinfo=timezone.utc
    )
    assert "From:" not in report.body
    assert report.metadata["content_type"] == "email"
    assert "researcher@example.com" in report.metadata["email_from"]


def test_email_date_offset_is_normalized_to_utc() -> None:
    payload = email_with_body(
        "Body text\n",
        **{
            "from": "researcher@example.com",
            "subject": "Report",
            "date": "Mon, 5 Oct 2026 16:30:00 +0200",
        },
    )

    report = MarkdownAdapter().normalize(payload)

    assert report.submitted_at == datetime(
        2026, 10, 5, 14, 30, tzinfo=timezone.utc
    )
    assert report.submitted_at is not None
    assert report.submitted_at.utcoffset() == timezone.utc.utcoffset(None)


def test_email_falls_back_to_body_date_without_header() -> None:
    payload = email_with_body(
        "Date: 2026-10-05\n\nBody text\n",
        **{
            "from": "researcher@example.com",
            "subject": "Report",
            "message-id": "<abc@example.com>",
        },
    )

    report = MarkdownAdapter().normalize(payload)

    assert report.submitted_at == datetime(
        2026, 10, 5, tzinfo=timezone.utc
    )


def test_email_title_falls_back_to_heading() -> None:
    payload = email_with_body(
        "# Heading From Body\n\nBody text\n",
        **{
            "from": "researcher@example.com",
            "date": "Mon, 5 Oct 2026 14:30:00 +0000",
        },
    )

    report = MarkdownAdapter().normalize(payload)

    assert report.title == "Heading From Body"


def test_multipart_email_prefers_plain_text() -> None:
    report = MarkdownAdapter().normalize(multipart_email())

    assert report.source is ReportSource.EMAIL
    assert "Plain text body." in report.body
    assert "HTML body." not in report.body
    assert report.project == "curl"


def test_html_only_email_is_stripped() -> None:
    report = MarkdownAdapter().normalize(html_only_email())

    assert report.source is ReportSource.EMAIL
    assert "Heap overflow" in report.body
    assert "&amp;" not in report.body
    assert "<p>" not in report.body


def test_email_body_can_supply_project_and_version() -> None:
    payload = email_with_body(
        "Project: openssl\nVersion: 3.0.1\n\nDetails.\n",
        **{
            "from": "researcher@example.com",
            "subject": "Report",
            "date": "Mon, 5 Oct 2026 14:30:00 +0000",
        },
    )

    report = MarkdownAdapter().normalize(payload)

    assert report.project == "openssl"
    assert report.reported_version == "3.0.1"


def test_adapter_satisfies_intake_protocol() -> None:
    assert isinstance(MarkdownAdapter(), IntakeAdapter)
