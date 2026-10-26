"""Intake adapter for raw Markdown pastes and security mailing-list email.

Mailing-list reports are usually Markdown inside a message, so one adapter
covers both.  The payload is sniffed for an RFC 5322 header block; message
input yields ``ReportSource.EMAIL`` while everything else stays
``ReportSource.MARKDOWN``.

The adapter is deliberately conservative about rewriting text: it normalizes
line endings and extracts labelled fields, but the report body is preserved.
``NormalizedReport`` already strips surrounding whitespace, and the body is the
basis of both ``content_hash`` and the derived ``report_id``, so mangling it
would break deduplication.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from datetime import datetime
from email import policy
from email.message import EmailMessage
from email.parser import Parser
from html import unescape
from typing import Any

from triage0.core.models import NormalizedReport, ReportSource
from triage0.intake._common import (
    IntakeError,
    clean_value,
    decode_text,
    first_line,
    derive_report_id,
    labelled_value,
    normalize_newlines,
    parse_timestamp,
)

__all__ = ["MarkdownAdapter"]

_HEADING_RE = re.compile(
    r"^#{1,6}[ \t]+(?P<title>.+?)[ \t]*#*[ \t]*$", re.MULTILINE
)

_HEADER_RE = re.compile(
    r"^(?P<name>[A-Za-z][A-Za-z0-9-]*):[ \t]*(?P<value>.*)$"
)

# Headers that only occur in a mail message.  Requiring two of them keeps a
# Markdown document that happens to start with ``Title:`` from being parsed as
# email.
_EMAIL_HEADERS = frozenset(
    {
        "cc",
        "date",
        "delivered-to",
        "from",
        "message-id",
        "received",
        "reply-to",
        "return-path",
        "subject",
        "to",
    }
)

_PROJECT_LABELS = ("project", "repository", "repo", "product", "component")
_VERSION_LABELS = (
    "affected version",
    "affected versions",
    "versions affected",
    "product version",
    "software version",
    "version",
    "release",
)
_DATE_LABELS = (
    "reported at",
    "submitted at",
    "date",
    "reported",
    "submitted",
)

_GITHUB_REPO_RE = re.compile(
    r"https?://(?:www\.)?github\.com/"
    r"(?P<slug>[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+)",
    re.IGNORECASE,
)


def _first_heading(text: str) -> str | None:
    match = _HEADING_RE.search(text)
    if match is None:
        return None
    title = clean_value(match.group("title"))
    return title or None


def _date_from_text(text: str) -> datetime | None:
    value = labelled_value(_DATE_LABELS, text)
    return parse_timestamp(value) if value is not None else None


def _github_project(text: str) -> str | None:
    match = _GITHUB_REPO_RE.search(text)
    if match is None:
        return None
    slug = match.group("slug")
    return slug[: -len(".git")] if slug.lower().endswith(".git") else slug


def _looks_like_email(text: str) -> bool:
    """Whether ``text`` opens with an RFC 5322 header block.

    Every line before the first blank line must be a header or a continuation
    of one, and at least two known mail headers must be present.
    """

    head, separator, _ = text.partition("\n\n")
    if not separator:
        return False

    lines = head.splitlines()
    if not lines or _HEADER_RE.match(lines[0]) is None:
        return False

    names: set[str] = set()
    for line in lines:
        if line[:1] in (" ", "\t"):
            continue
        match = _HEADER_RE.match(line)
        if match is None:
            return False
        names.add(match.group("name").lower())
    return len(names & _EMAIL_HEADERS) >= 2


def _parse_email(text: str) -> EmailMessage | None:
    if not _looks_like_email(text):
        return None
    message = Parser(policy=policy.default).parsestr(text)
    return message if isinstance(message, EmailMessage) else None


def _part_text(part: EmailMessage) -> str:
    """Decode one MIME part to text, falling back to raw bytes."""

    try:
        content = part.get_content()
    except (LookupError, UnicodeDecodeError):
        content = part.get_payload(decode=True)
    if isinstance(content, bytes):
        return content.decode("utf-8", errors="replace")
    return content if isinstance(content, str) else ""


def _strip_html(text: str) -> str:
    """Crude HTML-to-text for reports that only ship an HTML part.

    Removes ``<script>``, ``<style>`` and HTML comment blocks entirely,
    then strips remaining tags.
    """

    # Remove non-visible blocks before stripping tags; their contents
    # should not leak into the report body or hash.
    text = re.sub(r"<script[^>]*>.*?</script>", " ", text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"<style[^>]*>.*?</style>", " ", text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"<!--.*?-->", " ", text, flags=re.DOTALL)

    without_tags = re.sub(r"<[^>]+>", " ", text)
    return re.sub(r"[ \t]{2,}", " ", unescape(without_tags))


def _is_attachment(part: EmailMessage) -> bool:
    disp = part.get_content_disposition()
    return disp is not None and disp.startswith("attachment")


def _email_body(message: EmailMessage) -> tuple[str, list[str]]:
    """Return the best text rendering of an email message and attachment filenames.

    ``text/plain`` is preferred over ``text/html`` so Markdown survives intact;
    HTML is tag-stripped only when nothing else is available.  Parts with
    ``Content-Disposition: attachment`` are excluded from the body.
    """

    attachments: list[str] = []

    if not message.is_multipart():
        if _is_attachment(message):
            attachments.append(
                message.get_filename() or "attachment"
            )
            return "", attachments
        text = _part_text(message)
        if message.get_content_type() == "text/html":
            return _strip_html(text), attachments
        return text, attachments

    inline_plain: list[str] = []
    inline_html: list[str] = []
    for part in message.walk():
        if part.is_multipart():
            continue
        if _is_attachment(part):
            name = part.get_filename()
            if name is not None:
                attachments.append(str(name))
            continue
        content_type = part.get_content_type()
        if content_type == "text/plain":
            inline_plain.append(_part_text(part))
        elif content_type == "text/html":
            inline_html.append(_strip_html(_part_text(part)))
    body_text = "\n\n".join(
        part for part in (inline_plain or inline_html) if part.strip()
    ) if (inline_plain or inline_html) else ""
    return body_text, attachments


class MarkdownAdapter:
    """Normalize raw Markdown, or an email containing a report.

    ``source`` is declared as :attr:`ReportSource.MARKDOWN`, the adapter's
    primary format.  Mail input returns ``ReportSource.EMAIL`` so downstream
    consumers can distinguish the two; the registry maps both sources to this
    class.
    """

    source = ReportSource.MARKDOWN

    def normalize(
        self,
        payload: str | bytes | Mapping[str, Any],
        *,
        report_id: str | None = None,
        forced_source: ReportSource | None = None,
    ) -> NormalizedReport:
        if isinstance(payload, Mapping):
            raise IntakeError(
                "markdown intake expects text or bytes, not a mapping; use "
                "the HackerOne or GitHub adapter for JSON payloads"
            )

        text = normalize_newlines(decode_text(payload))

        metadata: dict[str, Any] = {"adapter": "markdown"}
        if forced_source is not None:
            # Explicit source override from --source; skip auto-sniffing.
            if forced_source is ReportSource.EMAIL:
                message = _parse_email(text) or None
                source = ReportSource.EMAIL
                metadata["content_type"] = "email"
                if message is not None:
                    body, attachments = _email_body(message)
                    subject = str(message.get("subject") or "").strip()
                    title = (
                        subject
                        or _first_heading(body)
                        or first_line(body)
                    )
                    submitted_at = parse_timestamp(
                        message.get("date")
                    ) or _date_from_text(body)
                    sender = message.get("from")
                    if sender:
                        metadata["email_from"] = str(sender).strip()
                    if attachments:
                        metadata["attachments"] = attachments
                else:
                    # Forced email but no RFC 5322 headers found;
                    # fall back to raw text with email source label.
                    body = text
                    title = _first_heading(text) or first_line(text)
                    submitted_at = _date_from_text(text)
            else:
                source = ReportSource.MARKDOWN
                metadata["content_type"] = "markdown"
                message = None
                body = text
                title = _first_heading(text) or first_line(text)
                submitted_at = _date_from_text(text)
        else:
            message = _parse_email(text)
            if message is None:
                source = ReportSource.MARKDOWN
                metadata["content_type"] = "markdown"
                body = text
                title = _first_heading(text) or first_line(text)
                submitted_at = _date_from_text(text)
            else:
                source = ReportSource.EMAIL
                metadata["content_type"] = "email"
                subject = str(message.get("subject") or "").strip()
                body, attachments = _email_body(message)
                title = (
                    subject
                    or _first_heading(body)
                    or first_line(body)
                )
                submitted_at = parse_timestamp(
                    message.get("date")
                ) or _date_from_text(body)
                sender = message.get("from")
                if sender:
                    metadata["email_from"] = str(sender).strip()
                if attachments:
                    metadata["attachments"] = attachments

        if not body.strip():
            raise IntakeError("markdown payload contains no report text")

        return NormalizedReport(
            report_id=report_id or derive_report_id("md", body),
            source=source,
            project=labelled_value(_PROJECT_LABELS, body)
            or _github_project(body),
            title=title,
            body=body,
            reported_version=labelled_value(_VERSION_LABELS, body),
            submitted_at=submitted_at,
            metadata=metadata,
        )
